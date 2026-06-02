"""Laplace solver for the magnetic scalar potential.

We solve
    laplacian(Phi) = 0           in Omega (the air domain)
    n . grad(Phi)  = -n . B_dip  on the SC surface     (Neumann)
    Phi            = 0           on the far-field      (Dirichlet)

using scikit-fem's quadratic Lagrange tetrahedral element.

The induced field on the dipole equals B_ind(r0) = grad(Phi)(r0); we evaluate
it via a centred finite difference of Phi (the P2 interpolant is C^0 across
element faces, so direct gradient evaluation is element-local; FD on a small
stencil away from boundaries is both cheap and free of element-face jumps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.sparse.linalg import splu

from skfem import (
    Basis,
    FacetBasis,
    ElementTetP2,
    ElementTetP1,
    BilinearForm,
    LinearForm,
    asm,
    condense,
)
from skfem.helpers import grad as skgrad, dot as skdot

from .dipole import B_dipole
from .half_space import (
    B_halfspace_image,
    find_nearest_sc_facet,
)
from .mesh import SCTrapMesh


@dataclass
class PhiSolution:
    """Container for a solved scalar potential.

    Attributes
    ----------
    coeffs : numpy array of degrees of freedom (length = basis.N)
    basis  : the scikit-fem Basis used for the solve

    When the solve was done with `subtract_singularity=True`, the solution
    represents the *residual* potential `Phi_residual` — the analytic
    image-dipole contribution must be added back when reconstructing
    `B_induced`. The fields below carry that information; they are `None`
    for a plain (non-subtracted) solve.
    """

    coeffs: np.ndarray
    basis: Basis
    image_source: tuple | None = None     # (m_source, r0_source, plane_pt, n_pl)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Interpolate Phi at arbitrary points (3, M)."""
        pts = _as_3xN(points)
        return self.basis.interpolator(self.coeffs)(pts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_3xN(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(3, 1)
    elif pts.shape[0] != 3 and pts.shape[-1] == 3:
        pts = pts.T
    return np.ascontiguousarray(pts)


def _B_field_on_quadpts(x_array: np.ndarray, m: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """Evaluate B_dipole at quadrature points laid out as (3, n_facets, n_qp)."""
    shape = x_array.shape  # (3, M, Q)
    flat_pts = x_array.reshape(3, -1).T   # (N, 3)
    B = B_dipole(flat_pts, m, r0)         # (N, 3)
    return B.T.reshape(shape)             # (3, M, Q)


@BilinearForm
def _stiffness(u, v, w):
    return skdot(skgrad(u), skgrad(v))


# ---------------------------------------------------------------------------
# Prepared (cached) Laplace operator
# ---------------------------------------------------------------------------
#
# The stiffness matrix `A`, the Dirichlet/gauge-pin set `D`, and therefore the
# sparse factorisation of the condensed system depend ONLY on the mesh and the
# element — not on the dipole position or moment. Only the Neumann RHS `b`
# (the surface integral of n.B_dipole) changes between evaluations. An
# equilibrium search + 5x5 Hessian calls `solve_phi` dozens to hundreds of
# times on a fixed mesh, so we assemble + factorise once and reuse the
# factorisation, turning every subsequent solve into a cheap back-substitution.
# Results are numerically identical to the previous per-call `spsolve` (which
# is itself SuperLU under the hood); this is a pure performance refactor.

class _PreparedLaplace:
    """One-time assembly + factorisation of the Laplace operator for a mesh."""

    def __init__(self, sctmesh: SCTrapMesh, element: str, sc_intorder: int):
        elem = {"P1": ElementTetP1, "P2": ElementTetP2}[element]()
        mesh = sctmesh.mesh

        self.basis = Basis(mesh, elem)
        # Surface basis for the Neumann RHS (depends only on mesh + element +
        # quadrature order, so it is cached alongside the factorisation).
        self.sc_basis = FacetBasis(mesh, elem, facets=sctmesh.sc_facets,
                                   intorder=sc_intorder)

        A = asm(_stiffness, self.basis)

        # Dirichlet on the far field, or — for a fully enclosed cavity — pin the
        # single interior DOF nearest the bbox centre to fix the gauge of Phi
        # (B_ind = grad(Phi) is gauge-invariant). Both depend only on the mesh.
        if sctmesh.ff_facets is not None and len(sctmesh.ff_facets) > 0:
            D = self.basis.get_dofs(facets=sctmesh.ff_facets)
        else:
            bbox_lo, bbox_hi = sctmesh.bounding_box()
            centre = 0.5 * (bbox_lo + bbox_hi)
            diffs = mesh.p.T - centre
            i_pin = int(np.argmin((diffs * diffs).sum(axis=1)))
            D = np.array([i_pin], dtype=np.int64)

        self.N = self.basis.N
        self.D = D
        # Condense once (with a dummy RHS) to recover the free-DOF set `I` and
        # the condensed matrix, then factorise that matrix a single time.
        dummy = np.zeros(self.N)
        A_c, _, _, self.I = condense(A, dummy, x=dummy, D=D)
        self._lu = splu(A_c.tocsc())

    def solve_rhs(self, b: np.ndarray) -> np.ndarray:
        """Back-solve for the full DOF vector given a Neumann RHS `b`.

        The Dirichlet/pin values are homogeneous (Phi = 0), so the condensed
        RHS is simply `b` restricted to the free DOFs (no lifting term).
        """
        x = np.zeros(self.N)
        x[self.I] = self._lu.solve(b[self.I])
        return x


def _get_prepared(sctmesh: SCTrapMesh, element: str,
                  sc_intorder: int) -> _PreparedLaplace:
    """Return a cached `_PreparedLaplace`, building it once per (mesh, element,
    intorder). The cache lives on the `SCTrapMesh` instance, so it is shared by
    every solve in a run and is naturally discarded when the mesh is."""
    cache = getattr(sctmesh, "_laplace_cache", None)
    if cache is None:
        cache = {}
        sctmesh._laplace_cache = cache
    key = (element, sc_intorder)
    prep = cache.get(key)
    if prep is None:
        prep = _PreparedLaplace(sctmesh, element, sc_intorder)
        cache[key] = prep
    return prep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_phi(
    sctmesh: SCTrapMesh,
    m: np.ndarray,
    r0: np.ndarray,
    element: str = "P2",
    sc_intorder: int = 8,
    subtract_singularity: bool = False,
    B_source: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> PhiSolution:
    """Solve Laplace's equation for the scalar potential Phi.

    Parameters
    ----------
    sctmesh     : SCTrapMesh from `load_msh`
    m           : (3,) magnetic dipole moment [A m^2]
    r0          : (3,) dipole position [m]
    element     : "P1" or "P2" Lagrange element (default P2)
    sc_intorder : surface quadrature order for the Neumann RHS (default 8).
                  The dipole field varies as |r-r0|^-3 on the SC face nearest
                  the dipole; bumping the integration order well above the
                  element order is essential for off-symmetry-point accuracy.
    B_source    : optional callable `B_source(points: (N,3)) -> (N,3)` used
                  to evaluate the inducing source field on the SC facets.
                  When provided, replaces the point-dipole field B_dipole(...)
                  in the Neumann RHS — used for finite-size magnet geometries
                  whose analytic field is known (e.g. via magpylib).
                  Incompatible with `subtract_singularity=True`.

    Returns
    -------
    PhiSolution
    """
    if B_source is not None and subtract_singularity:
        raise ValueError(
            "B_source is incompatible with subtract_singularity=True "
            "(the half-space image construction assumes a point dipole)."
        )
    # Reuse the per-mesh assembly + factorisation (see _PreparedLaplace). The
    # stiffness matrix and far-field/gauge condensation are mesh-only, so they
    # are built once and cached; only the Neumann RHS below depends on (m, r0).
    prep = _get_prepared(sctmesh, element, sc_intorder)
    basis = prep.basis
    sc_basis = prep.sc_basis

    m_arr  = np.asarray(m,  dtype=float).reshape(3)
    r0_arr = np.asarray(r0, dtype=float).reshape(3)

    image_source = None
    if subtract_singularity:
        plane_pt, plane_n, _ = find_nearest_sc_facet(sctmesh, r0_arr)
        image_source = (m_arr.copy(), r0_arr.copy(), plane_pt, plane_n)

    @LinearForm
    def neumann_rhs(v, w):
        # B at every quadrature point: shape (3, M, Q).
        # Default path uses the analytic point-dipole; if the caller passed a
        # `B_source(points)` callable we use that instead (volumetric magnet).
        if B_source is None:
            B = _B_field_on_quadpts(w.x, m_arr, r0_arr)
        else:
            shape = w.x.shape
            flat = w.x.reshape(3, -1).T          # (N, 3)
            B_pts = np.asarray(B_source(flat), dtype=float)
            if B_pts.shape != flat.shape:
                raise ValueError(
                    f"B_source must return shape (N, 3); got {B_pts.shape}"
                )
            B = B_pts.T.reshape(shape)
        if image_source is not None:
            # Add the image-dipole contribution so that the integrand
            # vanishes on the tangent plane and is smooth nearby.
            shape = w.x.shape
            flat = w.x.reshape(3, -1).T
            B_im = B_halfspace_image(
                flat, image_source[0], image_source[1],
                image_source[2], image_source[3],
            )
            B = B + B_im.T.reshape(shape)
        # n . B  (sum over component axis)
        nB = (w.n * B).sum(axis=0)            # (M, Q)
        return -nB * v

    b = asm(neumann_rhs, sc_basis)

    # Back-substitute against the cached factorisation. The boundary condition
    # (far-field Dirichlet or single-DOF gauge pin) is baked into `prep` and is
    # homogeneous, so this reproduces the previous condense + spsolve exactly.
    x = prep.solve_rhs(b)

    return PhiSolution(coeffs=x, basis=basis, image_source=image_source)


def B_induced_at(
    phi: PhiSolution,
    point: np.ndarray,
    h: Optional[float] = None,
) -> np.ndarray:
    """Evaluate B_induced = grad(Phi) at `point` via centred finite difference.

    Parameters
    ----------
    phi   : PhiSolution from `solve_phi`
    point : (3,) location at which to evaluate the induced field [m]
    h     : finite-difference step [m]; default = 1e-4 * mesh diameter
            (small enough for accuracy, large enough to avoid element-jump noise)

    Returns
    -------
    B_ind : (3,) numpy array [T]
    """
    p = np.asarray(point, dtype=float).reshape(3)
    if h is None:
        bbox = phi.basis.mesh.p
        diam = float(np.linalg.norm(bbox.max(axis=1) - bbox.min(axis=1)))
        h = max(1e-4 * diam, 1e-9)

    e = np.eye(3) * h
    pts = np.column_stack([
        p + e[0], p - e[0],
        p + e[1], p - e[1],
        p + e[2], p - e[2],
    ])  # (3, 6)
    vals = phi(pts)                          # (6,)
    g = np.array([
        (vals[0] - vals[1]) / (2.0 * h),
        (vals[2] - vals[3]) / (2.0 * h),
        (vals[4] - vals[5]) / (2.0 * h),
    ])
    # If the solve subtracted the singularity, Phi here represents only the
    # *residual* potential — add back the analytic image-dipole field at p.
    if phi.image_source is not None:
        m_src, r0_src, plane_pt, n_pl = phi.image_source
        B_im = B_halfspace_image(p[None, :], m_src, r0_src, plane_pt, n_pl)[0]
        g = g + B_im
    return g
