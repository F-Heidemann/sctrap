"""Magnetic interaction energy and 1-D parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import G_GRAV, MAGNET_MASS
from .solver import solve_phi, B_induced_at
from .mesh import SCTrapMesh


def U_mag(sctmesh: SCTrapMesh, m: np.ndarray, r0: np.ndarray, **kwargs) -> float:
    """Magnetic trap (self-)energy U_mag = -1/2 m . B_induced(r0).

    The induced field is created *by* the dipole's own image currents, so the
    trap potential is the image self-energy with the standard factor 1/2
    (Jackson Sec. 2.2). Omitting it makes every stiffness 2x and every trap
    frequency a factor sqrt(2) too high.

    Solves the Laplace problem once and evaluates the induced field at r0.
    Extra kwargs are forwarded to `solve_phi` (e.g. ``element="P1"``).
    """
    phi  = solve_phi(sctmesh, m, r0, **kwargs)
    Bind = B_induced_at(phi, r0)
    return -0.5 * float(np.dot(np.asarray(m, dtype=float).reshape(3), Bind))


def U_mag_and_grad(
    sctmesh: SCTrapMesh,
    m_mag: float,
    theta: float,
    phi: float,
    r0: np.ndarray,
    h_grad: float | None = None,
    **kwargs,
) -> tuple[float, np.ndarray]:
    """Magnetic energy AND its full 5D gradient from a SINGLE solve.

    Returns ``(U, g)`` with ``g = (dU/dx, dU/dy, dU/dz, dU/dtheta, dU/dphi)``
    — magnetic part only (no gravity).

    Physics: U(r0) = -1/2 m . B_ind(r0; r0) depends on r0 both through the
    evaluation point and through the source position in the Neumann boundary
    condition, so the force is NOT -1/2 grad_eval(m . B_ind). But the induced
    field derives from a *symmetric* Green's function (reciprocity), so the
    eval-point and source-point derivatives are equal and the factor 2 from
    the product rule cancels the 1/2 exactly:

        F      = + grad_r [ m . B_ind(r; r0) ] |_{r=r0}   (source FROZEN)
        dU/dq  = - (dm/dq) . B_ind(r0)   for q in {theta, phi}

    i.e. the classical "force/torque against the frozen image" result. Both
    follow from the one solve already needed for U:

    - the angular derivatives reuse B_ind(r0) directly;
    - the spatial force needs grad_r(m . B_ind) = m_j dB_j/dx_i: the
      analytic image part (when ``subtract_singularity=True``) is
      differentiated in closed form, and the smooth FEM *residual* part is
      differentiated by a short central difference over the exact
      element-local gradients (`PhiSolution.grad`) with step ``h_grad`` —
      this costs 6 extra gradient *interpolations*, not extra solves.
      Because grad(P2) is piecewise linear, its FD spans a few elements and
      acts as a local average of the piecewise-constant FEM Hessian.

    Parameters
    ----------
    m_mag        : |m| [A m^2]; orientation given by (theta, phi)
    theta, phi   : moment orientation [rad]
    r0           : (3,) dipole position [m]
    h_grad       : step for the residual-Hessian central difference [m];
                   default = 0.125 x mean edge length of the tet containing
                   r0 (capped at 0.7 x the distance to the nearest SC facet).
                   Empirically the best match to the discrete energy's own
                   FD gradient (0.4 % on the 0.2 mm Fuchs mesh, 0.05 % on
                   the 0.3 mm Vinante mesh); larger steps average across
                   elements and can bias the force by a few % on rough
                   landscapes, smaller ones pick up piecewise-Hessian noise.
    kwargs       : forwarded to `solve_phi` (element=, subtract_singularity=,
                   ...). `B_source` is not supported here (point dipole only).
    """
    from .dipole import dipole_moment, dipole_moment_derivatives
    from .half_space import grad_B_halfspace_image

    if kwargs.get("B_source") is not None:
        raise ValueError("U_mag_and_grad supports point dipoles only "
                         "(B_source= is not allowed).")

    r0 = np.asarray(r0, dtype=float).reshape(3)
    m = dipole_moment(m_mag, theta, phi)

    phi_sol = solve_phi(sctmesh, m, r0, **kwargs)
    B = B_induced_at(phi_sol, r0)                  # total induced field
    U = -0.5 * float(m @ B)

    dm_dth, dm_dph = dipole_moment_derivatives(m_mag, theta, phi)
    dU_dth = -float(dm_dth @ B)
    dU_dph = -float(dm_dph @ B)

    mesh = phi_sol.basis.mesh
    if h_grad is None:
        from .solver import _cached_element_finder
        cell = _cached_element_finder(phi_sol.basis)(*r0.reshape(3, 1))[0]
        verts = mesh.p[:, mesh.t[:, cell]]         # (3, 4)
        edges = [np.linalg.norm(verts[:, a] - verts[:, b])
                 for a in range(4) for b in range(a + 1, 4)]
        h_grad = 0.125 * float(np.mean(edges))
        # Keep the stencil inside the air domain: near the SC surface, cap
        # the step at a fraction of the distance to the nearest SC facet.
        sc_f = sctmesh.sc_facets
        if sc_f is not None and len(sc_f) > 0:
            cents = mesh.p[:, mesh.facets[:, sc_f]].mean(axis=1)  # (3, M)
            d_sc = float(np.sqrt(((cents - r0[:, None]) ** 2).sum(axis=0).min()))
            h_grad = min(h_grad, 0.7 * d_sc) if d_sc > 0 else h_grad

    # D[i, j] = d B_res,j / d x_i by central differences of the exact
    # element-local gradient (6 interpolations, no solves).
    e = np.eye(3) * h_grad
    pts = np.column_stack([r0 + e[0], r0 - e[0],
                           r0 + e[1], r0 - e[1],
                           r0 + e[2], r0 - e[2]])  # (3, 6)
    Bres = phi_sol.grad(pts)                       # (3, 6)
    D = np.empty((3, 3))
    for i in range(3):
        D[i, :] = (Bres[:, 2 * i] - Bres[:, 2 * i + 1]) / (2.0 * h_grad)

    if phi_sol.image_source is not None:
        m_src, r0_src, plane_pt, n_pl = phi_sol.image_source
        D += grad_B_halfspace_image(
            r0[None, :], m_src, r0_src, plane_pt, n_pl)[0]

    # grad B is symmetric for a curl-free field; symmetrise away FD noise.
    D = 0.5 * (D + D.T)
    F = D @ m                                      # F_i = m_j dB_j/dx_i
    g = np.empty(5)
    g[:3] = -F
    g[3] = dU_dth
    g[4] = dU_dph
    return U, g


DEFAULT_GRAVITY = np.array([0.0, 0.0, -G_GRAV])


def gravity_potential(r0: np.ndarray, mass: float,
                      g_vec: np.ndarray = DEFAULT_GRAVITY) -> float:
    """Gravitational PE = -m * g_vec . r0.

    With the default `g_vec = (0, 0, -g)` this is `+m g z` (PE rises with
    height). For a trap tilted by angle theta about the y-axis use
    `g_vec = g * (sin theta, 0, -cos theta)`; the +x component of gravity
    pushes the equilibrium downhill.
    """
    g = np.asarray(g_vec, dtype=float).reshape(3)
    return -float(mass) * float(np.dot(g, np.asarray(r0, dtype=float).reshape(3)))


def U_total(sctmesh: SCTrapMesh,
            m: np.ndarray,
            r0: np.ndarray,
            mass: float = MAGNET_MASS,
            g_vec: np.ndarray = DEFAULT_GRAVITY,
            **kwargs) -> float:
    """U_mag + gravitational PE in `g_vec`."""
    return U_mag(sctmesh, m, r0, **kwargs) + gravity_potential(r0, mass, g_vec)


# ---------------------------------------------------------------------------
# Volumetric-particle energy path
# ---------------------------------------------------------------------------
#
# These functions are additive: they leave U_mag / U_total untouched.  They
# route the inducing source field through `particle.B_field(...)` (analytic
# closed form for the volumetric magnet, via magpylib) instead of a point
# dipole, then evaluate the interaction energy via `particle.interaction_energy`.

def U_mag_particle(sctmesh: SCTrapMesh,
                   particle,
                   r0: np.ndarray,
                   theta: float,
                   phi: float,
                   **kwargs) -> float:
    """Magnetic interaction energy for a finite-size magnet.

    `particle` must expose:
      - `B_field(points, r0, theta, phi) -> (N, 3) Tesla` (used to build the
        Neumann RHS on the SC facets), and
      - `interaction_energy(phi_sol, r0, theta, phi) -> float` (used to turn
        the solved potential into the energy).

    Extra kwargs go to `solve_phi`. `subtract_singularity=True` is rejected
    (the half-space image is only valid for a point dipole).
    """
    if kwargs.get("subtract_singularity", False):
        raise ValueError(
            "U_mag_particle does not support subtract_singularity=True "
            "(image-method subtraction is point-dipole only)."
        )
    kwargs.pop("subtract_singularity", None)

    r0 = np.asarray(r0, dtype=float).reshape(3)

    def B_source(points: np.ndarray) -> np.ndarray:
        return particle.B_field(points, r0, theta, phi)

    # `m` and `r0` are still required positional args of solve_phi for
    # bookkeeping/debugging; the actual source field comes from B_source.
    phi_sol = solve_phi(sctmesh, np.zeros(3), r0,
                        subtract_singularity=False,
                        B_source=B_source, **kwargs)
    return particle.interaction_energy(phi_sol, r0, theta, phi)


def U_total_particle(sctmesh: SCTrapMesh,
                     particle,
                     r0: np.ndarray,
                     theta: float,
                     phi: float,
                     g_vec: np.ndarray = DEFAULT_GRAVITY,
                     **kwargs) -> float:
    """U_mag_particle + gravitational PE for a finite-size magnet."""
    return (U_mag_particle(sctmesh, particle, r0, theta, phi, **kwargs)
            + gravity_potential(r0, particle.mass, g_vec))


def gravity_from_tilt(angle: float, axis: str = "y",
                      g: float = G_GRAV) -> np.ndarray:
    """Build a gravity vector for a trap tilted by `angle` rad about `axis`.

    `axis="y"` rotates the trap so that +x becomes downhill (default),
    `axis="x"` rotates so +y becomes downhill. `angle=0` gives -z gravity.
    """
    s, c = np.sin(angle), np.cos(angle)
    if axis == "y":
        return np.array([g * s, 0.0, -g * c])
    if axis == "x":
        return np.array([0.0, g * s, -g * c])
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


@dataclass
class Sweep1D:
    axis: str               # one of "x","y","z"
    coords: np.ndarray      # (N,)
    U_mag: np.ndarray       # (N,)
    U_grav: np.ndarray      # (N,)

    @property
    def U_total(self) -> np.ndarray:
        return self.U_mag + self.U_grav


def sweep_axis(
    sctmesh: SCTrapMesh,
    axis: str,
    coords: np.ndarray,
    m: np.ndarray,
    fixed: Optional[np.ndarray] = None,
    mass: float = MAGNET_MASS,
    verbose: bool = False,
    **kwargs,
) -> Sweep1D:
    """Compute U_mag along one Cartesian axis with the other two coordinates fixed.

    Parameters
    ----------
    axis   : "x", "y", or "z"
    coords : (N,) values of the swept coordinate [m]
    m      : (3,) dipole moment [A m^2]
    fixed  : (3,) baseline position; the swept axis component is overwritten.
             Defaults to the origin.
    """
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    base = np.asarray(fixed, dtype=float).reshape(3) if fixed is not None else np.zeros(3)
    coords = np.asarray(coords, dtype=float).ravel()

    Um = np.empty(coords.size, dtype=float)
    Ug = np.empty(coords.size, dtype=float)

    for i, c in enumerate(coords):
        r0 = base.copy()
        r0[idx] = c
        Um[i] = U_mag(sctmesh, m, r0, **kwargs)
        Ug[i] = mass * G_GRAV * float(r0[2])
        if verbose:
            print(f"  [{i+1}/{coords.size}] {axis}={c:.4e}  "
                  f"U_mag={Um[i]:.4e}  U_tot={Um[i]+Ug[i]:.4e}", flush=True)

    return Sweep1D(axis=axis, coords=coords, U_mag=Um, U_grav=Ug)
