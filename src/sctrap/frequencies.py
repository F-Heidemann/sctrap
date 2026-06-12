"""Equilibrium location and trap frequencies via parabolic curvature fits."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .config import G_GRAV, MAGNET_MASS, MAGNET_INERTIA, MAGNET_MOMENT
from .dipole import dipole_moment
from .mesh import SCTrapMesh
from .potential import U_mag


@dataclass
class TrapFrequencies:
    """Results of a trap-frequency calculation around equilibrium."""

    eq_r: np.ndarray                  # (3,) equilibrium position [m]
    eq_theta: float = 0.0
    eq_phi:   float = 0.0
    f_x:     float = float("nan")
    f_y:     float = float("nan")
    f_z:     float = float("nan")
    f_theta: float = float("nan")
    f_phi:   float = float("nan")

    def __str__(self) -> str:
        ex, ey, ez = self.eq_r * 1e3
        def fmt(name, val):
            return (f"  {name:7s} = {val:10.4f} Hz" if np.isfinite(val)
                    else f"  {name:7s} = (skipped)")
        return "\n".join([
            "--- Trap frequencies ---",
            f"  equilibrium r = ({ex:.4f}, {ey:.4f}, {ez:.4f}) mm",
            f"  equilibrium orientation: theta={self.eq_theta:.4f}  phi={self.eq_phi:.4f}",
            fmt("f_x", self.f_x),
            fmt("f_y", self.f_y),
            fmt("f_z", self.f_z),
            fmt("f_theta", self.f_theta),
            fmt("f_phi",   self.f_phi),
        ])

    def as_dict(self) -> dict:
        return {
            "eq_r_m":      self.eq_r.tolist(),
            "eq_theta_rad": float(self.eq_theta),
            "eq_phi_rad":   float(self.eq_phi),
            "f_x_Hz":      float(self.f_x),
            "f_y_Hz":      float(self.f_y),
            "f_z_Hz":      float(self.f_z),
            "f_theta_Hz":  float(self.f_theta),
            "f_phi_Hz":    float(self.f_phi),
        }


# ---------------------------------------------------------------------------
# Parabolic frequency fit
# ---------------------------------------------------------------------------

def _parabola_freq(coords: np.ndarray, U: np.ndarray, M_eff: float) -> float:
    if len(coords) < 3:
        return float("nan")
    c2, c1, c0 = np.polyfit(coords, U, 2)
    k = 2.0 * c2
    if k <= 0:
        return float("nan")
    return float(np.sqrt(k / M_eff) / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Equilibrium finder
# ---------------------------------------------------------------------------

def find_equilibrium(
    sctmesh: SCTrapMesh,
    m: np.ndarray,
    r0_guess: np.ndarray,
    mass: float = MAGNET_MASS,
    g_vec: np.ndarray | None = None,
    method: str = "Nelder-Mead",
    xatol: float = 1e-7,
    maxiter: int = 200,
    verbose: bool = False,
    progress: bool = False,
    particle=None,
    **solver_kwargs,
) -> np.ndarray:
    """Refine the equilibrium position by minimising U_mag + gravity PE.

    `g_vec` defaults to (0,0,-g); pass a tilted vector to study tilt effects.

    When `particle=` is provided, the objective routes through
    `U_total_particle` (uses the particle's analytic source field on the SC
    facets instead of a point dipole) and the gravitational term uses
    `particle.mass`.  The orientation (theta, phi) is recovered from `m`.
    """
    from .potential import (DEFAULT_GRAVITY, U_total_particle,
                            gravity_potential)
    m_arr = np.asarray(m, dtype=float).reshape(3)
    r0_guess = np.asarray(r0_guess, dtype=float).reshape(3)
    g = DEFAULT_GRAVITY if g_vec is None else np.asarray(g_vec, dtype=float).reshape(3)

    n_calls = {"n": 0, "t0": time.time()}

    def _tick(r, u):
        n_calls["n"] += 1
        if verbose:
            print(f"  [{n_calls['n']:3d}] r=({r[0]:.4e},{r[1]:.4e},{r[2]:.4e})  U={u:.6e}",
                  flush=True)
        elif progress:
            sys.stdout.write(
                f"\r  Equilibrium: eval {n_calls['n']:3d}  "
                f"z={r[2]*1e3:+.4f} mm  [{time.time()-n_calls['t0']:5.0f}s]   ")
            sys.stdout.flush()

    if particle is None:
        def objective(r):
            u = (U_mag(sctmesh, m_arr, r, **solver_kwargs)
                 + gravity_potential(r, mass, g))
            _tick(r, u)
            return u
    else:
        m_mag = float(np.linalg.norm(m_arr))
        if m_mag <= 0:
            raise ValueError("m must have non-zero magnitude when particle= is set")
        mz = m_arr[2] / m_mag
        theta_eq = float(np.arccos(np.clip(mz, -1.0, 1.0)))
        phi_eq   = float(np.arctan2(m_arr[1], m_arr[0]))
        # Drop subtract_singularity from kwargs forwarded into the volumetric
        # path; it is not supported there (and image-method subtraction is
        # point-dipole only).
        vol_kwargs = {k: v for k, v in solver_kwargs.items()
                      if k != "subtract_singularity"}

        def objective(r):
            u = U_total_particle(sctmesh, particle, r,
                                 theta_eq, phi_eq, g_vec=g, **vol_kwargs)
            _tick(r, u)
            return u

    res = minimize(objective, r0_guess, method=method,
                   options={"xatol": xatol, "fatol": 1e-22, "maxiter": maxiter})
    if progress and not verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return res.x


# ---------------------------------------------------------------------------
# Joint (5D) position + orientation equilibrium
# ---------------------------------------------------------------------------
#
# `find_equilibrium` above only minimises over the three position DOFs and
# holds (theta, phi) fixed.  In an asymmetric cavity that can leave the
# magnet sitting at a *saddle* of U_total — the Hessian taken there has
# negative eigenvalues along the un-optimised orientation DOFs, and the
# resulting mode frequencies are not physical.
#
# `find_equilibrium_5d` jointly optimises (x, y, z, theta, phi).  Returns
# (r_eq, theta_eq, phi_eq).  The optimiser is Nelder-Mead with a hand-tuned
# initial simplex that gives lengths and angles comparable step sizes; this
# matters because raw 1 m vs 1 rad is a 1000x scale mismatch.

def find_equilibrium_5d(
    sctmesh: SCTrapMesh,
    m_mag: float,
    r0_guess: np.ndarray,
    theta_guess: float = np.pi / 2.0,
    phi_guess:   float = 0.0,
    *,
    mass: float = MAGNET_MASS,
    g_vec: np.ndarray | None = None,
    method: str = "auto",           # "auto" | "gradient" | "nelder-mead"
    xatol: float = 1e9,             # simplex tol in SCALED coords; large =>
                                    # disabled, so fatol alone governs (below)
    fatol: float | None = None,     # energy tol [J]; None -> ftol_rel * |U0|
    ftol_rel: float = 1e-4,         # relative energy tol when fatol is None
    maxiter: int = 300,
    step_pos: float = 50e-6,        # initial simplex step in x, y, z [m]
    step_ang: float = 5e-2,         # initial simplex step in theta, phi [rad]
    verbose: bool = False,
    progress: bool = False,
    particle=None,
    **solver_kwargs,
) -> tuple[np.ndarray, float, float]:
    """Joint 5D minimisation of U_total over (x, y, z, theta, phi).

    Parameters
    ----------
    sctmesh    : SCTrapMesh
    m_mag      : |m| [A m^2]; orientation is read from (theta, phi)
    r0_guess   : (3,) initial position [m]
    theta_guess, phi_guess : initial orientation [rad]
    method     : "gradient" uses L-BFGS-B driven by the analytic force/torque
                 from a single solve per iterate (`U_mag_and_grad`;
                 reciprocity / frozen-image form — validated against the FD
                 energy gradient to <0.1 % where the FD itself converges).
                 "nelder-mead" is the legacy derivative-free search. "auto"
                 (default) picks "gradient" for the point-dipole path and
                 falls back to "nelder-mead" when `particle=` is given (no
                 analytic force is available for the volumetric source).
    particle   : optional volumetric particle; routes through U_total_particle

    Returns
    -------
    (r_eq, theta_eq, phi_eq)
    """
    from .potential import (DEFAULT_GRAVITY, U_total_particle,
                            gravity_potential)
    r0 = np.asarray(r0_guess, dtype=float).reshape(3)
    g = DEFAULT_GRAVITY if g_vec is None else np.asarray(g_vec, dtype=float).reshape(3)

    if method not in ("auto", "gradient", "nelder-mead"):
        raise ValueError(f"method must be 'auto', 'gradient' or "
                         f"'nelder-mead'; got {method!r}")
    if method == "gradient" and particle is not None:
        raise ValueError("method='gradient' requires the point-dipole path "
                         "(no analytic force for volumetric particles); "
                         "use method='nelder-mead' or 'auto'.")
    use_gradient = (method != "nelder-mead") and particle is None

    if use_gradient:
        return _find_equilibrium_5d_gradient(
            sctmesh, m_mag, r0, float(theta_guess), float(phi_guess),
            mass=mass, g=g, ftol_rel=ftol_rel, fatol=fatol,
            maxiter=maxiter, step_pos=step_pos, step_ang=step_ang,
            verbose=verbose, progress=progress, **solver_kwargs)

    n_calls = {"n": 0, "t0": time.time()}

    def _tick(r, theta, phi, u):
        n_calls["n"] += 1
        if verbose:
            print(f"  [{n_calls['n']:4d}] r=({r[0]:+.3e},{r[1]:+.3e},{r[2]:+.3e}) "
                  f"theta={theta:+.4f} phi={phi:+.4f}  U={u:.6e}", flush=True)
        elif progress:
            sys.stdout.write(
                f"\r  Equilibrium (5D): eval {n_calls['n']:4d}  "
                f"z={r[2]*1e3:+.3f} mm  theta={theta:+.3f} phi={phi:+.3f} rad  "
                f"[{time.time()-n_calls['t0']:5.0f}s]   ")
            sys.stdout.flush()

    if particle is None:
        def objective(q):
            r = q[:3]
            theta, phi = float(q[3]), float(q[4])
            m_vec = dipole_moment(m_mag, theta, phi)
            u = (U_mag(sctmesh, m_vec, r, **solver_kwargs)
                 + gravity_potential(r, mass, g))
            _tick(r, theta, phi, u)
            return u
    else:
        vol_kwargs = {k: v for k, v in solver_kwargs.items()
                      if k != "subtract_singularity"}
        def objective(q):
            r = q[:3]
            theta, phi = float(q[3]), float(q[4])
            u = U_total_particle(sctmesh, particle, r,
                                 theta, phi, g_vec=g, **vol_kwargs)
            _tick(r, theta, phi, u)
            return u

    q0 = np.array([r0[0], r0[1], r0[2],
                   float(theta_guess), float(phi_guess)])

    # Non-dimensionalise: position (~1e-3 m) and orientation (~1 rad) live in
    # one vector, so they need comparable simplex steps. Rescale each DOF by its
    # natural step so the optimiser sees an isotropic O(1) problem (this only
    # shapes the initial simplex; `xatol` is disabled below).
    #
    # Convergence is governed by `fatol` ALONE (xatol defaulted huge). Reason:
    # the objective is noise-limited (floor ~1e-12 J here). A simplex-size
    # (`xatol`) criterion would demand the simplex shrink far below the size at
    # which the energy spread already hit the noise floor -- i.e. shrink into
    # the noise -- so it can never be met and the search grinds to `maxiter`.
    # Stopping on `fatol` (a few x the floor) halts exactly when the simplex can
    # no longer distinguish energies, which is the best achievable resolution
    # and auto-tightens on finer meshes (lower floor -> lower |U0|-relative
    # fatol -> smaller terminal simplex). The off-centre equilibrium error this
    # leaves (~10 um / ~1 deg) does NOT bias the trap frequencies: those come
    # from a symmetric 2nd-difference Hessian, which cancels the linear term.
    scale = np.array([step_pos, step_pos, step_pos, step_ang, step_ang])

    def objective_scaled(y):
        return objective(y * scale)

    y0 = q0 / scale

    # Energy tolerance. The historical default `fatol = 1e-22` is ~1e3x below
    # the FEM objective's own numerical floor (measured relative floor ~4e-5 for
    # the singularity-subtracted point-dipole solve: the nearest-facet image
    # reflection jumps discretely as the dipole moves), so the f-test never
    # passes and the search runs to `maxiter` long after physical convergence.
    # Tie it to the objective magnitude with `ftol_rel` safely *above* that
    # relative floor. This is mesh-robust because the floor is ~constant in
    # relative terms (a finer mesh lowers the absolute floor but not the ratio),
    # so no per-mesh noise characterisation is needed. Pinning the equilibrium
    # tighter than the floor buys nothing: the trap frequencies come from the
    # Hessian over a ~tens-of-microns window, not from the equilibrium position.
    if fatol is not None:
        fatol_eff = fatol
    else:
        u0 = abs(float(objective_scaled(y0)))   # one extra eval (fallback only)
        fatol_eff = max(ftol_rel * u0, 1e-30)
    if progress or verbose:
        print(f"  Equilibrium (5D): fatol = {fatol_eff:.2e} J"
              + (f" (ftol_rel={ftol_rel:.0e} x |U0|)" if fatol is None else ""),
              flush=True)

    # Initial simplex (n+1 = 6 vertices in 5D): unit step along each scaled DOF.
    simplex = np.empty((6, 5))
    simplex[0] = y0
    for i in range(5):
        v = y0.copy()
        v[i] += 1.0
        simplex[i + 1] = v

    res = minimize(objective_scaled, y0, method="Nelder-Mead",
                   options={"xatol": xatol, "fatol": fatol_eff,
                            "maxiter": maxiter,
                            "initial_simplex": simplex})
    if progress and not verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()
    q_eq = res.x * scale
    # Wrap phi into (-pi, pi] for tidy reporting (phi is exactly 2pi-periodic).
    # Leave theta as the optimiser returned it (folding it would change the
    # actual moment direction).
    phi_eq = (float(q_eq[4]) + np.pi) % (2.0 * np.pi) - np.pi
    return q_eq[:3].copy(), float(q_eq[3]), phi_eq


def _find_equilibrium_5d_gradient(
    sctmesh: SCTrapMesh,
    m_mag: float,
    r0: np.ndarray,
    theta0: float,
    phi0: float,
    *,
    mass: float,
    g: np.ndarray,
    ftol_rel: float = 1e-4,
    fatol: float | None = None,
    maxiter: int = 300,
    step_pos: float = 50e-6,
    step_ang: float = 5e-2,
    verbose: bool = False,
    progress: bool = False,
    **solver_kwargs,
) -> tuple[np.ndarray, float, float]:
    """Gradient-based 5D equilibrium search (L-BFGS-B, one solve per iterate).

    Each objective evaluation returns U_total AND its analytic 5D gradient
    from a SINGLE Laplace solve (`U_mag_and_grad`): force from the
    frozen-image / reciprocity form, torque from -dm/dq . B_ind. Compared to
    Nelder-Mead this both cuts the number of solves ~an order of magnitude
    and removes the stall-at-noise-floor pathology (the analytic gradient is
    smooth even where the *energy* sits at its discretisation floor).

    Non-dimensionalisation mirrors the Nelder-Mead path: DOFs scaled by
    (step_pos, step_ang), energy scaled by |U(q0)| so the optimiser sees an
    O(1) problem. Convergence is governed by FORCE BALANCE (projected
    gradient <= 1e-3 x the particle's weight, expressed through scipy's
    `gtol`), not by energy decrease: near equilibrium the landscape is so
    flat that an energy criterion stops with several % of the weight still
    unbalanced (~15 um early on the Fuchs benchmark).

    Position bounds: the mesh bounding box shrunk by 5 % of each extent.
    Trial points that still land outside the (non-convex) air domain are
    answered with a quadratic penalty bowl rather than a crash, and the
    residual-gradient stencil in `U_mag_and_grad` auto-shrinks near the SC
    surface, so every evaluation stays well-defined.
    """
    from scipy.optimize import minimize as _minimize
    from .half_space import find_nearest_sc_facet
    from .potential import U_mag_and_grad, gravity_potential

    solver_kwargs = dict(solver_kwargs)
    # Freeze the singularity-subtraction image plane for the duration of the
    # search (re-frozen below if the dipole migrates to another facet). The
    # per-solve nearest-facet selection makes U(r0) discontinuous across
    # facet-Voronoi boundaries — degenerate ON a symmetry axis, where it
    # carves an artificial groove that traps any faithful minimiser (the
    # groove bottom sits BELOW the smooth-branch minimum). With a frozen
    # plane the split Phi = Phi_image + Phi_residual is still exact and the
    # landscape is smooth.
    freeze_plane = (bool(solver_kwargs.get("subtract_singularity", False))
                    and solver_kwargs.get("image_plane") is None)

    scale = np.array([step_pos, step_pos, step_pos, step_ang, step_ang])
    q0 = np.array([r0[0], r0[1], r0[2], theta0, phi0])
    y0 = q0 / scale

    n_calls = {"n": 0, "t0": time.time()}

    def U_and_grad(q):
        U_m, g_m = U_mag_and_grad(sctmesh, m_mag, float(q[3]), float(q[4]),
                                  q[:3], **solver_kwargs)
        U = U_m + gravity_potential(q[:3], mass, g)
        g_tot = g_m.copy()
        g_tot[:3] += -mass * g          # d/dr [-mass g.r] = -mass g
        n_calls["n"] += 1
        if verbose:
            print(f"  [{n_calls['n']:4d}] r=({q[0]:+.3e},{q[1]:+.3e},"
                  f"{q[2]:+.3e}) theta={q[3]:+.4f} phi={q[4]:+.4f}  "
                  f"U={U:.6e}  |gr|={np.linalg.norm(g_tot[:3]):.3e}",
                  flush=True)
        elif progress:
            sys.stdout.write(
                f"\r  Equilibrium (5D, gradient): solve {n_calls['n']:4d}  "
                f"z={q[2]*1e3:+.4f} mm  theta={q[3]:+.3f} phi={q[4]:+.3f} rad"
                f"  [{time.time()-n_calls['t0']:5.0f}s]   ")
            sys.stdout.flush()
        return U, g_tot

    # Reference energy for non-dimensionalisation (first solve is reused by
    # scipy as the initial f/g evaluation via a tiny one-entry cache). If a
    # line-search trial point leaves the air domain (possible inside the
    # bounding box of a non-convex cavity), `element_finder` raises — answer
    # with a quadratic penalty bowl steering back to the last good point
    # instead of crashing.
    cache = {}
    last_good = {"y": y0.copy(), "U": None}

    def fun(y):
        key = y.tobytes()
        if key not in cache:
            cache.clear()
            try:
                U, g_tot = U_and_grad(y * scale)
            except ValueError:
                d = y - last_good["y"]
                U_base = last_good["U"] if last_good["U"] is not None else Uref
                return (abs(U_base) / Uref + float(d @ d),
                        2.0 * d)
            cache[key] = (U, g_tot)
            last_good["y"] = y.copy()
            last_good["U"] = U
        U, g_tot = cache[key]
        return U / Uref, g_tot * scale / Uref

    U0, _ = U_and_grad(q0)
    Uref = max(abs(U0), 1e-30)

    # Stop on FORCE BALANCE, not on energy decrease. Near equilibrium the
    # landscape is so flat that a residual force of several % of the weight
    # changes U by only ~1e-4 relative per step — an energy criterion quits
    # there (observed on Fuchs: stopped ~15 um early with 0.07 x weight of
    # unbalanced force). The analytic gradient is smooth and cheap, so the
    # natural criterion is |grad U| <= gtol_rel x (mass g): pgtol in scaled
    # units = gtol_rel * mass*|g| * step_pos / Uref. `ftol` is set tiny so
    # the gradient test governs; `fatol`/`ftol_rel` are accepted for
    # API compatibility and fold into a (rarely binding) energy backstop.
    gtol_rel = 1e-3
    g_mag = float(np.linalg.norm(g))
    pgtol = gtol_rel * mass * g_mag * step_pos / Uref
    ftol_eff = (fatol / Uref) if fatol is not None else 1e-12
    if progress or verbose:
        print(f"  Equilibrium (5D, gradient): pgtol = {gtol_rel:.0e} x weight"
              f"  (|U0| = {Uref:.3e} J)", flush=True)

    # Position bounds: bbox shrunk by max(5 % of extent, h_grad) per side so
    # the residual-gradient stencil in U_mag_and_grad stays inside the mesh.
    lo, hi = sctmesh.bounding_box()
    ext = hi - lo
    margin = 0.05 * ext
    bounds = [((lo[i] + margin[i]) / scale[i], (hi[i] - margin[i]) / scale[i])
              for i in range(3)] + [(None, None), (None, None)]

    # Outer loop: minimise with a frozen image plane; if the converged point
    # is nearest a DIFFERENT SC facet than the frozen one, re-freeze there
    # and continue (rarely more than one extra round — the seed is good).
    y_start = y0
    frozen_idx = None
    for outer in range(3):
        if freeze_plane:
            pt, n_pl, frozen_idx = find_nearest_sc_facet(
                sctmesh, y_start[:3] * scale[:3])
            solver_kwargs["image_plane"] = (pt, n_pl)
            cache.clear()
        res = _minimize(fun, y_start, jac=True, method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": maxiter, "ftol": ftol_eff,
                                 "gtol": pgtol, "maxls": 40})
        if verbose:
            print(f"  L-BFGS-B[{outer}]: {res.message}  nfev={res.nfev}  "
                  f"total solves={n_calls['n']}", flush=True)
        if not freeze_plane:
            break
        _, _, idx_here = find_nearest_sc_facet(sctmesh, res.x[:3] * scale[:3])
        if idx_here == frozen_idx:
            break
        y_start = res.x
    if progress and not verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()

    q_eq = res.x * scale
    phi_eq = (float(q_eq[4]) + np.pi) % (2.0 * np.pi) - np.pi
    return q_eq[:3].copy(), float(q_eq[3]), phi_eq


# ---------------------------------------------------------------------------
# 5-point parabolic stencil per DOF
# ---------------------------------------------------------------------------

def frequencies_at(
    sctmesh: SCTrapMesh,
    eq_r: np.ndarray,
    m: np.ndarray,
    *,
    mass: float    = MAGNET_MASS,
    inertia: float = MAGNET_INERTIA,
    h_trans: float = 50e-6,        # 50 microns
    h_ang:   float = 1e-3,         # 1 mrad
    n_pts:   int   = 5,
    dofs: Sequence[str] = ("x", "y", "z", "theta", "phi"),
    verbose: bool = False,
    **solver_kwargs,
) -> TrapFrequencies:
    """Sample U around equilibrium and parabola-fit each DOF separately.

    Parameters
    ----------
    eq_r       : (3,) equilibrium position [m]
    m          : (3,) dipole moment vector at equilibrium orientation
    h_trans    : translational stencil half-width [m]
    h_ang      : angular stencil half-width [rad]
    n_pts      : number of stencil points per DOF (odd; usually 5)
    dofs       : which DOFs to fit (default: all five)
    """
    if n_pts % 2 == 0:
        raise ValueError("n_pts must be odd")

    eq_r = np.asarray(eq_r, dtype=float).reshape(3)
    m_arr = np.asarray(m, dtype=float).reshape(3)
    m_mag = float(np.linalg.norm(m_arr))
    # Recover orientation (theta, phi) from m
    if m_mag > 0:
        mz = m_arr[2] / m_mag
        theta_eq = float(np.arccos(np.clip(mz, -1.0, 1.0)))
        phi_eq   = float(np.arctan2(m_arr[1], m_arr[0]))
    else:
        theta_eq = phi_eq = 0.0

    tf = TrapFrequencies(eq_r=eq_r.copy(), eq_theta=theta_eq, eq_phi=phi_eq)

    half = (n_pts - 1) // 2

    def U_at(r_vec, theta, phi):
        m_loc = dipole_moment(m_mag, theta, phi) if m_mag > 0 else m_arr
        return (U_mag(sctmesh, m_loc, r_vec, **solver_kwargs)
                + mass * G_GRAV * float(r_vec[2]))

    # --- Translational DOFs (gravity matters only along z) ---------------
    for label, idx, M_eff in (("x", 0, mass), ("y", 1, mass), ("z", 2, mass)):
        if label not in dofs:
            continue
        offsets = np.linspace(-half, half, n_pts) * h_trans
        coords  = eq_r[idx] + offsets
        U_vals  = np.empty(n_pts)
        for i, c in enumerate(coords):
            r = eq_r.copy()
            r[idx] = c
            U_vals[i] = U_at(r, theta_eq, phi_eq)
            if verbose:
                print(f"    {label}[{i+1}/{n_pts}] = {c:.4e}  U={U_vals[i]:.6e}", flush=True)
        f = _parabola_freq(coords, U_vals, M_eff)
        setattr(tf, f"f_{label}", f)

    # --- Librational DOFs (theta, phi); position fixed at equilibrium ----
    if m_mag > 0:
        if "theta" in dofs:
            offsets = np.linspace(-half, half, n_pts) * h_ang
            thetas  = theta_eq + offsets
            U_vals  = np.array([U_at(eq_r, t, phi_eq) for t in thetas])
            tf.f_theta = _parabola_freq(thetas, U_vals, inertia)

        if "phi" in dofs:
            offsets = np.linspace(-half, half, n_pts) * h_ang
            phis    = phi_eq + offsets
            U_vals  = np.array([U_at(eq_r, theta_eq, p) for p in phis])
            tf.f_phi = _parabola_freq(phis, U_vals, inertia)

    return tf
