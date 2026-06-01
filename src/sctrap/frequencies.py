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
    xatol: float = 1e-7,
    fatol: float = 1e-22,
    maxiter: int = 500,
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
    particle   : optional volumetric particle; routes through U_total_particle

    Returns
    -------
    (r_eq, theta_eq, phi_eq)
    """
    from .potential import (DEFAULT_GRAVITY, U_total_particle,
                            gravity_potential)
    r0 = np.asarray(r0_guess, dtype=float).reshape(3)
    g = DEFAULT_GRAVITY if g_vec is None else np.asarray(g_vec, dtype=float).reshape(3)

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

    # Hand-build an initial simplex (n+1 = 6 vertices in 5D) so position and
    # orientation steps are comparable in objective-value sensitivity.
    dq = np.array([step_pos, step_pos, step_pos, step_ang, step_ang])
    simplex = np.empty((6, 5))
    simplex[0] = q0
    for i in range(5):
        v = q0.copy()
        v[i] += dq[i]
        simplex[i + 1] = v

    res = minimize(objective, q0, method="Nelder-Mead",
                   options={"xatol": xatol, "fatol": fatol,
                            "maxiter": maxiter,
                            "initial_simplex": simplex})
    if progress and not verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()
    q_eq = res.x
    # Wrap phi into (-pi, pi] for tidy reporting (phi is exactly 2pi-periodic).
    # Leave theta as the optimiser returned it (folding it would change the
    # actual moment direction).
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
