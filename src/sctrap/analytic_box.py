"""Analytic cuboidal-box image-lattice trap estimate.

A magnetic point dipole ``m`` inside a closed rectangular superconducting
cavity ``[-a, a] x [-b, b] x [0, H]`` (``B.n = 0`` on all six walls) is
trapped by an infinite **image lattice** — a tensor product of three 1D
parity chains. The induced field gives the trap potential
``U = -1/2 m . B_ind(r0)``, and its Hessian yields every translational and
librational stiffness in closed form.

This module is a self-contained port of the cuboidal-trap theory in
``cuboidal_analytic_trap/src`` (``analytic_potential.py`` / ``closed_box_bar.py``),
kept dependency-free so ``sctrap`` can use it as

  * a cheap **seed** for the FEM equilibrium search (z_eq, orientation), and
  * a physically matched **fatol** for the optimiser (from the analytic
    curvature scale, not the irrelevant baseline energy magnitude).

It is also a useful **benchmark**: for the Fuchs geometry it reproduces the
reference prediction f_z ~ 27.8 Hz / k_z ~ 13.1 mN/m to point-dipole accuracy.

Conventions match the appendix and ``sctrap.dipole.dipole_moment``:
  a, b are HALF-widths; H is the full height in z; the dipole orientation is
  ``m = |m| (sin th cos ph, sin th sin ph, cos th)`` (theta from +z, phi from +x).
The energy uses the image self-energy convention with mu0/(4 pi) = 1e-7.
"""

from __future__ import annotations

import numpy as np

MU0_4PI = 1e-7  # T m / A

__all__ = ["box_energy", "box_energy_cesaro", "analytic_box_estimate"]


def _moment_vector(m_mag: float, theta: float, phi: float) -> np.ndarray:
    return m_mag * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def box_energy(a, b, H, x0, y0, z0, m, N=10):
    """Image-lattice trap energy at (x0,y0,z0) for moment vector ``m``.

    Cube truncation |p|,|q|,|r| <= N; the (0,0,0) term (the source) is
    excluded. Image positions / moments follow appendix eq 19:

        r_pqr = (2 p a + (-1)^p x0,  2 q b + (-1)^q y0,
                 r H + (-1)^r (z0 - H/2) + H/2)
        m_pqr = ((-1)^p m_x, (-1)^q m_y, (-1)^r m_z)
    """
    m = np.asarray(m, dtype=float).reshape(3)
    ps = np.arange(-N, N + 1)
    P, Q, R = np.meshgrid(ps, ps, ps, indexing="ij")
    P, Q, R = P.ravel(), Q.ravel(), R.ravel()
    mask = (P != 0) | (Q != 0) | (R != 0)
    P, Q, R = P[mask], Q[mask], R[mask]
    sx, sy, sz = (-1.0) ** P, (-1.0) ** Q, (-1.0) ** R

    Rx = 2 * P * a + sx * x0
    Ry = 2 * Q * b + sy * y0
    Rz = R * H + sz * (z0 - H / 2.0) + H / 2.0

    dx, dy, dz = x0 - Rx, y0 - Ry, z0 - Rz
    R2 = dx * dx + dy * dy + dz * dz
    Rn = np.sqrt(R2)
    R3 = R2 * Rn

    mx, my, mz = m
    mim_x, mim_y, mim_z = sx * mx, sy * my, sz * mz
    mm = mx * mim_x + my * mim_y + mz * mim_z
    mdR = (mx * dx + my * dy + mz * dz) / Rn
    mimdR = (mim_x * dx + mim_y * dy + mim_z * dz) / Rn
    return 0.5 * MU0_4PI * np.sum((mm - 3.0 * mdR * mimdR) / R3)


def box_energy_cesaro(a, b, H, x0, y0, z0, m, N=10):
    """Cesaro mean of truncations N and N+1 (cancels the parity oscillation of
    the conditionally convergent lattice sum)."""
    return 0.5 * (box_energy(a, b, H, x0, y0, z0, m, N)
                  + box_energy(a, b, H, x0, y0, z0, m, N + 1))


def analytic_box_estimate(
    a: float,
    b: float,
    H: float,
    m_mag: float,
    mass: float,
    *,
    theta0: float = np.pi / 2.0,
    phi0: float = np.pi / 2.0,
    g: float = 9.80665,
    N: int = 10,
    eps_pos: float = 2e-7,
    eps_ang: float = 1e-3,
    I_theta: float | None = None,
    I_phi: float | None = None,
) -> dict:
    """Full diagonal-Hessian estimate for a point dipole in a closed SC box.

    Steps: (1) find the gravity equilibrium height z_eq on the box axis for the
    given orientation; (2) take the diagonal position Hessian (k_x, k_y, k_z)
    and the orientation Hessian (k_theta, k_phi) by central differences;
    (3) convert to mode frequencies (librational ones only if the relevant
    moment of inertia is supplied).

    All cross terms vanish on the central axis by parity, so the diagonal
    Hessian is the full Hessian to leading order — which is exactly what the
    trap-frequency / fatol-scale estimate needs.

    Returns a dict with keys: ``z_eq``, ``k_x``, ``k_y``, ``k_z`` [N/m],
    ``k_theta``, ``k_phi`` [J/rad^2], ``f_x``, ``f_y``, ``f_z`` [Hz], and
    ``f_theta``, ``f_phi`` [Hz or None].
    """
    from scipy.optimize import brentq

    m_vec = _moment_vector(m_mag, theta0, phi0)

    # (1) Equilibrium height: dU/dz + M g = 0 on the axis (x=y=0).
    def dUtot_dz(z):
        up = box_energy_cesaro(a, b, H, 0.0, 0.0, z + eps_pos, m_vec, N)
        um = box_energy_cesaro(a, b, H, 0.0, 0.0, z - eps_pos, m_vec, N)
        return (up - um) / (2.0 * eps_pos) + mass * g

    z_eq = float(brentq(dUtot_dz, 0.05 * H, 0.95 * H, xtol=1e-9))

    # (2a) Translational Hessian at (0,0,z_eq). Gravity is linear in z, so it
    # does not contribute to the curvature -> use the magnetic energy alone.
    base = np.array([0.0, 0.0, z_eq])
    U0 = box_energy_cesaro(a, b, H, base[0], base[1], base[2], m_vec, N)
    k_trans = np.zeros(3)
    for i in range(3):
        bp, bm = base.copy(), base.copy()
        bp[i] += eps_pos
        bm[i] -= eps_pos
        up = box_energy_cesaro(a, b, H, bp[0], bp[1], bp[2], m_vec, N)
        um = box_energy_cesaro(a, b, H, bm[0], bm[1], bm[2], m_vec, N)
        k_trans[i] = (up - 2.0 * U0 + um) / (eps_pos * eps_pos)

    # (2b) Orientation Hessian at (theta0, phi0), position fixed at z_eq.
    def U_orient(theta, phi):
        return box_energy_cesaro(a, b, H, base[0], base[1], base[2],
                                 _moment_vector(m_mag, theta, phi), N)

    k_theta = (U_orient(theta0 + eps_ang, phi0)
               - 2.0 * U0 + U_orient(theta0 - eps_ang, phi0)) / (eps_ang ** 2)
    k_phi = (U_orient(theta0, phi0 + eps_ang)
             - 2.0 * U0 + U_orient(theta0, phi0 - eps_ang)) / (eps_ang ** 2)

    def freq(k, inertia):
        if inertia is None or inertia <= 0 or k <= 0:
            return None
        return float(np.sqrt(k / inertia) / (2.0 * np.pi))

    return {
        "z_eq": z_eq,
        "k_x": float(k_trans[0]), "k_y": float(k_trans[1]),
        "k_z": float(k_trans[2]),
        "k_theta": float(k_theta), "k_phi": float(k_phi),
        "f_x": freq(k_trans[0], mass), "f_y": freq(k_trans[1], mass),
        "f_z": freq(k_trans[2], mass),
        "f_theta": freq(k_theta, I_theta), "f_phi": freq(k_phi, I_phi),
    }
