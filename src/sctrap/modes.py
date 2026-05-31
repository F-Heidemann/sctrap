"""Full-Hessian normal-mode extraction.

For a generic SC geometry the principal axes of the trap are not aligned with
the chart coordinates (x, y, z, theta, phi). The correct procedure is to
build the 5x5 Hessian

    H_ij = d^2 U_total / dq_i dq_j   evaluated at equilibrium

via central finite differences, then solve the generalised eigenproblem

    H v_k = omega_k^2 M v_k     with M = diag(m, m, m, I, I)

(equivalently, eigen(M^-1 H) since M is diagonal). The eigenvalues give the
five normal-mode frequencies; the eigenvectors say how much each mode
mixes the chart coordinates.

Stencil cost:
    diagonal       :  5 DOFs * (2*N) U-evals,        N = number of
                      stencil points either side  (default 2 -> 5-point fit).
    cross-terms    :  10 pairs * 4 U-evals          ((+i,+j),(+i,-j),(-i,+j),(-i,-j))
    centre value   :  1
Total at default settings: 5*4 + 10*4 + 1 = 61 FEM solves per Hessian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import G_GRAV, MAGNET_MASS, MAGNET_INERTIA
from .dipole import dipole_moment
from .mesh import SCTrapMesh
from .potential import U_mag


DOF_NAMES = ("x", "y", "z", "theta", "phi")


@dataclass
class NormalModes:
    """Result of a Hessian-based normal-mode analysis around equilibrium."""

    eq_r: np.ndarray                     # (3,)
    eq_theta: float
    eq_phi: float
    H: np.ndarray                        # (5, 5) Hessian [J / (q-unit)^2]
    M: np.ndarray                        # (5, 5) diagonal mass-matrix
    eigvals: np.ndarray                  # (5,)  omega^2 in (rad/s)^2
    eigvecs: np.ndarray                  # (5, 5) columns are mode shapes (physical units)
    f_Hz: np.ndarray                     # (5,)  signed sqrt(omega^2)/(2 pi)
    U0: float                            # U_total at equilibrium [J]
    h_trans: float
    h_ang: float
    # Mass-weighted eigenvectors: orthonormal columns from
    # eigh(M^{-1/2} H M^{-1/2}). |w_i|^2 sums to 1 across DOFs and gives the
    # *physically meaningful* fraction of each DOF in a mode.
    eigvecs_mw: np.ndarray = field(default_factory=lambda: np.zeros((5, 5)))
    diag_anharm: np.ndarray = field(default_factory=lambda: np.array([]))
    # diag_anharm[i] = quartic coefficient for DOF i along its own axis [J/q^4]

    def label(self, k: int) -> str:
        """Short text label of mode k by its dominant DOF (mass-weighted)."""
        # Use the mass-weighted eigvec so 'dominant' is physically meaningful.
        if self.eigvecs_mw.size:
            v = self.eigvecs_mw[:, k]
        else:
            v = self.eigvecs[:, k]
        idx = int(np.argmax(np.abs(v)))
        return DOF_NAMES[idx]

    def __str__(self) -> str:
        ex, ey, ez = self.eq_r * 1e3
        lines = [
            "--- Normal modes (Hessian eigendecomposition) ---",
            f"  equilibrium r        = ({ex:+.4f}, {ey:+.4f}, {ez:+.4f}) mm",
            f"  equilibrium (theta,phi) = ({self.eq_theta:+.4f}, {self.eq_phi:+.4f}) rad",
            f"  U_eq                 = {self.U0:+.6e} J",
            "",
            "  mode  f [Hz]      stable?  dominant DOF  composition (|v_i|^2 in %)",
        ]
        order = np.argsort(self.f_Hz)  # ascending; complex/unstable will trail
        for rank, k in enumerate(order):
            f = self.f_Hz[k]
            stable = self.eigvals[k] > 0
            # Mass-weighted composition: |w_i|^2 are dimensionless and sum to 1.
            v_src = self.eigvecs_mw if self.eigvecs_mw.size else self.eigvecs
            v2 = np.abs(v_src[:, k]) ** 2
            v2 = v2 / v2.sum()
            comp = "  ".join(f"{n}:{p*100:5.1f}" for n, p in zip(DOF_NAMES, v2))
            tag = "yes" if stable else " no"
            f_repr = f"{f:11.4f}" if np.isfinite(f) else "        nan"
            lines.append(f"  {rank+1:>2}    {f_repr}    {tag}     "
                         f"{self.label(k):>5}        {comp}")
        if self.diag_anharm.size:
            lines.append("")
            lines.append("  per-DOF diagonal quartic coefficient k4 (along chart axes):")
            for n, v in zip(DOF_NAMES, self.diag_anharm):
                lines.append(f"    k4_{n:5s} = {v: .4e}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "eq_r_m":    self.eq_r.tolist(),
            "eq_theta_rad": float(self.eq_theta),
            "eq_phi_rad":   float(self.eq_phi),
            "U0_J":      float(self.U0),
            "H":         self.H.tolist(),
            "M_diag":    np.diag(self.M).tolist(),
            "eigvals":   self.eigvals.tolist(),
            "eigvecs":   self.eigvecs.tolist(),
            "f_Hz":      self.f_Hz.tolist(),
            "dof_names": list(DOF_NAMES),
            "h_trans":   self.h_trans,
            "h_ang":     self.h_ang,
            "diag_anharm": self.diag_anharm.tolist(),
        }


# ---------------------------------------------------------------------------

def _U_at(sctmesh: SCTrapMesh, q: np.ndarray, m_mag: float, mass: float,
          g_vec: np.ndarray | None = None,
          particle=None,
          **solver_kwargs) -> float:
    """Total potential energy at chart-coordinate q = (x, y, z, theta, phi).

    When `particle` is provided, the volumetric energy path
    (`U_total_particle`) is used in place of the point-dipole `U_mag`.
    """
    from .potential import (DEFAULT_GRAVITY, U_total_particle,
                            gravity_potential)
    r0 = q[:3]
    theta, phi = float(q[3]), float(q[4])
    g = DEFAULT_GRAVITY if g_vec is None else g_vec
    if particle is not None:
        vol_kwargs = {k: v for k, v in solver_kwargs.items()
                      if k != "subtract_singularity"}
        return U_total_particle(sctmesh, particle, r0, theta, phi,
                                g_vec=g, **vol_kwargs)
    m_vec = dipole_moment(m_mag, theta, phi)
    return (U_mag(sctmesh, m_vec, r0, **solver_kwargs)
            + gravity_potential(r0, mass, g))


def normal_modes(
    sctmesh: SCTrapMesh,
    eq_r: np.ndarray,
    m_mag: float,
    *,
    eq_theta: float = 0.0,
    eq_phi: float   = 0.0,
    mass: float    = MAGNET_MASS,
    inertia: float | tuple | list | np.ndarray = MAGNET_INERTIA,
    h_trans: float = 50e-6,
    h_ang: float   = 1e-3,
    g_vec: np.ndarray | None = None,
    compute_anharmonic: bool = False,
    n_diag_pts: int = 5,
    verbose: bool = False,
    particle=None,
    **solver_kwargs,
) -> NormalModes:
    """Build the 5x5 Hessian at equilibrium and diagonalise it.

    Parameters
    ----------
    sctmesh    : SCTrapMesh
    eq_r       : (3,) equilibrium position [m]
    m_mag      : |m| [A m^2] (orientation set by eq_theta / eq_phi)
    eq_theta, eq_phi : equilibrium dipole orientation [rad]
    mass, inertia    : effective masses for translation / libration
    h_trans, h_ang   : stencil half-widths for translation / angular DOFs
    compute_anharmonic : if True, additionally fit a quartic per DOF
                         (uses an n_diag_pts stencil along each chart axis).
    """
    eq_r = np.asarray(eq_r, dtype=float).reshape(3)

    # Chart-coordinate vector (5,)
    q0 = np.array([eq_r[0], eq_r[1], eq_r[2], float(eq_theta), float(eq_phi)])
    h_vec = np.array([h_trans, h_trans, h_trans, h_ang, h_ang])

    # If a particle object was supplied, prefer its mass/inertia (the values
    # passed in `mass=`/`inertia=` are only used as fallbacks).  The angular
    # mass-matrix entries are (I_theta, I_phi) which correspond to libration
    # about the body-y and body-z axes -> inertia_yy, inertia_zz.
    if particle is not None:
        mass = float(particle.mass)
        inertia = (float(particle.inertia_yy),
                   float(particle.inertia_zz))

    # Mass matrix M = diag(m, m, m, I_theta, I_phi)
    # `inertia` may be:
    #   - scalar          -> isotropic libration, I_theta = I_phi = inertia
    #   - 2-tuple/2-array -> (I_theta, I_phi) (anisotropic libration)
    if np.isscalar(inertia):
        I_theta = I_phi = float(inertia)
    else:
        arr = np.asarray(inertia, dtype=float).ravel()
        if arr.size == 1:
            I_theta = I_phi = float(arr[0])
        elif arr.size == 2:
            I_theta, I_phi = float(arr[0]), float(arr[1])
        else:
            raise ValueError("inertia must be scalar or 2-element sequence")
    M = np.diag([mass, mass, mass, I_theta, I_phi])

    def U(q):
        return _U_at(sctmesh, q, m_mag, mass, g_vec=g_vec,
                     particle=particle, **solver_kwargs)

    if verbose:
        print(f"  centre U(q0) = ", end="", flush=True)
    U0 = U(q0)
    if verbose:
        print(f"{U0:.6e} J", flush=True)

    H = np.zeros((5, 5))

    # Diagonal entries: U(q +/- h e_i)
    for i in range(5):
        ei = np.zeros(5); ei[i] = h_vec[i]
        Up = U(q0 + ei)
        Um = U(q0 - ei)
        H[i, i] = (Up - 2.0 * U0 + Um) / (h_vec[i] ** 2)
        if verbose:
            print(f"  H[{i},{i}] = {H[i,i]: .4e}", flush=True)

    # Off-diagonal entries: 4-point stencil
    for i in range(5):
        for j in range(i + 1, 5):
            ei = np.zeros(5); ei[i] = h_vec[i]
            ej = np.zeros(5); ej[j] = h_vec[j]
            Upp = U(q0 + ei + ej)
            Upm = U(q0 + ei - ej)
            Ump = U(q0 - ei + ej)
            Umm = U(q0 - ei - ej)
            val = (Upp - Upm - Ump + Umm) / (4.0 * h_vec[i] * h_vec[j])
            H[i, j] = val
            H[j, i] = val
            if verbose:
                print(f"  H[{i},{j}] = {val: .4e}", flush=True)

    # Generalised eigenproblem: H v = w^2 M v   ->   M^-1 H v = w^2 v
    # Solve symmetrically: M is diag, take Minv12 = diag(1/sqrt(M_ii))
    Minv12 = np.diag(1.0 / np.sqrt(np.diag(M)))
    Hsym = Minv12 @ H @ Minv12
    Hsym = 0.5 * (Hsym + Hsym.T)            # symmetrise (kill FD asymmetry)
    omega2, V = np.linalg.eigh(Hsym)
    # Convert back to physical eigenvectors v = Minv12 @ V (so v^T M v = I)
    eigvecs = Minv12 @ V

    # Frequencies (Hz). Negative omega2 -> unstable mode -> negative f for tagging.
    sign = np.sign(omega2)
    f = sign * np.sqrt(np.abs(omega2)) / (2.0 * np.pi)

    # Optional diagonal quartic fit
    diag_anharm = np.zeros(0)
    if compute_anharmonic:
        if n_diag_pts % 2 == 0 or n_diag_pts < 5:
            raise ValueError("n_diag_pts must be odd and >= 5")
        half = (n_diag_pts - 1) // 2
        diag_anharm = np.empty(5)
        for i in range(5):
            offsets = np.linspace(-half, half, n_diag_pts) * h_vec[i]
            U_line = np.empty(n_diag_pts)
            for k, dq in enumerate(offsets):
                qk = q0.copy()
                qk[i] += dq
                U_line[k] = U(qk)
            # Fit U = U0 + (1/2) k2 q^2 + (1/24) k4 q^4 (assume q-q_eq centred)
            c4, c3, c2, c1, c0 = np.polyfit(offsets, U_line, 4)
            diag_anharm[i] = 24.0 * c4   # convention U = ... + (1/24) k4 q^4
            if verbose:
                print(f"  k4_{DOF_NAMES[i]} = {diag_anharm[i]: .4e}", flush=True)

    return NormalModes(
        eq_r=eq_r, eq_theta=float(eq_theta), eq_phi=float(eq_phi),
        H=H, M=M,
        eigvals=omega2, eigvecs=eigvecs, eigvecs_mw=V, f_Hz=f,
        U0=float(U0),
        h_trans=h_trans, h_ang=h_ang,
        diag_anharm=diag_anharm,
    )
