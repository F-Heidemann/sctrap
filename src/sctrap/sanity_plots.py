"""Extra diagnostic plots for the quickstart pipeline.

These supplement the small set produced by `report.write_report`. They are
deliberately written to be cheap (no extra FEM solves where possible — the
expensive ones are clearly labelled in their docstrings).

Available plots:

    plot_mesh_overview          mesh skeleton + SC facets in 3D
    plot_particle_diagram       schematic of the composite particle (cubes + sphere)
    plot_U_long_axes            long-range U(x), U(y), U(z) line scans
    plot_B_induced_slice        |B_induced| on a vertical (xz) slice
    plot_frequency_bar          stable/unstable mode bar chart (Hz)
    plot_equilibrium_residual   gradient + curvature sanity table -> .png

Each function is independent and silently no-ops if matplotlib is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import G_GRAV, MAGNET_MASS
from .dipole import dipole_moment, B_dipole
from .mesh import SCTrapMesh
from .modes import NormalModes, DOF_NAMES
from .potential import U_mag
from .solver import solve_phi, B_induced_at


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                                # pragma: no cover
        print(f"  (matplotlib unavailable: {exc}; skipping plot)")
        return None


# ---------------------------------------------------------------------------
# 1. Mesh overview
# ---------------------------------------------------------------------------

def plot_mesh_overview(sctmesh: SCTrapMesh, out_path: Path,
                       eq_r: Optional[np.ndarray] = None) -> None:
    plt = _plt()
    if plt is None:
        return
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

    p = sctmesh.mesh.p                                 # (3, n_pts)
    sc_f = sctmesh.sc_facets
    ff_f = sctmesh.ff_facets
    facets = sctmesh.mesh.facets                       # (3, n_facets)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    def _draw_facets(face_idx, color, label, alpha):
        if face_idx is None or len(face_idx) == 0:
            return
        # Sample for speed
        idx = face_idx
        if len(idx) > 4000:
            idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
        tri = facets[:, idx]                           # (3, k)
        verts = p[:, tri]                              # (3, 3, k)
        verts = np.transpose(verts, (2, 1, 0))         # (k, 3 verts, 3 xyz)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        coll = Poly3DCollection(verts, alpha=alpha, facecolor=color,
                                edgecolor=(0, 0, 0, 0.15), linewidth=0.2)
        ax.add_collection3d(coll)

    _draw_facets(sc_f, "#3171b5", "SC", 0.45)
    _draw_facets(ff_f, "#d6604d", "FF", 0.20)

    # Sample volume vertices for scale
    n_show = min(2000, p.shape[1])
    pick = np.random.default_rng(1).choice(p.shape[1], n_show, replace=False)
    ax.scatter(p[0, pick] * 1e3, p[1, pick] * 1e3, p[2, pick] * 1e3,
               s=1, alpha=0.15, color="0.4")

    if eq_r is not None:
        ax.scatter([eq_r[0] * 1e3], [eq_r[1] * 1e3], [eq_r[2] * 1e3],
                   s=80, color="red", marker="x", label="r_eq")

    pmin, pmax = sctmesh.bounding_box()
    ax.set_xlim(pmin[0]*1e3, pmax[0]*1e3)
    ax.set_ylim(pmin[1]*1e3, pmax[1]*1e3)
    ax.set_zlim(pmin[2]*1e3, pmax[2]*1e3)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.set_title(f"Mesh: {p.shape[1]} pts, "
                 f"{len(sc_f)} SC facets, {len(ff_f)} FF facets")
    ax.legend(["SC", "FF", "vertices", "r_eq"], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Particle schematic
# ---------------------------------------------------------------------------

def plot_particle_diagram(particle, out_path: Path) -> None:
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    a = particle.cube_edge
    N = particle.n_cubes
    rsp = particle.sphere_position

    for i in range(N):
        x_left = (i - N / 2.0) * a
        rect = plt.Rectangle((x_left * 1e3, -0.5 * a * 1e3),
                             a * 1e3, a * 1e3,
                             facecolor="#444", edgecolor="black")
        ax.add_patch(rect)
        # Magnetisation arrow inside each cube
        ax.annotate("", xy=((x_left + 0.85 * a) * 1e3, 0),
                    xytext=((x_left + 0.15 * a) * 1e3, 0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

    # Project bead into xz plane (y is into the page)
    if particle.sphere_radius > 0:
        circ = plt.Circle((rsp[0] * 1e3, rsp[2] * 1e3),
                          particle.sphere_radius * 1e3,
                          facecolor="#9ec6f5", edgecolor="black", alpha=0.85)
        ax.add_patch(circ)

    # COM marker
    com = particle.com
    ax.plot(com[0] * 1e3, com[2] * 1e3, "rx", ms=12, mew=2,
            label=f"COM (z={com[2]*1e6:+.1f} um)")
    # Bar axis
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(0, color="0.7", lw=0.5)

    span = max(particle.bar_length,
               np.linalg.norm(rsp) + particle.sphere_radius) * 1.4
    ax.set_xlim(-span * 1e3, span * 1e3)
    ax.set_ylim(-span * 0.9 * 1e3, span * 0.6 * 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]   (bar = dipole axis)")
    ax.set_ylabel("z  [mm]   (vertical at equilibrium)")
    ax.set_title(
        f"Particle: {N} cubes ({particle.cube_edge*1e3:.3f} mm) + "
        f"sphere R={particle.sphere_radius*1e3:.3f} mm   "
        f"M={particle.mass*1e6:.3f} mg")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.01, 0.98,
            f"I_xx  = {particle.inertia_xx:.3e}\n"
            f"I_yy  = {particle.inertia_yy:.3e}\n"
            f"I_zz  = {particle.inertia_zz:.3e}\n"
            f"|m|   = {particle.moment:.3e} A m^2",
            transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Long-range U(x), U(y), U(z) line scans (each is an FEM solve per point)
# ---------------------------------------------------------------------------

def plot_U_long_axes(
    sctmesh: SCTrapMesh,
    eq_r: np.ndarray,
    eq_theta: float, eq_phi: float,
    m_mag: float, mass: float,
    span: float,
    n_pts: int,
    out_path: Path,
    **solver_kwargs,
) -> dict:
    plt = _plt()
    m_vec = dipole_moment(m_mag, eq_theta, eq_phi)

    pmin, pmax = sctmesh.bounding_box()
    margin = 0.05 * (pmax - pmin)
    pmin = pmin + margin
    pmax = pmax - margin

    fig, axes = (plt.subplots(1, 3, figsize=(15, 4)) if plt is not None
                  else (None, [None, None, None]))
    out = {}
    for i, (axis_name, ax_idx) in enumerate(zip("xyz", range(3))):
        # Scan along axis through equilibrium, clipped to bounding box
        lo = max(eq_r[ax_idx] - span, pmin[ax_idx])
        hi = min(eq_r[ax_idx] + span, pmax[ax_idx])
        coords = np.linspace(lo, hi, n_pts)
        U = np.empty(n_pts)
        for k, c in enumerate(coords):
            r0 = eq_r.copy(); r0[ax_idx] = c
            U[k] = (U_mag(sctmesh, m_vec, r0, **solver_kwargs)
                    + mass * G_GRAV * float(r0[2]))
        out[axis_name] = {"coords_m": coords.tolist(), "U_J": U.tolist()}

        if plt is not None:
            ax = axes[i]
            ax.plot((coords - eq_r[ax_idx]) * 1e3, (U - U.min()) * 1e21, "o-")
            ax.axvline(0, color="red", lw=0.7, ls="--")
            ax.set_xlabel(f"{axis_name} − {axis_name}_eq  [mm]")
            ax.set_ylabel("U − U_min  [zJ]")
            ax.set_title(f"long-range U along {axis_name} (FEM, {n_pts} pts)")
            ax.grid(alpha=0.3)
    if plt is not None:
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. |B_induced| on a vertical slice (one FEM solve, many evaluations)
# ---------------------------------------------------------------------------

def plot_B_induced_slice(
    sctmesh: SCTrapMesh,
    eq_r: np.ndarray,
    eq_theta: float, eq_phi: float,
    m_mag: float,
    out_path: Path,
    span: float = 1.5e-3,
    n_pts: int = 25,
    **solver_kwargs,
) -> None:
    plt = _plt()
    if plt is None:
        return
    m_vec = dipole_moment(m_mag, eq_theta, eq_phi)
    phi = solve_phi(sctmesh, m_vec, eq_r, **solver_kwargs)

    xs = np.linspace(-span, span, n_pts)
    zs = np.linspace(-span, span, n_pts)
    Bmag = np.empty((n_pts, n_pts))
    for ix, dx in enumerate(xs):
        for iz, dz in enumerate(zs):
            r = eq_r.copy(); r[0] += dx; r[2] += dz
            try:
                B = B_induced_at(phi, r)
            except Exception:
                Bmag[iz, ix] = np.nan; continue
            Bmag[iz, ix] = np.linalg.norm(B)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.contourf(xs * 1e3, zs * 1e3, Bmag * 1e3, levels=20, cmap="magma")
    ax.plot(0, 0, "cx", ms=12, mew=2, label="r_eq")
    ax.set_xlabel("x − x_eq  [mm]")
    ax.set_ylabel("z − z_eq  [mm]")
    ax.set_title("|B_induced|  [mT]   (xz slice through equilibrium)")
    fig.colorbar(im, ax=ax, label="mT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Frequency bar chart
# ---------------------------------------------------------------------------

def plot_frequency_bar(modes: NormalModes, out_path: Path,
                       compare_f_Hz: Optional[float] = None,
                       compare_label: str = "") -> None:
    plt = _plt()
    if plt is None:
        return
    order = np.argsort(np.where(modes.eigvals > 0,
                                np.sqrt(np.abs(modes.eigvals)),
                                -np.sqrt(np.abs(modes.eigvals))))
    f_signed = modes.f_Hz[order]
    labels = [modes.label(int(k)) for k in order]
    colors = ["#3171b5" if v > 0 else "#d6604d" for v in modes.eigvals[order]]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = np.arange(len(f_signed))
    ax.bar(xs, np.abs(f_signed), color=colors, edgecolor="black")
    for x, f, lab in zip(xs, f_signed, labels):
        ax.text(x, abs(f) * 1.02 + 1, f"{f:+.2f} Hz\n({lab})",
                ha="center", va="bottom", fontsize=8)
    if compare_f_Hz is not None:
        ax.axhline(compare_f_Hz, color="green", lw=1.2, ls="--",
                   label=f"{compare_label or 'reference'}: {compare_f_Hz:.2f} Hz")
        ax.legend()
    ax.set_xticks(xs)
    ax.set_xticklabels([f"#{i+1}" for i in xs])
    ax.set_ylabel("|f|  [Hz]   (red bar = unstable, omega^2 < 0)")
    ax.set_title("Normal-mode frequencies")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Equilibrium-residual sanity card
# ---------------------------------------------------------------------------

def plot_equilibrium_residual(
    modes: NormalModes,
    out_path: Path,
    grad_estimate: Optional[np.ndarray] = None,
) -> None:
    """Render a small diagnostic image: gradient at eq (should be ~0),
    diagonal Hessian entries (sign tells stability), off-diagonal coupling."""
    plt = _plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    H = modes.H
    im0 = axes[0].imshow(np.sign(H) * np.log10(np.abs(H) + 1e-30),
                         cmap="RdBu_r")
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(DOF_NAMES)
    axes[0].set_yticks(range(5)); axes[0].set_yticklabels(DOF_NAMES)
    axes[0].set_title("sign(H_ij) · log10|H_ij|")
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f"{H[i,j]:+.1e}",
                         ha="center", va="center", fontsize=7,
                         color="black" if abs(H[i, j]) < 1e-10 else "white")
    fig.colorbar(im0, ax=axes[0])

    if grad_estimate is not None:
        axes[1].bar(range(5), grad_estimate, edgecolor="black")
        axes[1].set_xticks(range(5)); axes[1].set_xticklabels(DOF_NAMES)
        axes[1].set_title("∂U/∂q at equilibrium  (should be ~0)")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5,
                     "(gradient not provided; supply grad_estimate to fill)",
                     ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
