"""Write a full report (JSON/CSV/TXT + matplotlib plots) for a NormalModes
result computed at a given equilibrium.

Layout written to `out_dir`:
    result.json        full Hessian, eigenvectors, eigenvalues, frequencies
    modes.csv          one row per mode (rank, label, f_Hz, eigenvector, ...)
    summary.txt        human-readable summary (`str(NormalModes)`)
    U_along_DOFs.png   5-panel U(q) scan around equilibrium with parabola overlay
    U_xz_slice.png     2-D U(x, z) contour at y = y_eq
    U_xy_slice.png     2-D U(x, y) contour at z = z_eq
    mode_shapes.png    bar chart of |v_i|^2 per mode
    anharm.png         residual U - U_parabola per DOF (if anharmonic fit was done)

The plots are optional: they're only created if matplotlib imports cleanly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .config import G_GRAV, MAGNET_MASS
from .dipole import dipole_moment
from .mesh import SCTrapMesh
from .modes import NormalModes, DOF_NAMES
from .potential import U_mag


# ---------------------------------------------------------------------------
# Text + machine-readable artifacts
# ---------------------------------------------------------------------------

def write_text_artifacts(modes: NormalModes, out_dir: Path,
                         uncertainty=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = str(modes)
    payload = modes.as_dict()
    if uncertainty is not None:
        summary += "\n\n  Stencil-Richardson uncertainty band  f ± df  [Hz]:\n"
        for k, (f, df) in enumerate(zip(uncertainty.f_Hz, uncertainty.df_Hz)):
            summary += f"    mode {k+1}:  {f:11.4f}  ±  {df:8.4f}\n"
        payload["uncertainty"] = uncertainty.as_dict()

    (out_dir / "summary.txt").write_text(summary + "\n")
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))

    with (out_dir / "modes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "f_Hz", "stable", "label",
                    "v_x", "v_y", "v_z", "v_theta", "v_phi"])
        order = np.argsort(modes.f_Hz)
        for rank, k in enumerate(order):
            f = modes.f_Hz[k]
            stable = bool(modes.eigvals[k] > 0)
            v = modes.eigvecs[:, k]
            v = v / np.linalg.norm(v)
            w.writerow([rank + 1, f, stable, modes.label(k),
                        *[float(vi) for vi in v]])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")        # headless
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:           # pragma: no cover
        print(f"  (matplotlib unavailable: {exc}; skipping plots)")
        return None


def plot_U_along_dofs(
    sctmesh: SCTrapMesh,
    modes: NormalModes,
    m_mag: float,
    out_path: Path,
    n_pts: int = 21,
    span_factor: float = 4.0,
    mass: float = MAGNET_MASS,
    **solver_kwargs,
) -> None:
    """5-panel U(q) scan with parabola overlay using H[i,i] from the Hessian."""
    plt = _try_matplotlib()
    if plt is None:
        return

    h_vec = np.array([modes.h_trans] * 3 + [modes.h_ang] * 2)
    half = span_factor   # number of stencil half-widths to either side
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    q0 = np.array([*modes.eq_r, modes.eq_theta, modes.eq_phi])

    for i, name in enumerate(DOF_NAMES):
        offsets = np.linspace(-half, half, n_pts) * h_vec[i]
        U = np.empty(n_pts)
        for k, dq in enumerate(offsets):
            q = q0.copy(); q[i] += dq
            r0 = q[:3]
            m_vec = dipole_moment(m_mag, q[3], q[4])
            U[k] = (U_mag(sctmesh, m_vec, r0, **solver_kwargs)
                    + mass * G_GRAV * float(r0[2]))
        ax = axes[i]
        ax.plot(offsets, (U - modes.U0) * 1e21, "o-", label="U − U₀")
        # Parabola from H_ii
        Up = 0.5 * modes.H[i, i] * offsets ** 2
        ax.plot(offsets, Up * 1e21, "--", label="½ H_ii q²")
        ax.set_xlabel(f"δ{name}")
        ax.set_ylabel("U − U₀  [zJ]")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_potential_slice(
    sctmesh: SCTrapMesh,
    modes: NormalModes,
    m_mag: float,
    out_path: Path,
    plane: str,                # "xz" or "xy"
    span: float = 1.5e-3,
    n_pts: int = 25,
    mass: float = MAGNET_MASS,
    **solver_kwargs,
) -> None:
    plt = _try_matplotlib()
    if plt is None:
        return

    if plane == "xz":
        a_axis, b_axis, label_a, label_b = 0, 2, "x", "z"
    elif plane == "xy":
        a_axis, b_axis, label_a, label_b = 0, 1, "x", "y"
    else:
        raise ValueError(f"unknown plane '{plane}'")

    a_vals = np.linspace(-span, span, n_pts)
    b_vals = np.linspace(-span, span, n_pts)
    Z = np.empty((n_pts, n_pts))

    q_eq = np.array([*modes.eq_r, modes.eq_theta, modes.eq_phi])
    m_vec_eq = dipole_moment(m_mag, modes.eq_theta, modes.eq_phi)

    for ia, da in enumerate(a_vals):
        for ib, db in enumerate(b_vals):
            r0 = modes.eq_r.copy()
            r0[a_axis] = modes.eq_r[a_axis] + da
            r0[b_axis] = modes.eq_r[b_axis] + db
            try:
                Z[ib, ia] = (U_mag(sctmesh, m_vec_eq, r0, **solver_kwargs)
                             + mass * G_GRAV * float(r0[2]))
            except ValueError:                  # outside the mesh
                Z[ib, ia] = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf((a_vals) * 1e3, (b_vals) * 1e3,
                     (Z - modes.U0) * 1e21, levels=20, cmap="viridis")
    ax.plot(0, 0, "rx", ms=12, mew=2, label="r_eq")
    ax.set_xlabel(f"{label_a} − {label_a}_eq  [mm]")
    ax.set_ylabel(f"{label_b} − {label_b}_eq  [mm]")
    ax.set_title(f"U − U₀  [zJ]   ({plane}-slice through equilibrium)")
    fig.colorbar(im, ax=ax, label="zJ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mode_shapes(modes: NormalModes, out_path: Path) -> None:
    plt = _try_matplotlib()
    if plt is None:
        return

    order = np.argsort(modes.f_Hz)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.16
    xs = np.arange(5)
    for rank, k in enumerate(order):
        v2 = np.abs(modes.eigvecs[:, k]) ** 2
        v2 = v2 / v2.sum()
        f = modes.f_Hz[k]
        ax.bar(xs + (rank - 2) * width, v2, width=width,
               label=f"#{rank+1}: {f:.2f} Hz ({modes.label(k)})")
    ax.set_xticks(xs)
    ax.set_xticklabels(DOF_NAMES)
    ax.set_ylabel("|v_i|²  (normalised)")
    ax.set_title("Mode-shape composition")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def write_report(
    sctmesh: SCTrapMesh,
    modes: NormalModes,
    m_mag: float,
    out_dir: str | Path,
    *,
    plot_slice_span: float = 1.5e-3,
    plot_slice_pts: int    = 21,
    plot_dof_pts: int      = 17,
    plot_dof_span: float   = 4.0,
    mass: float = MAGNET_MASS,
    skip_plots: bool = False,
    uncertainty=None,
    **solver_kwargs,
) -> Path:
    """Write all artifacts into `out_dir`. Returns the resolved path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_text_artifacts(modes, out_dir, uncertainty=uncertainty)

    if skip_plots:
        return out_dir

    plot_mode_shapes(modes, out_dir / "mode_shapes.png")

    plot_U_along_dofs(
        sctmesh, modes, m_mag,
        out_dir / "U_along_DOFs.png",
        n_pts=plot_dof_pts, span_factor=plot_dof_span,
        mass=mass, **solver_kwargs,
    )

    plot_potential_slice(
        sctmesh, modes, m_mag,
        out_dir / "U_xz_slice.png",
        plane="xz", span=plot_slice_span, n_pts=plot_slice_pts,
        mass=mass, **solver_kwargs,
    )
    plot_potential_slice(
        sctmesh, modes, m_mag,
        out_dir / "U_xy_slice.png",
        plane="xy", span=plot_slice_span, n_pts=plot_slice_pts,
        mass=mass, **solver_kwargs,
    )

    return out_dir
