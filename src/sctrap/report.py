"""Write a full report (JSON/CSV/TXT + matplotlib plots) for a NormalModes
result computed at a given equilibrium.

Layout written to `out_dir`:
    result.json        full Hessian, eigenvectors, eigenvalues, frequencies
    modes.csv          one row per mode (rank, label, f_Hz, eigenvector, ...)
    summary.txt        human-readable summary (`str(NormalModes)`)
    U_along_DOFs.png   5-panel U(q) scan around equilibrium with parabola overlay
    U_along_DOFs/      per-DOF raw scan data as CSV (U_x.csv … U_phi.csv +
                       combined U_along_DOFs.csv), SI units, for re-plotting
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


#: SI unit of each DOF's scan offset, for the exported data headers.
_DOF_OFFSET_UNIT = {"x": "m", "y": "m", "z": "m",
                    "theta": "rad", "phi": "rad"}


def compute_U_along_dofs(
    sctmesh: SCTrapMesh,
    modes: NormalModes,
    m_mag: float,
    n_pts: int = 21,
    span_factor: float = 4.0,
    mass: float = MAGNET_MASS,
    **solver_kwargs,
) -> dict:
    """Scan the total potential U(q) along each of the five DOFs about the
    equilibrium, one DOF varied at a time (the others held at equilibrium).

    Returns a dict keyed by DOF name (``x, y, z, theta, phi``); each value is a
    dict with arrays ``offset`` (δq, SI: m or rad), ``U`` (absolute total
    potential [J]), ``U_minus_U0`` [J], and ``U_parabola`` (the harmonic fit
    ½·H_ii·δq² [J] used for the trap frequency), plus scalars ``H_ii`` [SI],
    ``h_stencil`` (the Hessian half-width [SI]) and ``unit``. This is the raw
    data behind the ``U_along_DOFs`` plot; it is written to CSV so users can
    re-plot and fit it themselves.
    """
    h_vec = np.array([modes.h_trans] * 3 + [modes.h_ang] * 2)
    q0 = np.array([*modes.eq_r, modes.eq_theta, modes.eq_phi])

    data: dict = {}
    for i, name in enumerate(DOF_NAMES):
        offsets = np.linspace(-span_factor, span_factor, n_pts) * h_vec[i]
        U = np.empty(n_pts)
        for k, dq in enumerate(offsets):
            q = q0.copy(); q[i] += dq
            r0 = q[:3]
            m_vec = dipole_moment(m_mag, q[3], q[4])
            U[k] = (U_mag(sctmesh, m_vec, r0, **solver_kwargs)
                    + mass * G_GRAV * float(r0[2]))
        data[name] = {
            "offset": offsets,
            "U": U,
            "U_minus_U0": U - modes.U0,
            "U_parabola": 0.5 * modes.H[i, i] * offsets ** 2,
            "H_ii": float(modes.H[i, i]),
            "h_stencil": float(h_vec[i]),
            "unit": _DOF_OFFSET_UNIT[name],
        }
    return data


def write_U_along_dofs_data(data: dict, out_dir: Path) -> None:
    """Write one CSV per DOF (``U_x.csv`` … ``U_phi.csv``) plus a combined
    ``U_along_DOFs.csv`` into ``out_dir``. All columns are SI so experimentalists
    can load and analyse them with no unit conversion. The per-DOF parabola
    column is the harmonic fit that defines the trap frequency; deviations of
    ``U_minus_U0`` from it beyond ±``h_stencil`` are the FEM noise floor
    (see the package docs), not physical anharmonicity."""
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for name, d in data.items():
        unit = d["unit"]
        header = [f"offset[{unit}]", "U[J]", "U_minus_U0[J]", "U_parabola[J]"]
        with (out_dir / f"U_{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([f"# DOF={name}  H_ii={d['H_ii']:.8e}  "
                        f"h_stencil={d['h_stencil']:.8e} {unit}"])
            w.writerow(header)
            for o, u, du, up in zip(d["offset"], d["U"],
                                    d["U_minus_U0"], d["U_parabola"]):
                row = [o, u, du, up]
                w.writerow(row)
                combined_rows.append([name, unit, o, u, du, up])

    with (out_dir / "U_along_DOFs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dof", "offset_unit", "offset", "U_J",
                    "U_minus_U0_J", "U_parabola_J"])
        w.writerows(combined_rows)


def plot_U_along_dofs(
    sctmesh: SCTrapMesh,
    modes: NormalModes,
    m_mag: float,
    out_path: Path,
    n_pts: int = 21,
    span_factor: float = 4.0,
    mass: float = MAGNET_MASS,
    write_data: bool = True,
    **solver_kwargs,
) -> dict:
    """5-panel U(q) scan with the harmonic (½ H_ii q²) overlay, and — unless
    ``write_data=False`` — the underlying per-DOF data dumped as CSV into a
    ``<out_path stem>/`` sibling directory. Returns the data dict."""
    data = compute_U_along_dofs(
        sctmesh, modes, m_mag, n_pts=n_pts,
        span_factor=span_factor, mass=mass, **solver_kwargs)

    if write_data:
        write_U_along_dofs_data(data, out_path.parent / out_path.stem)

    plt = _try_matplotlib()
    if plt is None:
        return data

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, name in enumerate(DOF_NAMES):
        d = data[name]
        offsets, h = d["offset"], d["h_stencil"]
        ax = axes[i]
        # Markers (not a connected line): the single-sample FEM noise-floor
        # excursions (e.g. the z DOF, where the C0 P2 gradient jumps as an FD
        # stencil point crosses an element face) then read as scattered points
        # rather than a spurious V-shaped "kink" in a join-the-dots line.
        ax.plot(offsets, d["U_minus_U0"] * 1e21, "o", ms=4, label="U − U₀")
        ax.plot(offsets, d["U_parabola"] * 1e21, "--", label="½ H_ii q²")
        # Shade the ±h_stencil window the Hessian (hence the trap frequency)
        # is actually evaluated over. Wiggles OUTSIDE this band are the FEM
        # noise floor and do not enter the frequencies.
        ax.axvspan(-h, h, color="0.6", alpha=0.15,
                   label="Hessian window (±h)")
        ax.set_xlabel(f"δ{name}  [{d['unit']}]")
        ax.set_ylabel("U − U₀  [zJ]")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("U along each DOF about equilibrium.  The trap frequency uses "
                 "only the shaded ±h window; structure beyond it is FEM "
                 "noise-floor, not physical.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return data


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
