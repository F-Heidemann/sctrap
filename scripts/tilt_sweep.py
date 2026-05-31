"""Sweep the trap tilt angle and watch the 5 mode frequencies move.

Reads the *same* parameters file as `quickstart_elliptical.py` (default:
scripts/parameters.py); the only knob that this script overrides is
TILT_ANGLE — it loops over a list of angles. At each tilt it re-finds the
equilibrium, rebuilds the 5x5 Hessian, and records all five mode
frequencies plus their mass-weighted compositions.

Outputs:
    results/<OUT_DIR>_tilt_sweep/
        sweep.csv        rows: angle_deg, f1..f5, dominant DOF labels
        sweep.json       full eigvecs + Hessian per angle (cumulative)
        f_vs_tilt.png    five frequency branches vs angle
        mode_track.png   composition of each branch vs angle (heatmap)

Usage:
    python3 scripts/tilt_sweep.py                                 # default
    python3 scripts/tilt_sweep.py examples/parameters_vinante.py  # preset

Edit `TILT_ANGLES_DEG` at the top of this file to control the sweep.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_ROOT = REPO / "results"

# ----- the only knob that lives in this script ------------------------------
TILT_ANGLES_DEG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def _load_parameters(arg: str | None):
    path = Path(arg).resolve() if arg else (HERE / "parameters.py")
    spec = importlib.util.spec_from_file_location("user_parameters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


from sctrap.dipole import dipole_moment
from sctrap.frequencies import find_equilibrium
from sctrap.generators import elliptical_cavity
from sctrap.mesh import load_msh
from sctrap.modes import normal_modes, DOF_NAMES
from sctrap.potential import gravity_from_tilt

# import quickstart helpers w/o running its main()
sys.path.insert(0, str(HERE))
from quickstart_elliptical import build_particle, particle_inertia_for_modes  # noqa: E402


def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    P, params_path = _load_parameters(arg)
    out_dir = RESULTS_ROOT / f"{P.OUT_DIR}_tilt_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "_mesh" / "cavity.msh"
    mesh_path.parent.mkdir(exist_ok=True)

    # one mesh for the whole sweep
    elliptical_cavity(P.CAVITY_A, P.CAVITY_B, P.CAVITY_HEIGHT,
                       mesh_size_sc=P.MESH_SIZE,
                       mesh_size_near=P.MESH_SIZE_NEAR,
                       refine_distance=P.MESH_REFINE_DIST,
                       output_file=str(mesh_path), verbose=False)
    sct = load_msh(mesh_path)
    print(f"  mesh: {sct}")

    particle = build_particle(P)
    print(particle)

    solver_kwargs = dict(element=P.ELEMENT,
                         subtract_singularity=P.SUBTRACT_SINGULARITY)
    m_vec = dipole_moment(particle.moment, P.EQ_THETA, P.EQ_PHI)

    rows = []
    eq_r_prev = np.asarray(P.EQ_GUESS, dtype=float).copy()
    cumulative = []

    for theta_deg in TILT_ANGLES_DEG:
        theta = np.deg2rad(theta_deg)
        g_vec = gravity_from_tilt(theta, P.TILT_AXIS)
        print(f"\n--- tilt = {theta_deg:.2f} deg about {P.TILT_AXIS} ---")
        t0 = time.time()
        eq_r = find_equilibrium(sct, m_vec, eq_r_prev,
                                 mass=particle.mass, g_vec=g_vec,
                                 **solver_kwargs)
        print(f"  r_eq = ({eq_r[0]*1e6:+.1f}, {eq_r[1]*1e6:+.1f}, "
              f"{eq_r[2]*1e6:+.1f}) um   ({time.time()-t0:.1f} s)")
        nm = normal_modes(sct, eq_r, particle.moment,
                            eq_theta=P.EQ_THETA, eq_phi=P.EQ_PHI,
                            mass=particle.mass,
                            inertia=particle_inertia_for_modes(particle),
                            h_trans=P.H_TRANS, h_ang=P.H_ANG,
                            g_vec=g_vec, **solver_kwargs)
        print(f"  f_Hz = {np.round(nm.f_Hz, 2)}")

        order = np.argsort(nm.f_Hz)
        row = [theta_deg, *nm.f_Hz[order].tolist(),
               *(nm.label(int(k)) for k in order)]
        rows.append(row)
        cumulative.append({
            "tilt_deg":    float(theta_deg),
            "g_vec":       g_vec.tolist(),
            "eq_r_m":      eq_r.tolist(),
            "f_Hz":        nm.f_Hz.tolist(),
            "eigvecs_mw":  nm.eigvecs_mw.tolist(),
            "label_order": [nm.label(int(k)) for k in order],
        })
        eq_r_prev = eq_r            # warm-start next angle

        # rewrite outputs after each angle so a crash leaves something useful
        with (out_dir / "sweep.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tilt_deg", *(f"f{k+1}_Hz" for k in range(5)),
                        *(f"label{k+1}" for k in range(5))])
            for r in rows: w.writerow(r)
        (out_dir / "sweep.json").write_text(json.dumps(cumulative, indent=2))

    # ------- plots ----------------------------------------------------------
    plt = _try_plt()
    if plt is None:
        print("  (matplotlib unavailable; CSV/JSON only)")
        return

    angles = np.array([r[0] for r in rows])
    F = np.array([r[1:6] for r in rows])           # (n_angles, 5)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in range(5):
        ax.plot(angles, F[:, k], "o-", label=f"branch {k+1}")
    ax.set_xlabel("tilt [deg]")
    ax.set_ylabel("|f|  [Hz]")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Mode frequencies vs trap tilt")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "f_vs_tilt.png", dpi=120)
    plt.close(fig)

    # mode-tracking heatmap: at each tilt, plot |w_i|^2 of each branch
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
    for k in range(5):
        comp = np.array([np.abs(c["eigvecs_mw"])[:, np.argsort(c["f_Hz"])[k]] ** 2
                          for c in cumulative])
        comp = comp / comp.sum(axis=1, keepdims=True)
        axes[k].imshow(comp.T, aspect="auto", origin="lower",
                       extent=[angles[0], angles[-1], -0.5, 4.5],
                       vmin=0, vmax=1, cmap="viridis")
        axes[k].set_yticks(range(5)); axes[k].set_yticklabels(DOF_NAMES)
        axes[k].set_xlabel("tilt [deg]")
        axes[k].set_title(f"branch {k+1}")
    fig.suptitle("Mass-weighted DOF composition along each branch")
    fig.tight_layout()
    fig.savefig(out_dir / "mode_track.png", dpi=120)
    plt.close(fig)

    print(f"\nDONE -> {out_dir}")


if __name__ == "__main__":
    main()
