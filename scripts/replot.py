"""Re-render reports + sanity plots from a cached result.json.

Usage:
    python3 scripts/replot.py results/quickstart_elliptical/

Loads the saved Hessian / equilibrium from `result.json`, regenerates the
mesh from `parameters.py`, and re-runs only the plotting stage. Useful
when a plot crashed at the end of a long run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P                 # type: ignore  noqa: E402

from sctrap.generators import elliptical_cavity
from sctrap.mesh import load_msh
from sctrap.modes import NormalModes
from sctrap.particle import CompositeParticle
from sctrap.report import write_report
from sctrap.sanity_plots import (
    plot_B_induced_slice, plot_equilibrium_residual,
    plot_frequency_bar, plot_U_long_axes,
)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        HERE.parent / "results" / P.OUT_DIR)
    payload = json.loads((out_dir / "result.json").read_text())

    eq_r = np.array(payload["eq_r_m"])
    nm = NormalModes(
        eq_r        = eq_r,
        eq_theta    = float(payload["eq_theta_rad"]),
        eq_phi      = float(payload["eq_phi_rad"]),
        H           = np.array(payload["H"]),
        M           = np.diag(np.array(payload["M_diag"])),
        eigvals     = np.array(payload["eigvals"]),
        eigvecs     = np.array(payload["eigvecs"]),
        f_Hz        = np.array(payload["f_Hz"]),
        U0          = float(payload["U0_J"]),
        h_trans     = float(payload["h_trans"]),
        h_ang       = float(payload["h_ang"]),
        diag_anharm = np.array(payload.get("diag_anharm", [])),
    )

    # Particle / mesh from parameters.py
    sphere_pos = np.asarray(P.SPHERE_POSITION, dtype=float)
    common = dict(
        cube_edge=P.CUBE_EDGE, n_cubes=P.N_CUBES,
        sphere_radius=P.SPHERE_RADIUS, sphere_position=sphere_pos,
        magnet_density=P.DENSITY_NDFEB,
        sphere_density=P.DENSITY_BOROSILICATE,
        B_remanence=P.B_REMANENCE)
    if P.RESCALE_TO_MASS is not None:
        particle = CompositeParticle.from_total_mass(P.RESCALE_TO_MASS, **common)
    else:
        particle = CompositeParticle(**common)

    mesh_path = out_dir / "_meshes" / "cavity.msh"
    if not mesh_path.exists():
        mesh_path.parent.mkdir(exist_ok=True)
        elliptical_cavity(P.CAVITY_A, P.CAVITY_B, P.CAVITY_HEIGHT,
                           mesh_size_sc=P.MESH_SIZE,
                           output_file=str(mesh_path))
    sct = load_msh(mesh_path)

    solver_kwargs = dict(element=P.ELEMENT,
                         subtract_singularity=P.SUBTRACT_SINGULARITY)

    # Pick a tighter slice span that avoids the elliptical cavity wall.
    # min(A, B) / sqrt(2) is the largest square that fits inside the ellipse.
    safe_span = 0.85 * min(P.CAVITY_A, P.CAVITY_B) / np.sqrt(2.0)
    print(f"slice span = {safe_span*1e3:.3f} mm")

    print("standard report ...")
    write_report(sct, nm, particle.moment, out_dir,
                 mass=particle.mass,
                 plot_slice_span=float(safe_span),
                 **solver_kwargs)

    print("frequency bar ...")
    plot_frequency_bar(nm, out_dir / "frequencies.png",
                       compare_f_Hz=P.COMPARE_F_HZ,
                       compare_label=P.COMPARE_LABEL)
    print("hessian card ...")
    plot_equilibrium_residual(nm, out_dir / "hessian_card.png")
    print("B-induced slice (1 FEM solve) ...")
    plot_B_induced_slice(sct, eq_r, P.EQ_THETA, P.EQ_PHI, particle.moment,
                         out_dir / "B_induced_xz.png",
                         span=float(safe_span), n_pts=25, **solver_kwargs)
    print("long-range U scans (slow) ...")
    long_axes = plot_U_long_axes(
        sct, eq_r, P.EQ_THETA, P.EQ_PHI, particle.moment, particle.mass,
        span=float(safe_span), n_pts=11,
        out_path=out_dir / "U_long_axes.png",
        **solver_kwargs)
    (out_dir / "U_long_axes.json").write_text(json.dumps(long_axes, indent=2))

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
