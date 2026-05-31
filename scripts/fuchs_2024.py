"""Fuchs et al. 2024 benchmark — measured z-mode 26.7 Hz, analytic estimate 27 Hz.

Reference
---------
T. M. Fuchs et al., "Measuring gravity with milligram levitated masses",
Sci. Adv. 10, eadk2949 (2024)  /  arXiv:2303.03545.

Geometry (closed trap, type-I superconductor, tantalum)
    elliptical cavity 4.5 mm x 3.5 mm, height 4.7 mm.

Particle
    Three 0.25 mm Nd2Fe14B cubes end-to-end + 0.25 mm glass bead.
    B_rem = 1.4 T, total mass 0.43 mg.

We model the particle as a point dipole.

Quick-start usage:
    python3 scripts/fuchs_2024.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sctrap.config import MU_0, G_GRAV
from sctrap.dipole import dipole_moment
from sctrap.generators import elliptical_cavity
from sctrap.mesh import load_msh
from sctrap.frequencies import find_equilibrium
from sctrap.modes import normal_modes
from sctrap.particle import CompositeParticle


# --- Paper parameters ------------------------------------------------------
A, B           = 4.5e-3 / 2.0,  3.5e-3 / 2.0      # semi-axes [m]
HEIGHT         = 4.7e-3                            # cavity height [m]

# Three 0.25 mm NdFeB cubes glued along the easy axis, glass bead glued
# under the middle cube. Densities are rescaled so the total mass matches
# the paper's published 0.43 mg.
PARTICLE = CompositeParticle.from_total_mass(0.43e-6)
M_MOMENT         = PARTICLE.moment
PARTICLE_MASS    = PARTICLE.mass
# Anisotropic transverse libration: I_yy != I_zz once the bead breaks symmetry.
PARTICLE_INERTIA = (PARTICLE.inertia_yy, PARTICLE.inertia_zz)


HERE    = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
MESH    = RESULTS / "_meshes" / "fuchs_cavity.msh"
MESH.parent.mkdir(exist_ok=True)


def main() -> None:
    print("Fuchs et al. 2024 trap benchmark")
    print("=" * 50)
    print(f"  cavity            : {2*A*1e3:.2f} x {2*B*1e3:.2f} mm, "
          f"height {HEIGHT*1e3:.2f} mm  (closed)")
    print(f"  magnet moment     : {M_MOMENT:.3e} A m^2")
    print(f"  particle mass     : {PARTICLE_MASS*1e6:.3f} mg")
    print(f"  particle inertia  : I_yy={PARTICLE.inertia_yy:.3e}, "
          f"I_zz={PARTICLE.inertia_zz:.3e} kg m^2")
    print(f"  COM offset (z)    : {PARTICLE.com[2]*1e6:+.2f} um (bead below bar)")
    print(f"  paper z-mode      : 26.7 Hz (measured), 27 Hz (analytic)")

    print("\n[1/4] Generating closed elliptical cavity mesh...")
    elliptical_cavity(A, B, HEIGHT,
                      mesh_size_sc=0.5e-3,
                      output_file=str(MESH), verbose=False)
    sct = load_msh(MESH)
    print(f"      {sct}")

    # Magnet horizontal: along x (matches the paper's bar lying in the plane).
    m = dipole_moment(M_MOMENT, theta=np.pi / 2.0, phi=0.0)
    print(f"\n  dipole moment vector m = {m}  (along x)")

    # Initial guess: centre of cavity floor area, lifted to the centre.
    # Equilibrium height is set by gravity vs. Meissner repulsion from floor.
    r0_guess = np.array([0.0, 0.0, HEIGHT * 0.5])
    print(f"\n[2/4] Finding equilibrium (gravity on)...")
    t0 = time.time()
    eq = find_equilibrium(
        sct, m, r0_guess,
        mass=PARTICLE_MASS,
        element="P2", subtract_singularity=True,
    )
    print(f"      r_eq = ({eq[0]*1e3:+.4f}, {eq[1]*1e3:+.4f}, {eq[2]*1e3:+.4f}) mm  "
          f"({time.time()-t0:.1f} s)")

    print(f"\n[3/4] Building 5x5 Hessian and diagonalising...")
    t0 = time.time()
    nm = normal_modes(
        sct, eq, M_MOMENT,
        eq_theta=np.pi / 2.0, eq_phi=0.0,
        mass=PARTICLE_MASS, inertia=PARTICLE_INERTIA,
        h_trans=20e-6, h_ang=1e-3,
        element="P2", subtract_singularity=True,
    )
    print(f"      ({time.time()-t0:.1f} s)")
    print()
    print(nm)

    # Pick the z-dominated mode and compare to the paper.
    z_idx = int(np.argmax(np.abs(nm.eigvecs[2, :])))
    f_z   = float(nm.f_Hz[z_idx])
    print()
    print(f"  z-dominant mode    : {f_z:.3f} Hz   (paper: 26.7 Hz measured, 27 Hz est.)")
    diff_pct = (f_z - 26.7) / 26.7 * 100.0
    print(f"  agreement vs 26.7  : {diff_pct:+.2f} %")

    out = {
        "geometry": {
            "axes_mm":   [2*A*1e3, 2*B*1e3],
            "height_mm": HEIGHT*1e3,
            "type":      "closed elliptical cavity",
        },
        "particle": {
            "moment_Am2":  M_MOMENT,
            "mass_kg":     PARTICLE_MASS,
            "inertia_kgm2": PARTICLE_INERTIA,
        },
        "result": {
            "eq_r_mm":          (eq * 1e3).tolist(),
            "f_Hz":             nm.f_Hz.tolist(),
            "z_dominant_idx":   z_idx,
            "f_z_Hz":           f_z,
            "paper_f_z_Hz":     26.7,
            "paper_estimate_Hz": 27.0,
            "agreement_pct":    diff_pct,
        },
    }
    out_path = RESULTS / "fuchs_2024.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
