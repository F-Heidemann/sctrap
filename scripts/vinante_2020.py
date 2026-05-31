"""Vinante et al. (Phys. Rev. Applied 13, 064027 (2020); arXiv:1912.12252)
benchmark — micromagnet levitated in a closed cylindrical Pb cavity.

Reference numbers from the paper
--------------------------------
    Trap        : cylindrical well, 4 mm diameter, 4 mm depth, 99.95% Pb.
                  We model it as a *closed* cavity (top is the wider Pb body
                  visible in Fig. 1a, also a SC surface; the asymmetry
                  between floor and ceiling is what gives translational
                  confinement at z0 ≈ 311 um above the floor).
    Particle    : NdFeB-based microsphere, R = (27 ± 1) um.
    Density     : rho = 7430 kg/m^3.
    Magnetisation: mu_0 M = 0.71 T  -> M = 0.71 / mu_0  A/m
                   moment mu = M * V = 4.66e-8 A m^2

Predicted by image-method (Eqs. 2,5,6):
    z0 = (3 mu_0 mu^2 / (64 pi m g))^{1/4}    ≈ 287 um (paper says 311 um;
                                              difference is the closed-trap
                                              vs single-plane geometry).
    omega_z  = sqrt( 4 g / z0 )               -> f_z  ≈ 56.5 Hz
    omega_β = sqrt( 5 z0 g / (3 R^2) )        -> f_β  ≈ 377 Hz

Other measured modes (paper Fig. 2):
    x ≈ 4 Hz, y ≈ 10 Hz                       (tilt-sensitive)
    α ≈ 150 Hz                                (FEM predicts ~100 Hz)

Run with:
    python3 scripts/vinante_2020.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sctrap.config import G_GRAV, MU_0
from sctrap.dipole import dipole_moment
from sctrap.frequencies import find_equilibrium
from sctrap.generators import elliptical_cavity
from sctrap.mesh import load_msh
from sctrap.modes import normal_modes
from sctrap.particle import SphericalParticle
from sctrap.report import write_report
from sctrap.sanity_plots import (
    plot_B_induced_slice, plot_equilibrium_residual,
    plot_frequency_bar, plot_mesh_overview,
    plot_U_long_axes,
)


# --- paper geometry --------------------------------------------------------
TRAP_DIAMETER = 4.0e-3                # 4 mm
TRAP_HEIGHT   = 4.0e-3                # 4 mm
TRAP_RADIUS   = TRAP_DIAMETER / 2.0

# --- particle --------------------------------------------------------------
PARTICLE = SphericalParticle(
    radius      = 27.0e-6,
    density     = 7430.0,
    B_remanence = 0.71,                # mu_0 M  [T]
)

# --- run config ------------------------------------------------------------
ELEMENT              = "P2"
SUBTRACT_SINGULARITY = True
# Mesh resolution: bulk size ~150 um, with refinement near the SC walls
# down to 30 um, so the dipole-singularity subtraction sees a fine surface
# at the floor (where the magnet sits ~300 um above).
MESH_SIZE          = 0.30e-3        # uniform tet size [m]
MESH_SIZE_NEAR     = None           # rely on singularity subtraction at floor
MESH_REFINE_DIST   = None

H_TRANS = 5.0e-6                      # 5 um position step (R/5)
H_ANG   = 5.0e-3                      # 5 mrad angle step

EQ_THETA = np.pi / 2.0                # bar (= dipole moment) horizontal
EQ_PHI   = 0.0

HERE   = Path(__file__).resolve().parent.parent
OUTDIR = HERE / "results" / "vinante_2020"
OUTDIR.mkdir(parents=True, exist_ok=True)
MESHF  = OUTDIR / "_meshes" / "cavity.msh"
MESHF.parent.mkdir(exist_ok=True)


def image_method_estimate(p: SphericalParticle) -> dict:
    """Eqs. 2, 5, 6 of the paper."""
    z0 = (3.0 * MU_0 * p.moment ** 2
          / (64.0 * np.pi * p.mass * G_GRAV)) ** 0.25
    f_z  = np.sqrt(4.0 * G_GRAV / z0) / (2.0 * np.pi)
    f_b  = np.sqrt(5.0 * z0 * G_GRAV / (3.0 * p.radius ** 2)) / (2.0 * np.pi)
    return {"z0_m": float(z0), "f_z_Hz": float(f_z), "f_beta_Hz": float(f_b)}


def main() -> None:
    print("=" * 70)
    print("Vinante 2020 benchmark — closed cylindrical Pb cavity")
    print("=" * 70)
    print(f"  trap            : {TRAP_DIAMETER*1e3:.2f} mm dia "
          f"x {TRAP_HEIGHT*1e3:.2f} mm deep, closed")
    print(PARTICLE)

    # --- analytic estimate (open infinite plane)
    img = image_method_estimate(PARTICLE)
    print()
    print(f"  image-method estimates (single SC plane):")
    print(f"    z0       = {img['z0_m']*1e6:7.2f} um   (paper: 311 um)")
    print(f"    f_z      = {img['f_z_Hz']:7.2f} Hz   (paper: 56.5 Hz)")
    print(f"    f_beta   = {img['f_beta_Hz']:7.2f} Hz   (paper: 377 Hz)")

    # ------------------------------------------------------------------ mesh
    print("\n[1/4]  Generating mesh ...")
    t0 = time.time()
    elliptical_cavity(
        TRAP_RADIUS, TRAP_RADIUS, TRAP_HEIGHT,
        mesh_size_sc    = MESH_SIZE,
        mesh_size_near  = MESH_SIZE_NEAR,
        refine_distance = MESH_REFINE_DIST,
        output_file     = str(MESHF), verbose = False,
    )
    sct = load_msh(MESHF)
    print(f"      {sct}   ({time.time()-t0:.1f} s)")
    plot_mesh_overview(sct, OUTDIR / "mesh.png")

    solver_kwargs = dict(element=ELEMENT,
                         subtract_singularity=SUBTRACT_SINGULARITY)

    # ------------------------------------------------------------- equilibrium
    print("\n[2/4]  Equilibrium (gravity vs Meissner repulsion) ...")
    r_guess = np.array([0.0, 0.0, img["z0_m"]])
    m_vec = dipole_moment(PARTICLE.moment, EQ_THETA, EQ_PHI)
    t0 = time.time()
    eq_r = find_equilibrium(
        sct, m_vec, r_guess, mass = PARTICLE.mass, **solver_kwargs)
    print(f"      r_eq = ({eq_r[0]*1e6:+.1f}, {eq_r[1]*1e6:+.1f}, "
          f"{eq_r[2]*1e6:+.1f}) um   ({time.time()-t0:.1f} s)")
    print(f"      paper z0_image = {img['z0_m']*1e6:.1f} um")

    plot_mesh_overview(sct, OUTDIR / "mesh.png", eq_r=eq_r)

    # ------------------------------------------------------------------ modes
    print("\n[3/4]  Normal modes ...")
    t0 = time.time()
    nm = normal_modes(
        sct, eq_r, PARTICLE.moment,
        eq_theta = EQ_THETA, eq_phi = EQ_PHI,
        mass     = PARTICLE.mass,
        inertia  = PARTICLE.inertia,                # isotropic (sphere)
        h_trans  = H_TRANS, h_ang = H_ANG,
        **solver_kwargs,
    )
    print(f"      ({time.time()-t0:.1f} s)")
    print()
    print(nm)

    # ------------------------------------------------------------------ report
    print("\n[4/4]  Lightweight report ...")
    safe_span = 0.85 * min(TRAP_RADIUS, TRAP_HEIGHT) / np.sqrt(2.0)
    # Skip the heavy 25x25 slice plots — write_text_artifacts + frequency bar
    # + DOF scans are enough for the benchmark comparison.
    from sctrap.report import write_text_artifacts, plot_U_along_dofs
    write_text_artifacts(nm, OUTDIR)
    plot_U_along_dofs(sct, nm, PARTICLE.moment, OUTDIR / "U_along_DOFs.png",
                      n_pts=13, span_factor=4.0,
                      mass=PARTICLE.mass, **solver_kwargs)
    plot_frequency_bar(nm, OUTDIR / "frequencies.png",
                       compare_f_Hz=56.5, compare_label="Vinante 2020 (z)")
    plot_equilibrium_residual(nm, OUTDIR / "hessian_card.png")
    long_axes = {}                        # heavy plot skipped to save time

    # ----- summary --------------------------------------------------------
    z_idx = int(np.argmax(np.abs(nm.eigvecs[2, :])))
    f_z_sim = float(nm.f_Hz[z_idx])
    summary = {
        "geometry": {
            "diameter_mm": TRAP_DIAMETER * 1e3,
            "height_mm":   TRAP_HEIGHT * 1e3,
            "kind":        "closed cylindrical SC cavity",
        },
        "particle":     PARTICLE.as_dict(),
        "image_method": img,
        "paper":        {"f_z_Hz": 56.5, "f_beta_Hz": 377.0,
                         "z0_um": 311.0, "f_x_Hz": 4.0, "f_y_Hz": 10.0,
                         "f_alpha_Hz": 150.0},
        "fem":          {"eq_r_um": (eq_r * 1e6).tolist(),
                         "f_Hz": nm.f_Hz.tolist(),
                         "z_dominant_idx": z_idx,
                         "f_z_Hz": f_z_sim},
        "long_axes_scan": long_axes,
    }
    (OUTDIR / "vinante_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  z-dominant FEM mode : {f_z_sim:8.2f} Hz")
    print(f"  paper z-mode        : {56.5:8.2f} Hz   "
          f"({(f_z_sim-56.5)/56.5*100:+.1f} %)")
    print(f"  artifacts           : {OUTDIR}")


if __name__ == "__main__":
    main()
