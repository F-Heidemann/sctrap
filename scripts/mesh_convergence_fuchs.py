"""Mesh convergence study for the Fuchs 2024 elliptical-cavity bar magnet.

Pipeline
--------
1. Build the volumetric BarMagnetParticle (mass-matched to 0.43 mg).
2. On the coarsest mesh (0.5 mm SC mesh), run a full 5D
   `find_equilibrium_5d` to locate the *true* minimum of U_total over
   (x, y, z, theta, phi).  Verify all 5 Hessian eigenvalues are positive.
3. For each finer mesh size, warm-start from the previous mesh's
   equilibrium config (orientation is expected to be near mesh-invariant),
   re-optimise position, take the full 5D Hessian, and record:
       (mesh_size, n_tets, n_dofs, eq_r, eq_theta, eq_phi,
        f_modes (5), all_stable_bool)
4. Write a CSV/JSON summary and a log-log plot of f_z vs DOF count.

Run:
    cd sctrap
    python3 scripts/mesh_convergence_fuchs.py
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "results" / "mesh_convergence_fuchs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from sctrap.dipole import dipole_moment
from sctrap.frequencies import find_equilibrium, find_equilibrium_5d
from sctrap.generators import elliptical_cavity
from sctrap.mesh import load_msh
from sctrap.modes import normal_modes
from sctrap.particle import BarMagnetParticle

# Fuchs 2024 cavity (4.5 x 3.5 x 4.7 mm, closed)
CAVITY_A = 4.5e-3 / 2.0
CAVITY_B = 3.5e-3 / 2.0
CAVITY_H = 4.7e-3

# Bar particle (volumetric): 3-cube-equivalent geometry, density rescaled to
# match the Fuchs published total mass 0.43 mg.
PARTICLE = BarMagnetParticle(
    length=0.75e-3, width=0.25e-3, height=0.25e-3,
    density=9173.3, B_remanence=1.4,
)
INERTIA = (PARTICLE.inertia_yy, PARTICLE.inertia_zz)
M_MAG   = PARTICLE.moment

# Mesh sizes to sweep (SC surface mesh size, in m). Finest values can be
# very slow at this geometry — adjust as needed.
MESH_SIZES = [0.5e-3, 0.4e-3, 0.3e-3, 0.25e-3]

# Hessian stencil widths for normal_modes (kept fixed across all meshes).
H_TRANS = 20e-6
H_ANG   = 1e-3


def _z_mode(nm):
    z_idx = int(np.argmax(np.abs(nm.eigvecs_mw[2, :])))
    return float(nm.f_Hz[z_idx])


def _all_stable(nm) -> bool:
    return bool(np.all(nm.eigvals > 0))


def build_mesh(mesh_size: float) -> tuple:
    mpath = OUT_DIR / f"cavity_{int(mesh_size*1e6)}um.msh"
    print(f"  building mesh ({mesh_size*1e3:.3f} mm) -> {mpath.name}", flush=True)
    t0 = time.time()
    elliptical_cavity(CAVITY_A, CAVITY_B, CAVITY_H,
                      mesh_size_sc=mesh_size,
                      output_file=str(mpath),
                      verbose=False)
    sct = load_msh(mpath)
    print(f"  mesh built in {time.time()-t0:.1f} s: {sct}", flush=True)
    return sct


def run_at_mesh(sct, *, theta0: float, phi0: float, r0: np.ndarray,
                refine_5d: bool) -> dict:
    """Find equilibrium and compute modes on this mesh.

    `refine_5d=True` does a full 5D Nelder-Mead.  `refine_5d=False` only
    re-optimises position (3D), holding orientation at (theta0, phi0).
    The orientation comes from the previous, coarser mesh.
    """
    t0 = time.time()
    if refine_5d:
        # On the first (coarsest) mesh we cast a wide simplex; on subsequent
        # meshes the bar's preferred (theta, phi) is essentially mesh-
        # invariant so we use a tighter simplex starting from the previous
        # mesh's solution.
        wide = (theta0 == np.pi / 2 and phi0 == 0.0)
        print(f"  5D optimisation ({'wide simplex' if wide else 'warm-start'}) ...",
              flush=True)
        r_eq, theta_eq, phi_eq = find_equilibrium_5d(
            sct, M_MAG, r0,
            theta_guess=theta0, phi_guess=phi0,
            mass=PARTICLE.mass,
            particle=PARTICLE,
            element="P2",
            verbose=False,
            maxiter=300 if wide else 150,
            step_pos=50e-6 if wide else 20e-6,
            step_ang=0.05 if wide else 0.02,
            xatol=1e-7,
        )
    else:
        print("  3D position refinement (orientation locked) ...", flush=True)
        m_vec = dipole_moment(M_MAG, theta0, phi0)
        r_eq = find_equilibrium(
            sct, m_vec, r0,
            mass=PARTICLE.mass,
            particle=PARTICLE,
            element="P2",
            verbose=False,
            maxiter=200,
        )
        theta_eq, phi_eq = theta0, phi0
    dt_eq = time.time() - t0
    print(f"    eq: r=({r_eq[0]*1e3:+.4f},{r_eq[1]*1e3:+.4f},"
          f"{r_eq[2]*1e3:+.4f}) mm  theta={theta_eq:+.4f}  phi={phi_eq:+.4f}  "
          f"({dt_eq:.1f} s)", flush=True)

    t0 = time.time()
    print("  normal_modes ...", flush=True)
    nm = normal_modes(
        sct, r_eq, M_MAG,
        eq_theta=theta_eq, eq_phi=phi_eq,
        mass=PARTICLE.mass, inertia=INERTIA,
        h_trans=H_TRANS, h_ang=H_ANG,
        particle=PARTICLE,
        element="P2",
    )
    dt_nm = time.time() - t0
    f_z = _z_mode(nm)
    stable = _all_stable(nm)
    print(f"    modes: f_z = {f_z:7.3f} Hz   all_stable = {stable}   ({dt_nm:.1f} s)",
          flush=True)
    print(f"    f_Hz = {np.array2string(nm.f_Hz, precision=3)}", flush=True)
    return {
        "n_verts":  int(sct.mesh.p.shape[1]),
        "n_tets":   int(sct.mesh.t.shape[1]),
        "n_sc_facets": int(len(sct.sc_facets)),
        "eq_r_m":   r_eq.tolist(),
        "eq_theta_rad": float(theta_eq),
        "eq_phi_rad":   float(phi_eq),
        "f_Hz":     nm.f_Hz.tolist(),
        "eigvals":  nm.eigvals.tolist(),
        "f_z_Hz":   float(f_z),
        "all_stable": bool(stable),
        "dt_eq_s":  float(dt_eq),
        "dt_modes_s": float(dt_nm),
    }


def main() -> None:
    print("===== Fuchs 2024 mesh convergence study (volumetric bar) =====\n",
          flush=True)
    print("Particle:", flush=True)
    print(PARTICLE, flush=True)
    print(flush=True)

    results = []
    # Track orientation/position from previous mesh as warm start for next.
    prev_theta = np.pi / 2.0
    prev_phi   = 0.0
    prev_r     = np.array([0.0, 0.0, CAVITY_H * 0.5])

    for k, h in enumerate(MESH_SIZES):
        print(f"--- Mesh #{k+1}/{len(MESH_SIZES)}: SC mesh size = {h*1e3:.3f} mm ---",
              flush=True)
        sct = build_mesh(h)
        # Always re-run 5D at each mesh; warm-start from the previous mesh's
        # solution so finer meshes converge in 50-100 evaluations rather than
        # the 500 needed when starting from (theta=pi/2, phi=0).
        rec = run_at_mesh(
            sct,
            theta0=prev_theta,
            phi0=prev_phi,
            r0=prev_r,
            refine_5d=True,
        )
        rec["mesh_size_m"] = float(h)
        results.append(rec)

        # Warm start for next mesh.
        prev_theta = rec["eq_theta_rad"]
        prev_phi   = rec["eq_phi_rad"]
        prev_r     = np.asarray(rec["eq_r_m"])

        # Persist partial results after every mesh.
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump({"particle": PARTICLE.as_dict(),
                       "cavity_a_m": CAVITY_A,
                       "cavity_b_m": CAVITY_B,
                       "cavity_h_m": CAVITY_H,
                       "results": results}, f, indent=2)
        print(f"  -> saved partial results to {OUT_DIR/'results.json'}\n",
              flush=True)

    # ------ Summary ------
    print("===== Convergence summary =====", flush=True)
    print(f"{'mesh [mm]':>10} {'n_tets':>8} {'n_dofs':>8} "
          f"{'f_z [Hz]':>10} {'all_stable':>10}", flush=True)
    for r in results:
        # Rough DOF count for P2 tets: 4 verts + 6 edges per tet (shared),
        # but easier: n_verts + n_edges ≈ n_verts * 7 for our meshes.
        ndof = r["n_verts"] + 0  # placeholder
        print(f"{r['mesh_size_m']*1e3:>10.3f} "
              f"{r['n_tets']:>8d} {r['n_verts']:>8d} "
              f"{r['f_z_Hz']:>10.3f} {str(r['all_stable']):>10}",
              flush=True)

    # ------ Plot ------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        nverts = np.array([r["n_verts"] for r in results])
        fz     = np.array([r["f_z_Hz"]  for r in results])
        ax.semilogx(nverts, fz, "o-", lw=1.5, ms=7)
        ax.axhline(26.7, color="red", ls="--", lw=1, label="Fuchs 2024 (measured)")
        ax.set_xlabel("mesh vertex count (P2 DOFs ~ this)")
        ax.set_ylabel("z-mode frequency [Hz]")
        ax.set_title("Fuchs bar — mesh convergence (5D-minimised, volumetric)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "mesh_convergence.png", dpi=120)
        print(f"\nPlot saved to {OUT_DIR/'mesh_convergence.png'}", flush=True)
    except ImportError:
        print("\n(matplotlib not available; skipping plot)", flush=True)


if __name__ == "__main__":
    main()
