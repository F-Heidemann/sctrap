"""Off-midplane two-plate accuracy sweep.

Runs the FEM at seven z-positions between the two SC plates under four
configurations and compares to the analytic image-dipole series:

    1. baseline      : uniform 1.5 mm mesh, no singularity subtraction
    2. refined       : Distance+Threshold size field near SC
    3. subtraction   : uniform mesh, singularity subtraction enabled
    4. refined+sub   : both

This is what closes the loop on the NEXT-1 / NEXT-2 acceptance criteria
in REDESIGN.md (max rel err < 5 % refined, < 1 % refined+subtraction).

Output: JSON + text summary in sctrap/results/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sctrap import MAGNET_MOMENT
from sctrap.generators import two_parallel_plates
from sctrap.mesh import load_msh
from sctrap.potential import U_mag
from sctrap.validation import U_analytic_two_plates


HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


PLATE_SEP   = 4e-3
HALF_SIDE   = 15e-3
THICKNESS   = 0.5e-3
FAR_RADIUS  = 60e-3
SIZE_SC     = 2.0e-3
SIZE_FF     = 20e-3
SIZE_NEAR   = 0.8e-3
REFINE_DIST = 1.2e-3
N_IMAGES    = 80

Z_POINTS = np.linspace(-0.8, 0.8, 5) * (PLATE_SEP / 2.0)   # ±0.8 of half-gap


def make_mesh(refined: bool, out: Path) -> Path:
    kw = dict(
        plate_half_side=HALF_SIDE, plate_thickness=THICKNESS,
        plate_separation=PLATE_SEP, far_radius=FAR_RADIUS,
        mesh_size_sc=SIZE_SC, mesh_size_ff=SIZE_FF,
        output_file=str(out), verbose=False,
    )
    if refined:
        kw["mesh_size_near"]   = SIZE_NEAR
        kw["refine_distance"] = REFINE_DIST
    two_parallel_plates(**kw)
    return out


def sweep(mesh_path: Path, subtract: bool) -> dict:
    sct = load_msh(mesh_path)
    m = np.array([MAGNET_MOMENT, 0.0, 0.0])

    t0 = time.time()
    U_fem = []
    for z in Z_POINTS:
        r0 = np.array([0.0, 0.0, float(z)])
        U_fem.append(U_mag(sct, m, r0,
                           element="P2",
                           subtract_singularity=subtract))
    elapsed = time.time() - t0

    U_ana = [U_analytic_two_plates(float(z), m, PLATE_SEP, n_images=N_IMAGES)
             for z in Z_POINTS]
    U_fem = np.array(U_fem)
    U_ana = np.array(U_ana)
    rel = (U_fem - U_ana) / np.where(np.abs(U_ana) > 0, np.abs(U_ana), 1.0)

    return {
        "z_m":        Z_POINTS.tolist(),
        "U_fem_J":    U_fem.tolist(),
        "U_ana_J":    U_ana.tolist(),
        "rel_err":    rel.tolist(),
        "max_abs_rel_err_pct": float(100.0 * np.max(np.abs(rel))),
        "elapsed_s":  elapsed,
        "n_vertices": int(sct.mesh.p.shape[1]),
        "n_tets":     int(sct.mesh.t.shape[1]),
    }


def main() -> None:
    out_root = RESULTS
    mesh_dir = out_root / "_meshes"
    mesh_dir.mkdir(exist_ok=True)

    print("[1/2] generating meshes...", flush=True)
    base_msh = make_mesh(False, mesh_dir / "two_plates_uniform.msh")
    ref_msh  = make_mesh(True,  mesh_dir / "two_plates_refined.msh")

    configs = [
        ("baseline",        base_msh, False),
        ("refined",         ref_msh,  False),
        ("subtraction",     base_msh, True),
        ("refined+sub",     ref_msh,  True),
    ]

    print("[2/2] running sweeps...", flush=True)
    results = {}
    for name, mp, sub in configs:
        print(f"  - {name}: mesh={mp.name} subtract={sub}", flush=True)
        results[name] = sweep(mp, sub)
        print(f"      max |rel err| = "
              f"{results[name]['max_abs_rel_err_pct']:.2f} %  "
              f"({results[name]['elapsed_s']:.1f} s, "
              f"{results[name]['n_tets']} tets)", flush=True)

    payload = {
        "geometry": {
            "plate_separation_m": PLATE_SEP,
            "plate_half_side_m":  HALF_SIDE,
            "plate_thickness_m":  THICKNESS,
            "far_radius_m":       FAR_RADIUS,
        },
        "mesh_settings": {
            "size_sc_m":     SIZE_SC,
            "size_ff_m":     SIZE_FF,
            "size_near_m":   SIZE_NEAR,
            "refine_dist_m": REFINE_DIST,
        },
        "z_points_m":  Z_POINTS.tolist(),
        "moment_Am2":  MAGNET_MOMENT,
        "n_images":    N_IMAGES,
        "results":     results,
    }
    json_out = out_root / "two_plate_off_midplane.json"
    json_out.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON -> {json_out}")

    # Text summary
    lines = []
    lines.append("Two-plate off-midplane accuracy sweep")
    lines.append("=" * 62)
    lines.append(f"plate_separation = {PLATE_SEP*1e3:.2f} mm   "
                 f"size_sc = {SIZE_SC*1e3:.2f} mm   "
                 f"size_near = {SIZE_NEAR*1e3:.2f} mm")
    lines.append("")
    header = f"{'z [mm]':>8}  " + "  ".join(f"{n:>14}" for n, *_ in configs)
    lines.append(header)
    lines.append("-" * len(header))
    for i, z in enumerate(Z_POINTS):
        cells = [f"{z*1e3:+.3f}"]
        for name, *_ in configs:
            r = results[name]["rel_err"][i] * 100.0
            cells.append(f"{r:+9.2f} %")
        lines.append(f"{cells[0]:>8}  " + "  ".join(f"{c:>14}" for c in cells[1:]))
    lines.append("-" * len(header))
    summary = ["max |rel err|"]
    for name, *_ in configs:
        summary.append(f"{results[name]['max_abs_rel_err_pct']:9.2f} %")
    lines.append(f"{summary[0]:>8}  " + "  ".join(f"{s:>14}" for s in summary[1:]))
    lines.append("")
    lines.append("Mesh & timing:")
    for name, *_ in configs:
        r = results[name]
        lines.append(f"  {name:>14}: {r['n_vertices']:>7d} verts  "
                     f"{r['n_tets']:>7d} tets   {r['elapsed_s']:6.1f} s")

    txt_out = out_root / "two_plate_off_midplane.txt"
    txt_out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nTXT  -> {txt_out}")


if __name__ == "__main__":
    main()
