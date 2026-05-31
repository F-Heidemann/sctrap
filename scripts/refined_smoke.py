"""Single-point smoke test: does the refined mesh now solve without NaN?

Runs one solve at z=0 on a refined two-plate mesh after the
`_add_sc_distance_field` fix (no longer disables MeshSizeFromPoints).
Compares to analytic.
"""
from __future__ import annotations

import time
import numpy as np
from pathlib import Path

from sctrap import MAGNET_MOMENT
from sctrap.generators import two_parallel_plates
from sctrap.mesh import load_msh
from sctrap.potential import U_mag
from sctrap.validation import U_analytic_two_plates

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

PLATE_SEP = 4e-3
out = RESULTS / "_meshes" / "smoke_refined.msh"
out.parent.mkdir(exist_ok=True)

print("Generating refined mesh...", flush=True)
two_parallel_plates(
    plate_half_side=15e-3, plate_thickness=0.5e-3,
    plate_separation=PLATE_SEP, far_radius=60e-3,
    mesh_size_sc=2.0e-3, mesh_size_ff=20e-3,
    mesh_size_near=1.0e-3, refine_distance=1.0e-3,   # gentle refinement
    output_file=str(out), verbose=False,
)
sct = load_msh(out)
print(f"  {sct.mesh.p.shape[1]} verts, {sct.mesh.t.shape[1]} tets", flush=True)

m  = np.array([MAGNET_MOMENT, 0.0, 0.0])
r0 = np.zeros(3)

t0 = time.time()
U_fem_plain = U_mag(sct, m, r0, element="P2", subtract_singularity=False)
print(f"refined plain:    {time.time()-t0:.1f}s   U={U_fem_plain:.4e}", flush=True)

t0 = time.time()
U_fem_sub = U_mag(sct, m, r0, element="P2", subtract_singularity=True)
print(f"refined+sub:      {time.time()-t0:.1f}s   U={U_fem_sub:.4e}", flush=True)

U_ana = U_analytic_two_plates(0.0, m, PLATE_SEP, n_images=80)
print(f"analytic:                       U={U_ana:.4e}")
print()
print(f"refined plain  rel_err = {(U_fem_plain - U_ana)/abs(U_ana)*100:+.3f} %")
print(f"refined+sub    rel_err = {(U_fem_sub  - U_ana)/abs(U_ana)*100:+.3f} %")
