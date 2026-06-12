# Fix log: A/B tables against the Step-0 baseline (PROFILE_BASELINE.md)

## Fix 1 — element-local grad(Phi) in `B_induced_at` (exact P2 gradient)

**Change.** `PhiSolution.grad` locates the tetrahedron containing the point
once, pulls back to reference coordinates, and contracts the analytic P2
basis gradients with the element DOF values (mirrors `CellBasis.probes`,
which does the same for values). `B_induced_at` now uses it by default
(`method="exact"`); the legacy 6-point FD survives behind `method="fd"`.
The image-field add-back for singularity-subtracted solves is unchanged.

**Why this is an accuracy fix, not a speed fix.** Baseline profile showed
`B_induced_at` at ~2 % of runtime. Within one tetrahedron the P2 interpolant
is exactly quadratic, so the old centred FD was *already exact* whenever its
6-point stencil stayed inside one element — but near element faces / mesh
nodes (e.g. an equilibrium pinned on a symmetry axis) the stencil straddles
a gradient discontinuity and the answer depended on `h`. The exact method
removes that failure mode entirely; `h` is now ignored by default.

**A/B (after Fix 1 vs baseline):**

| Oracle | Baseline | After Fix 1 |
|---|---|---|
| fast tests | 22 passed / 1.4 s | 25 passed / 1.5 s (3 new fast tests) |
| validate 1.5 mm, plain | 135.02 % FAIL(>10 %) | 135.02 % (identical) |
| validate 1.5 mm, subtract | 6.17 % PASS | 6.16 % PASS |
| Vinante z-mode | 58.8511 Hz (+4.16 %) | 58.8511 Hz (+4.16 %) — all 5 modes identical to 1e-4 Hz |
| Fuchs z-mode | 24.9154 Hz (−6.68 %) | 24.9154 Hz (−6.68 %) — all 5 modes identical to 1e-4 Hz |
| Vinante wall | 19.4 s | 20.2 s (run-to-run noise) |
| Fuchs wall | 80.4 s | 82.4 s (run-to-run noise) |
| `B_induced_at` per call (0.3 mm Vinante mesh) | 0.58 ms (fd) | 0.45 ms (exact), same B to all printed digits |

**New tests** (`tests/test_b_induced.py`): exact==FD to <1e-6 at element
barycentres (synthetic + real two-plate solve), exact gradient of a
projected known quadratic to 1e-9, h-independence, and exact B vs the
analytic two-plate image series (<5 %). All pass.
