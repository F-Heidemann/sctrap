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

## Fix 4 — gradient-based equilibrium from a single solve per iterate

(Fixes 2 and 3 deferred at user request; done out of order.)

**Physics.** U(r0) = -1/2 m.B_ind(r0; r0) depends on r0 through the
evaluation point AND the source position in the Neumann BC. By reciprocity
of the induced Green's function the two derivative contributions are equal,
so the factor 2 cancels the 1/2:
F = +grad_r[m.B_ind(r; r0)]|_{r=r0} with the source FROZEN (no 1/2 — the
classical force-against-the-frozen-image result), and
dU/dq = -(dm/dq).B_ind(r0) for q in {theta, phi}. Both come from the ONE
solve already needed for U: torques reuse B_ind(r0); the force needs
m_j dB_j/dx_i = analytic image-gradient (closed form, `grad_B_dipole`) +
a short central difference of the smooth residual over the exact
element-local gradients (6 interpolations, zero extra solves).

**Validation gate (before trusting the force).** Against the FD gradient of
the full energy (source moves too) on the 0.3 mm Vinante cavity:
force 0.02-0.10 %, torque 0.001-0.6 % of the gradient norm, wherever the FD
itself is step-converged. (At mid-cavity points the FD "truth" swings ~2.5x
with its own step — facet-jump noise on a force 300x below gravity — while
the analytic value stays put; irrelevant for optimisation since gravity
dominates the total gradient there.) Gate: PASSED.

**Two failure modes found and fixed during integration:**
1. *Energy-based stopping is wrong for this problem.* The landscape is so
   flat that 7 % of the weight of unbalanced force changes U by ~1e-4
   relative per step — scipy's ftol quit there (Fuchs stopped ~15 um early).
   Replaced by force balance: projected gradient <= 1e-3 x weight.
2. *The nearest-facet image selection is degenerate on the symmetry axis*:
   at nm-scale x-y perturbations the image plane flips between facets,
   carving an artificial 2.3e-12 J groove in U whose bottom lies BELOW the
   smooth-branch minimum — any faithful minimiser (L-BFGS-B included)
   correctly converges into the artifact; baseline Nelder-Mead escaped it by
   luck of its wandering. Scoped remedy: freeze the image plane during the
   search (`solve_phi(image_plane=...)`; the continuum split is exact for
   any plane), re-freezing if the dipole migrates to another facet. The
   full continuous-image construction remains Fix 3.

**A/B (after Fix 4 vs baseline):**

| Oracle | Baseline (NM) | After Fix 4 (L-BFGS-B) |
|---|---|---|
| fast tests | 22 passed | 28 passed (6 new) |
| validate 1.5 mm plain / subtract | 135.02 % / 6.17 % | 135.02 % / 6.16 % (unchanged) |
| Vinante all 5 modes | -3.13/1.25/1.40/58.8511/394.54 | -3.44/1.23/1.40/**58.8511**/394.54 (z, theta identical; sub-Hz x/y/phi shifts are the flat-basin floor) |
| Vinante equilibrium stage | 3.6 s | 1.8 s (2x) |
| solves to equilibrium (coarse cavity, bad seed) | 101 | 15-16 (~6.5x fewer) |
| Fuchs equilibrium | z=1.9937 mm, 0.07 x weight UNBALANCED (noise-floor stop) | z=1.9918 mm, true force balance (<1e-3 x weight) |
| Fuchs z-mode | 24.92 Hz (-6.7 % vs published 26.7) | 30.33 Hz (+13.6 %) — see caveat |
| Fuchs equilibrium stage | 36.0 s (~100 solves) | 46.1 s (32 solves x ~2-3 re-freeze rounds; per-solve cost unchanged: U+grad costs the same 0.23 s as U alone) |

**Honest caveat (the Fuchs frequencies are noise, and now we know why).**
Across three correctly-converged equilibria within a 15 um basin the Fuchs
translational modes scatter (24.9-34.1 Hz for z; x/y similar) while the
theta/phi modes are reproducible to 1e-3 Hz. Diagnosed, in order:
(a) NOT the optimiser — the equilibria agree on force balance;
(b) NOT primarily the image-plane flips — freezing the plane changes the
    Hessian by <0.3 Hz;
(c) the 20 um Hessian stencil carries a curvature signal of only ~8.6e-5
    relative in U — barely 2x the 4e-5 FEM energy noise floor — and the
    H_xz cross term mixes 26-30 % of x into the "z" mode, moving its
    eigenvalue by +-20 % depending on where in the basin the Hessian is
    taken. The diagonal H_zz itself is stable (31.9/30.7/29.4 Hz at
    h = 20/50/100 um — the drift is real quartic anharmonicity).
The baseline's -6.7 % agreement with the published 26.7 Hz was one draw
from this +-20 % distribution. Conclusion: Fix 4's "same frequencies"
acceptance is not evaluable on Fuchs until the noise floor drops — that is
exactly Fix 2 (mesh convergence) + Fix 3 (continuous image construction),
which should be done next, then Fuchs re-baselined.

**New tests** (`tests/test_force_gradient.py`): grad_B_dipole vs FD +
symmetry/tracelessness, moment derivatives vs FD, the <1 % gate vs the FD
energy gradient on a real cavity, and NM-vs-gradient equilibrium agreement
with a >3x solve-count reduction. All pass.
