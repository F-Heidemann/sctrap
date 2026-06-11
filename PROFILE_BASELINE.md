# Baseline: accuracy + timing + profile (pre-optimization oracle)

Recorded 2026-06-11 on Darwin 25.4.0, Python 3.12.2 (miniconda3), repo @ main.
Every subsequent fix is A/B-compared against these numbers.

## 1. Test suite

| Command | Result | Wall time |
|---|---|---|
| `pytest tests/ -m "not slow"` | **22 passed**, 7 deselected | 1.4 s (1.9 s total) |
| `pytest tests/` | **28 passed, 1 FAILED** | 110.5 s (111 s total) |

Pre-existing failure (present before any change; not introduced by this work):

```
FAILED tests/test_magnet_field.py::test_fuchs_bar_volumetric
  Fuchs bar z-mode out of range: f_z = 35.844 Hz vs 26.7 Hz (rel err = 34.25 %)
  assert 0.3424687239615702 < 0.2
```

## 2. Analytic two-plate validation (`sctrap validate`)

**Default invocation (`sctrap validate`, mesh-size-sc = 0.6 mm) CRASHES at
baseline** after ~38 min wall: the gmsh-generated mesh at that resolution
contains exactly one orphan vertex (a stray point at (14.10, -9.97, 0.10) mm
referenced by no tetrahedron and no facet), producing a zero row in the
stiffness matrix -> `RuntimeError: Factor is exactly singular` in `splu`.
Coarser meshes (1.5 mm, 1.0 mm) have no orphan vertices and work. Pre-existing
defect; remedy (prune unreferenced vertices in `load_msh`) deferred — Step 0
is measurement only.

Working oracle at 1.5 mm SC mesh / 12 mm FF mesh (matches the slow test's
resolution), 11 z-points, ~19 s wall each:

| Invocation | Max rel. error | Verdict | Wall |
|---|---|---|---|
| `sctrap validate --size-sc 1.5e-3 --size-ff 12e-3` | **135.02 %** | FAIL (tol 10 %) | 19.1 s |
| `... --subtract-singularity` | **6.17 %** | PASS | 19.5 s |

With subtraction, 9 of 11 points are < 1 %; the max (6.17 %) occurs at
z = +0.36 mm while the mirror point z = -0.36 mm has only 0.56 % — the
discrete nearest-facet image jump (Fix 3's target) breaks the up/down
symmetry. (The "51 % -> 2 %" figure quoted for this benchmark corresponds to
the finer default mesh, which currently cannot run due to the orphan-vertex
crash.)

## 3. Benchmarks (`scripts/quickstart_elliptical.py`)

Run sequentially, no other load. Both match the expected baseline values
(z ~58.8 Hz Vinante, ~24.9 Hz Fuchs).

| Benchmark | z-mode [Hz] | Published [Hz] | Rel. err | Wall time (stages) |
|---|---|---|---|---|
| Vinante (`parameters_vinante.py`) | **58.851** | 56.5 | **+4.16 %** | 19.4 s (mesh 0.3, equil 3.6, Hessian 3.7, report/plots ~11) |
| Fuchs (`parameters_fuchs.py`) | **24.915** | 26.7 | **-6.68 %** | 80.4 s (mesh 0.8, equil 36.0, Hessian 12.3, report/plots ~31) |

Full mode tables (for regression comparison):

Vinante (0.3 mm mesh, 1943 verts / 8691 tets / 2038 SC facets;
eq r = (0, 0, +0.2869) mm, theta = +1.5708, phi = +0.0500):

| mode | f [Hz] | stable | dominant |
|---|---|---|---|
| 1 | -3.1336 | no | phi |
| 2 | 1.2479 | yes | y |
| 3 | 1.4046 | yes | x |
| 4 | 58.8511 | yes | z |
| 5 | 394.5409 | yes | theta |

(U_eq = +2.324476e-12 J. The unstable phi mode is expected for a sphere:
phi-libration of an in-plane moment in a circularly symmetric cavity is
~neutral; its sign sits in the noise floor.)

Fuchs (0.2 mm mesh, 6919 verts / 34928 tets / 5106 SC facets;
eq r = (-0.0046, -0.0038, +1.9937) mm, theta = +1.5788, phi = +1.5814;
U_eq = +3.023906e-08 J):

| mode | f [Hz] | stable | dominant |
|---|---|---|---|
| 1 | 24.9154 | yes | z |
| 2 | 40.5519 | yes | x |
| 3 | 59.8261 | yes | y |
| 4 | 69.1821 | yes | phi |
| 5 | 126.1869 | yes | theta |

Note: total wall time is nowhere near the historical ~20 min figure — the
`_PreparedLaplace` factorisation cache already collapsed the per-solve cost;
equilibrium + Hessian on Fuchs is now ~48 s, and ~1/3 to 1/2 of each
benchmark's wall time is matplotlib plotting/report, not physics.

## 4. Profile (cProfile, coarse Vinante-like pipeline)

Driver: `scripts/profile_driver.py` — same pipeline as the quickstart
(mesh -> 5D Nelder-Mead equilibrium -> 5x5 Hessian) on a coarse 0.5 mm mesh
(577 verts, 2175 tets, 808 SC facets), P2, `subtract_singularity=True`.
Command: `python -m cProfile -o /tmp/sctrap_profile.pstats scripts/profile_driver.py`

Stage timings (under profiler): mesh 0.3 s, equilibrium 4.5 s (35 U-evals),
Hessian 6.2 s (51 U-evals), total 11.1 s. Output sanity: z-mode 59.28 Hz,
theta-mode 396.2 Hz (consistent with full-mesh Vinante numbers).

Top cumulative entries (12.13 s total under profiler):

```
ncalls  tottime  cumtime  filename:lineno(function)
  86      0.001   10.741  potential.py:15(U_mag)
  86      0.003   10.531  solver.py:177(solve_phi)
  87      0.002   10.112  skfem assembly asm()
  86      0.041   10.052  skfem linear_form.py:18(_assemble)
 860      0.376    9.870  solver.py:229(neumann_rhs)        <- RHS kernel
1806      8.813    8.900  dipole.py:17(B_dipole)            <- THE hotspot
 860      0.241    4.705  solver.py:83(_B_field_on_quadpts) (direct dipole term)
 946      0.045    4.592  half_space.py:113(B_halfspace_image) (image term)
  86      0.005    0.216  solver.py:145(solve_rhs)          (SuperLU back-sub)
  86      0.003    0.208  solver.py:269(B_induced_at)       (6-pt FD gradient)
  86      0.007    0.159  skfem cell_basis.py:197(probes)
   1      0.100    0.100  scipy splu (one-time factorisation, cached)
```

### Where the time actually goes (evidence-based)

- **82 % of runtime is Neumann RHS assembly** (`asm(neumann_rhs)`, 10.1 s of
  12.1 s), and within it **73 % is `B_dipole`** (8.9 s) evaluated at the SC
  surface quadrature points: 808 facets x 16 quad points (intorder=8)
  = ~13 k points per field, x2 fields when singularity subtraction is on
  (direct + image), x86 solves ≈ 2.2M point-evals... at an anomalously high
  4.9 ms per 13k-point vectorised call (fractional powers `r2**-1.5`,
  `r2**-2.5` + `np.where` guards dominate inside `B_dipole`).
- **The linear solve is already negligible**: one-time `splu` factorisation
  0.10 s + 86 back-substitutions totalling 0.21 s (~2 %). The
  `_PreparedLaplace` cache is doing its job.
- **`B_induced_at` is NOT the bottleneck**: 0.21 s total (~1.7 %), i.e.
  ~2.4 ms per call for the 6-point FD stencil (six `probes` interpolations).
  The original hypothesis that `basis.interpolator` at 6 FD points dominates
  is **refuted** at this mesh size. Fix 1 remains worthwhile for *accuracy*
  (no FD step-size dependence, exact element-local P2 gradient) but will not
  move total runtime materially; the performance lever is (a) fewer solves
  (Fix 4) and (b) cheaper RHS assembly per solve.

### Implications for the planned fixes

- Fix 1: justify on accuracy grounds (h-dependence elimination), not speed.
- Fix 4 (fewer full solves) attacks the true cost driver: every U-eval pays
  ~0.12 s of RHS assembly on the coarse mesh (much more on the production
  0.2 mm Fuchs mesh); equilibrium + Hessian = 86 solves here.
- A cheap independent win (not in the fix list, noting for later): rewrite
  `B_dipole`'s `r2**(-1.5)`/`r2**(-2.5)` as multiplications of one
  `1/sqrt(r2)` and hoist the `np.where` guards — same math, large constant-
  factor saving on the 73 % hotspot.
