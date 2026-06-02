# Changelog

All notable changes to `sctrap` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to use
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- **Per-DOF potential-scan data export.** The report now writes the raw numbers
  behind the `U_along_DOFs` plot as CSV (SI units) into a `U_along_DOFs/`
  directory: one file per DOF (`U_x.csv` … `U_phi.csv`, columns `offset`, `U`,
  `U − U₀`, harmonic-fit `½ H_ii q²`) plus a combined `U_along_DOFs.csv`, each
  with a header recording `H_ii` and the Hessian half-width `h`. Experimentalists
  can load and re-plot/fit these directly. New public helpers
  `report.compute_U_along_dofs` / `report.write_U_along_dofs_data`;
  `plot_U_along_dofs` now returns the data dict and takes `write_data=True`. The
  plot also shades the ±h window the trap frequency is fit over, so the
  noise-floor structure beyond it is visibly out of scope.
  load a STEP/IGES/BREP solid as the superconductor, auto-wrap a far-field
  sphere, and write a meshed `SC`/`FF`/`air` trap geometry that the rest of
  the pipeline consumes verbatim. A `--scale` option converts CAD units
  (e.g. mm → m). Ships an example `examples/sc_disc.step`.
- `simulate` now reports the **levitation height** (equilibrium height above
  the bottom of the geometry) alongside the equilibrium position.
- Finite-size magnet particles `BarMagnetParticle`, `CylinderMagnetParticle`,
  `RingMagnetParticle` (magpylib-backed source fields) and the `B_source`
  hook in `solve_phi` / `U_mag_particle`.
- `tests/test_cad_import.py` covering STEP import, unit scaling, and the
  STL / unsupported-format error paths.
- **Progress indicators** for the slow steps: `find_equilibrium`,
  `find_equilibrium_5d`, and `normal_modes` take a `progress=True` flag and
  print a single live-updating line (eval count / `solve k/N` with elapsed +
  ETA). Enabled by default in `quickstart_elliptical.py` and `sctrap simulate`
  (suppressed under `--verbose`).

### Changed
- **Cached Laplace factorisation — every command is faster (≈5× and rising
  with mesh size).** The stiffness matrix, the far-field/gauge condensation,
  and therefore the sparse LU factorisation depend only on the mesh, not on the
  dipole position or moment; only the Neumann RHS changes between evaluations.
  `solve_phi` now assembles and factorises **once per mesh** (`_PreparedLaplace`,
  cached on the `SCTrapMesh`) and back-substitutes each call. An equilibrium
  search + 5×5 Hessian calls the solver dozens–hundreds of times on a fixed
  mesh, so this removes the dominant repeated cost. Measured 5× on a 7.3 k-DOF
  cavity; larger on finer meshes (factorisation scales super-linearly, the
  back-solve does not). Results are **numerically identical** to the previous
  per-call `spsolve` (machine-precision agreement; SuperLU under the hood
  either way) — a pure performance refactor, no accuracy trade-off.
- **The quickstart now relaxes orientation, not just position.** It uses
  `find_equilibrium_5d` to jointly minimise over (x, y, z, θ, φ), so the
  preferred orientation is *found* rather than assumed and the Hessian is
  taken at a true minimum. The Fuchs preset now *seeds* the short (y) axis
  (`EQ_PHI = π/2`) — the orientation energy minimum (the short-axis-preference
  result of the cuboidal analysis) — so the solver starts at the minimum
  rather than walking in from the long axis. Set `FIND_ORIENTATION = False`
  to pin (EQ_THETA, EQ_PHI) instead. (A negative mode frequency at a *pinned*
  non-preferred orientation is correct physics — a saddle — not a solver
  failure.)
- **Analytic seed + benchmark** (new module `sctrap.analytic_box`). A
  self-contained port of the closed cuboidal-cavity image-lattice theory
  (`cuboidal_analytic_trap`): a closed-form, few-µs estimate of the levitation
  height `z_eq` and the full diagonal Hessian (k_x, k_y, k_z, k_θ, k_φ) → all
  five mode frequencies. The Fuchs quickstart uses it to (a) **seed** `z_eq`
  (the optimiser starts *at* the minimum) and (b) print an **independent
  analytic benchmark** beside the FEM frequencies — which immediately catches
  gross FEM errors (it flagged a 2× z-mode artifact on a too-coarse mesh).
  Validated vs the reference: Fuchs `z_eq` 1.978 mm, analytic f_z 28.2 Hz.
  `half_space.image_equilibrium_height` gives the cruder single-image
  half-space height as a fallback for non-cuboidal geometries.
- **Equilibrium-finder convergence rewritten** (`find_equilibrium_5d`). The old
  `fatol = 1e-22` was ~1e3× below the FEM objective's numerical floor, so the
  search never converged and ran to `maxiter` (20+ min on Fuchs). Now: (1) the
  5D vector is non-dimensionalised so position (~mm) and orientation (~rad)
  share a balanced simplex; (2) `fatol = ftol_rel·|U₀|` with `ftol_rel = 1e-4`,
  safely above the measured **relative noise floor (~4e-5)**, so it actually
  converges; (3) `xatol` is disabled (a simplex-*size* criterion can never be
  met on a noise-limited objective and is what caused the grind). Fuchs
  equilibrium now converges in ~25 evals and stops cleanly. The leftover
  off-centre error (~5–10 µm) does not bias frequencies — the symmetric
  2nd-difference Hessian cancels the linear term.
- **Fuchs preset mesh → 0.2 mm** (was 0.3 mm). At 0.3 mm the FEM noise floor
  (~1e-12 J) exceeds the trap's energy variation over ~10 µm, which corrupts
  the normal-mode Hessian (spurious z-mode 62 Hz, heavily mixed mode shapes).
  0.2 mm lowers the floor enough to recover clean modes (z-mode 24.9 Hz, mode
  shapes >95–99 % pure) at ~12× the per-solve cost. **Use ≥0.2 mm for
  publishable frequencies; 0.3 mm is a quick coarse preview only.**
- **`magpylib>=5.0`** is now required (was `>=4.0`). The finite-size particle
  field relies on the magpylib-5 `magnetization` (A/m) convention; magpylib 4
  interprets the same call as polarization in mT and would be wrong by ~1e6.
- **`requires-python>=3.11`** (was `>=3.9`) to match magpylib 5.
- Added PyPI classifiers, keywords, and project URLs.

### Fixed
- **Mesh overview plot was blank.** `plot_mesh_overview` built the SC/FF facet
  polygons in metres but drew them inside millimetre axes (the vertex scatter,
  `r_eq` marker, and axis limits are all ×1e3), collapsing the whole shell onto
  the origin so only the axes and `r_eq` rendered. Facet vertices are now scaled
  to mm and the cavity surface draws correctly.
- **Self-energy ½ (trap frequencies were √2 too high).** The trap potential is
  the image self-energy `U = -½ m·B_induced` (Jackson §2.2); the package
  previously used the bare `-m·B_induced`, making every stiffness 2× and every
  trap frequency a factor √2 too high. Applied the ½ in `U_mag`,
  `_interaction_energy_from_phi`, and the two-plate reference series. After the
  fix both published benchmarks agree with experiment: Vinante z-mode ≈ 58.8 Hz
  (vs 56.5, +4 %) and Fuchs z-mode ≈ 25.4 Hz (vs 26.7, −5 %). An independent
  FEniCSx solve shows the Fuchs z-mode is the same for a point dipole and the
  finite bar, so finite magnet extent was **not** the cause of the earlier
  Fuchs discrepancy — the missing ½ was.

### Notes
- **Mesh resolution drives frequency accuracy via the noise floor.** The
  singularity-subtracted point-dipole objective is deterministic but
  *non-smooth* at ~1e-12 J (relative ~4e-5) because the nearest-facet image
  reflection jumps discretely as the dipole moves. This caps equilibrium
  resolution (~10 µm / ~1° at 0.3 mm) and, more importantly, corrupts the
  normal-mode Hessian when the mesh is too coarse. The relative floor is
  ~mesh-independent, so a single `ftol_rel` is robust — no per-mesh
  characterisation is needed. The diagnostic `scripts`-style probe that
  measures it lives in the dev notes, not the run path. Rule of thumb:
  **≥0.2 mm mesh for trustworthy frequencies; always sanity-check against the
  `analytic_box` benchmark the quickstart prints.**
- **STL import is not yet supported.** STL is a faceted mesh format and
  gmsh's reconstruction of a meshable solid from arbitrary STL is not robust
  enough to ship; `import_cad` raises a clear error pointing to STEP. Tracked
  for a future release.

## [0.1.0]
- Initial alpha: scikit-fem Laplace solver with singularity subtraction,
  closed- and open-trap support, 5×5 Hessian normal modes, gravity/tilt,
  `SphericalParticle` / `CompositeParticle`, reporting, browser mesh studio,
  and the two-plate / Vinante / Fuchs benchmarks.
