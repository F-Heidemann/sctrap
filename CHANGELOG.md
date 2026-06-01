# Changelog

All notable changes to `sctrap` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to use
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- **CAD import** (`sctrap.generators.import_cad`, CLI `sctrap mesh import`):
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
- **The quickstart now relaxes orientation, not just position.** It uses
  `find_equilibrium_5d` to jointly minimise over (x, y, z, θ, φ), so the
  preferred orientation is *found* rather than assumed and the Hessian is
  taken at a true minimum. For the Fuchs cavity the bar is released from the
  long axis (`EQ_PHI = 0`) and the solver settles onto the **short** (y) axis
  — the orientation energy minimum (the short-axis-preference result of the
  cuboidal analysis) — where all five modes are stable. Set
  `FIND_ORIENTATION = False` in the parameters file to pin (EQ_THETA, EQ_PHI)
  instead. The Fuchs preset mesh is also finer (0.3 mm) for a better-resolved
  z-stiffness. (A negative mode frequency at a *pinned* non-preferred
  orientation is correct physics — a saddle — not a solver failure.)
- **`magpylib>=5.0`** is now required (was `>=4.0`). The finite-size particle
  field relies on the magpylib-5 `magnetization` (A/m) convention; magpylib 4
  interprets the same call as polarization in mT and would be wrong by ~1e6.
- **`requires-python>=3.11`** (was `>=3.9`) to match magpylib 5.
- Added PyPI classifiers, keywords, and project URLs.

### Fixed
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
- **STL import is not yet supported.** STL is a faceted mesh format and
  gmsh's reconstruction of a meshable solid from arbitrary STL is not robust
  enough to ship; `import_cad` raises a clear error pointing to STEP. Tracked
  for a future release.

## [0.1.0]
- Initial alpha: scikit-fem Laplace solver with singularity subtraction,
  closed- and open-trap support, 5×5 Hessian normal modes, gravity/tilt,
  `SphericalParticle` / `CompositeParticle`, reporting, browser mesh studio,
  and the two-plate / Vinante / Fuchs benchmarks.
