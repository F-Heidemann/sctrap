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

### Changed
- **`magpylib>=5.0`** is now required (was `>=4.0`). The finite-size particle
  field relies on the magpylib-5 `magnetization` (A/m) convention; magpylib 4
  interprets the same call as polarization in mT and would be wrong by ~1e6.
- **`requires-python>=3.11`** (was `>=3.9`) to match magpylib 5.
- Added PyPI classifiers, keywords, and project URLs.

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
