# sctrap

> **Status: alpha (v0.1).** Research code under active development. API may
> change between minor versions. Validated against analytic limits and
> published levitation experiments; see Status section below for accuracy.

Trap frequencies of a magnetic dipole levitated above an arbitrary
superconductor geometry — from a `gmsh` `.msh` file in one command.

The package solves the magnetic-scalar-potential Laplace problem on the air
domain with Neumann (`n·∇Φ = -n·B_dipole`) on the superconductor and
Dirichlet (`Φ = 0`) on the far-field boundary, using `scikit-fem` (pure
Python, NumPy / SciPy only). No conda, no MPI, no PETSc.

## Installation

`sctrap` requires Python 3.11 or later and installs with `pip` alone — no
conda, MPI, or PETSc.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The `[mesh]` extra adds `gmsh`, which is required to generate the built-in
trap geometries and to import CAD files:

```bash
pip install -e ".[mesh]"
```

## Quick start

```bash
sctrap simulate --mesh examples/two_plates.msh --subtract-singularity
```

This loads the mesh, finds the equilibrium position (gravity included via
`U_mag + m·g·z`), and prints the five trap frequencies (x, y, z, θ, φ) at
equilibrium. Additional options: `--out results.json` writes the result to
disk, `--modes` computes the full 5×5 Hessian and normal modes, `--converge`
adds Richardson uncertainty bands, and `--report DIR/` writes a complete
PNG/CSV/JSON report.

The `--subtract-singularity` flag is strongly recommended: it is the single
most important accuracy control, reducing the two-plate off-midplane error
from 51 % to 2 % at no additional mesh-refinement cost.

## Mesh studio (browser UI)

For interactive geometry/mesh tweaking with a live 3D preview:

```bash
pip install -e ".[web]"
sctrap-serve              # then open http://127.0.0.1:8765
```

The page exposes the cavity dimensions and mesh-resolution parameters as
sliders and re-meshes live (debounced). **Download .msh** saves a file the
rest of the pipeline consumes verbatim, and **Copy parameters.py snippet**
produces a configuration block for your `parameters.py`. The UI runs entirely on `localhost`
and ships its own copy of Three.js, so no network is needed once
installed.

Closed elliptical cavities only in v0.1; open traps + surface tagging
follow.

## Unified quickstart

One command, one parameters file. Two presets ship with the repo:

```bash
# Vinante 2020: closed circular Pb cavity, NdFeB sphere
python scripts/quickstart_elliptical.py examples/parameters_vinante.py
# -> z-mode 53.9 Hz  vs paper 56.5 Hz  (-4.6 %)
# -> beta-mode 423.7 Hz vs paper 377 Hz  (+12.4 %)

# Fuchs 2024: closed elliptical Ta cavity, composite (3 cubes + bead) magnet
python scripts/quickstart_elliptical.py examples/parameters_fuchs.py
# -> z-mode 43.6 Hz (point-dipole approximation; see Status for the
#    expected departure from experiment when the magnet is not small
#    compared to the cavity)
```

To run your own geometry, copy one of the example preset files, edit the
numbers, and pass it as the argv. With no argument it reads
`scripts/parameters.py`.

Each run writes a full report under `results/<OUT_DIR>/`:
`summary.txt`, `result.json`, `modes.csv`, `mesh.png`, `particle.png` (composite
only), `mode_shapes.png`, `frequencies.png`, `hessian_card.png`,
`U_along_DOFs.png`, `U_xz_slice.png`, `U_xy_slice.png`,
`B_induced_xz.png`, `U_long_axes.png`.

### Tilt sweep

```bash
python scripts/tilt_sweep.py examples/parameters_vinante.py
```

Sweeps `TILT_ANGLES_DEG = [0, 0.5, 1, 1.5, 2, 3, 4, 5]`, warm-starts
equilibrium between angles, dumps `sweep.csv`, `sweep.json`,
`f_vs_tilt.png`, `mode_track.png` (mass-weighted DOF heatmap per branch).

## Mesh contract

A `.msh` should contain three physical groups:

| Group name (substring match, case-insensitive) | Dim | Role                  |
| ---------------------------------------------- | --- | --------------------- |
| `SC` (or `superconductor`)                     | 2   | Neumann surface       |
| `FF` (or `far_field`, `outer`, `infinity`)     | 2   | Dirichlet `Φ = 0`     |
| any name                                       | 3   | air domain (volume)   |

If named groups are missing, `sctrap` falls back to a geometric heuristic
(outermost boundary = far field, everything else = SC) and warns.

## Built-in geometries

```bash
sctrap mesh plates  --L 0.015 --t 5e-4 --d 4e-3 --R 0.06 --out plates.msh
sctrap mesh ellipse --a 0.01  --b 0.012 --h 4e-3 --R 0.06 --out ellipse.msh
sctrap mesh bowl    --r 5e-3  --R 0.05  --out bowl.msh
```

For closed-cavity traps (the magnet sits *inside* a hollow SC volume —
e.g. the Fuchs 2024 geometry), use the Python API:

```python
from sctrap.generators import elliptical_cavity
elliptical_cavity(a=2.25e-3, b=1.75e-3, height=4.7e-3,
                  mesh_size_sc=0.5e-3, output_file="cavity.msh")
```

This produces a mesh tagged with only an `SC` group (no `FF`), which
`load_msh` recognises and routes through the pure-Neumann solver path
(one DOF gauge-pinned).

All units are SI (metres). Two example meshes ship in `examples/`.

## Import your own CAD geometry

Draw your superconductor in any CAD package and export it as **STEP**
(`.step`/`.stp`; IGES and BREP also work). `sctrap` imports the solid,
wraps it in a far-field sphere, meshes the air domain around it, and tags
`SC`/`FF`/`air` automatically:

```bash
sctrap mesh import --cad my_trap.step --out my_trap.msh
sctrap simulate --mesh my_trap.msh --subtract-singularity --modes
```

`simulate` prints the equilibrium position, the **levitation height**, and
the five trap frequencies. If your CAD file is in millimetres, convert to
SI metres with `--scale 1e-3`. From Python:

```python
from sctrap.generators import import_cad
import_cad("my_trap.step", scale=1e-3, mesh_size_sc=0.5e-3,
           output_file="my_trap.msh")
```

A worked example, `examples/sc_disc.step`, ships with the package.

> **STL is not supported yet** — it is a faceted mesh format and gmsh's
> reconstruction of a meshable solid from arbitrary STL is not robust. Export
> STEP instead (every CAD tool can). Tracked for a future release.

## Validate against the analytic image-dipole result

```bash
sctrap validate
```

Generates a parallel-plate mesh, sweeps `U_mag(z)` between the plates, and
compares against the image-dipole series (arXiv:2504.18852). Pass criterion
defaults to <10% max relative error.

## Python API

```python
import numpy as np
from sctrap import (
    load_msh, dipole_moment, find_equilibrium, frequencies_at,
    MAGNET_MOMENT, MAGNET_MASS, MAGNET_INERTIA,
)

sct = load_msh("examples/two_plates.msh")
m   = dipole_moment(MAGNET_MOMENT, theta=np.pi/2, phi=0.0)
eq  = find_equilibrium(sct, m, r0_guess=np.array([0.0, 0.0, 1e-4]))
tf  = frequencies_at(sct, eq, m, mass=MAGNET_MASS, inertia=MAGNET_INERTIA)
print(tf)
```

## Tests

```bash
pip install pytest
pytest tests/ -m "not slow"     # ~0.5 s
pytest tests/                   # includes the slow FEM benchmark, ~25 s
```

## Layout

```
src/sctrap/
    config.py        physical constants
    dipole.py        analytic B-field of a point dipole (vectorised numpy)
    half_space.py    image-dipole reflection through a tangent plane
    particle.py      SphericalParticle, CompositeParticle
    mesh.py          load_msh: meshio + scikit-fem, named-group detection
    solver.py        scikit-fem Laplace solve (P1/P2, singularity subtraction)
    potential.py     U_mag, U_total, gravity_potential, gravity_from_tilt
    frequencies.py   find_equilibrium (g_vec-aware), 5-DOF parabola fit
    modes.py         5x5 Hessian + eigendecomposition, mass-weighted labels
    generators.py    optional gmsh mesh builders (cavity, ellipse, plates, bowl)
    report.py        text + plot artifacts (summary, JSON, CSV, plots)
    sanity_plots.py  mesh / particle / B-field / Hessian diagnostic plots
    validation.py    image-dipole series + two-plate benchmark
    cli.py           sctrap CLI (simulate, mesh, validate)
    web/             FastAPI mesh studio + vendored Three.js viewer

scripts/
    quickstart_elliptical.py  unified one-command pipeline
    tilt_sweep.py             angle sweep with mode tracking
    serve.py                  launch the browser UI
    parameters.py             user-editable defaults
    replot.py                 redraw artifacts from a cached result.json

examples/
    parameters_vinante.py     sphere magnet, 4 mm circular Pb cavity
    parameters_fuchs.py       composite magnet, 4.5x3.5 mm Ta cavity
    parameters_bar.py         single bar magnet (finite-size)
    parameters_cylinder.py    single cylinder magnet (finite-size)
    sc_disc.step              example CAD geometry for `sctrap mesh import`
    two_plates.msh
    elliptical_cylinder.msh
```

## Status (May 2026)

- **Accuracy** — two-plate sweep, uniform 2 mm P2 mesh + singularity
  subtraction: max 1.96 % across z = ±1.6 mm, <0.1 % near the plates.
  Worst-case error without subtraction was 51 %.
- **Vinante 2020 benchmark** — closed circular Pb cavity + sphere magnet.
  z-mode 53.9 Hz vs paper 56.5 Hz (-4.6 %); β-mode 423.7 Hz vs 377 Hz
  (+12.4 %); equilibrium 341 µm vs 311 µm (+9.6 %). For homogeneous
  spheres the external field is *exactly* a point dipole at the centre,
  so the point-dipole solver is physically correct here.
- **Fuchs 2024 benchmark** — closed elliptical Ta cavity + composite
  (3 cubes + bead) magnet. Within the point-dipole approximation, the
  predicted z-mode departs from the measured value as expected when
  the magnet's extent (~0.75 mm bar) is no longer small compared to
  the cavity (~4.7 mm). Inertia and mass are not the limitation;
  finite-extent magnetisation support is the next planned improvement.
- **Gravity** is included automatically via `gravity_potential`. For
  tilt studies pass a non-default `g_vec = gravity_from_tilt(angle, axis)`
  to `find_equilibrium` and `normal_modes`.
- **Closed-trap support** — an `SC`-only mesh routes through a pure-Neumann
  solve with one interior DOF pinned. Generator: `elliptical_cavity`.
- **Mode labels** — mass-weighted eigenvectors (`NormalModes.eigvecs_mw`)
  drive the per-mode composition table, so a 100 % z-translation mode
  reads as `z: 100 %` instead of `phi: 100 %` (m-vs-rad units bug, fixed).
- **Particle kinds** — `SphericalParticle` (Vinante-style) or
  `CompositeParticle` (Fuchs-style, full 3×3 inertia about COM,
  interpenetration check). Pick via `PARTICLE_KIND` in `parameters.py`.
- **Mesh refinement** (`--size-near` / `--refine-distance`) is available
  but **off the default path** — size-field slivers raise residual error
  vs. uniform-mesh + subtraction. Treat as experimental.
- **Browser UI** (`sctrap-serve`) ships with closed-elliptical
  parametric meshing + Three.js viewer. End-to-end verified on Chrome.
  Safari init bug under investigation (diagnostic instrumentation in
  `viewer.js v3` localises the offending step on next page load).
- **Runtime** — dominated by the sparse direct Laplace solve and the
  Nelder-Mead equilibrium finder. Vinante (1.9 k-vertex P2 mesh):
  ~11 min equilibrium + ~5 min Hessian + ~5 min sanity plots.
