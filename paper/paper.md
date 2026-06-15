---
title: 'sctrap: trap frequencies of magnetically levitated particles over arbitrary superconductor geometries'
tags:
  - Python
  - superconductivity
  - magnetic levitation
  - magnetomechanics
  - finite element method
  - levitated optomechanics
authors:
  - name: Francis Headley
    orcid: 0009-0000-7585-4957
    affiliation: 1
affiliations:
  - name: Institut für Theoretische Physik, Eberhard Karls Universität Tübingen, Germany
    index: 1
date: 11 June 2026
bibliography: paper.bib
---

# Summary

`sctrap` computes the static equilibrium and the small-oscillation trap
frequencies of a magnetically levitated particle held above an arbitrary
superconductor (SC) geometry. Given a finite-element mesh of the
superconductor surface, the package returns the equilibrium position, the
levitation height, and the five trap frequencies of the magnet — three
translational ($x, y, z$) and two librational ($\theta, \phi$) — in a single
command.

The physics is the Meissner effect: a superconductor expels magnetic flux, so
the field of a nearby magnet induces screening currents whose field, in turn,
confines the magnet. `sctrap` models this by solving Laplace's equation for the
magnetic scalar potential $\Phi$ on the air domain surrounding the
superconductor, imposing the perfect-diamagnet boundary condition
$\mathbf{n}\cdot\mathbf{B}_\mathrm{total}=0$ (a Neumann condition
$\mathbf{n}\cdot\nabla\Phi = -\,\mathbf{n}\cdot\mathbf{B}_\mathrm{source}$) on
the superconductor surface and $\Phi = 0$ on a far-field truncation boundary.
The trap (self-)energy of the magnet is the image self-energy
$U = -\tfrac{1}{2}\,\mathbf{m}\cdot\mathbf{B}_\mathrm{induced}$, from which the
equilibrium is found by minimisation and the trap frequencies by a parabolic
fit of the $5\times5$ energy Hessian at equilibrium.

The solver is built on `scikit-fem` [@scikit-fem] and uses quadratic
tetrahedral elements with an optional analytic singularity-subtraction scheme
that removes the $|\mathbf{r}-\mathbf{r}_0|^{-3}$ behaviour of the dipole field
on the nearest superconductor facet. Meshes are generated with `gmsh`
[@gmsh] — including built-in parametric trap geometries and import of arbitrary
CAD solids — and finite-size magnet bodies are described with analytic fields
from `magpylib` [@magpylib]. The implementation is pure Python and installs with
`pip` alone: it requires no conda environment, MPI, or PETSc, lowering the
barrier to reproducing and extending levitation calculations.

# Statement of need

Magnetically levitated micro- and milligram particles in superconducting traps
are an active platform for precision force sensing, tests of quantum mechanics
at large mass, and magnetomechanics [@vinante2020; @fuchs2024]. Designing such a
trap requires predicting its mechanical mode frequencies from the
superconductor geometry, the magnet, and gravity — quantities that are measured
directly in experiment and that set the device's sensitivity and the regime of
operation.

These predictions are usually obtained either from analytic image-dipole models
valid only for simple geometries (a flat plane, a sphere in a uniform field), or
from general-purpose commercial finite-element packages that require substantial
setup, licensing, and expertise to apply correctly to the magnetostatic-scalar
problem with the subtle image self-energy factor. There is a gap for a small,
open, installable tool that takes a mesh of *any* superconductor geometry and
returns the trap frequencies directly, with the physics conventions fixed
correctly and validated.

`sctrap` fills that gap. It is aimed at experimental and theoretical groups
working on levitated superconducting magnetomechanics who need fast,
reproducible trap-frequency predictions for non-trivial geometries — closed
cavities, open traps, and imported CAD solids — without standing up a heavy
simulation stack. The package exposes both a command-line interface and a Python
API, ships parametric mesh generators and an optional browser-based mesh studio,
and writes a complete report (frequencies, mode shapes, potential scans, and the
raw data behind every plot in SI units) for each run.

# Validation

The accuracy of `sctrap` is established at two levels. Against the analytic
image-dipole series for a magnet between two parallel superconducting plates
[@imagedipole2025], the two-plate frequency sweep agrees to better than 2 %
across the trap volume on a uniform quadratic mesh with singularity
subtraction. Against published experiments, the package reproduces measured
$z$-mode frequencies:

| Benchmark | System | `sctrap` | Reference | Deviation |
|-----------|--------|---------:|----------:|----------:|
| @vinante2020 | circular Pb cavity, NdFeB sphere | 58.8 Hz | 56.5 Hz | $+4\%$ |
| @fuchs2024 | elliptical Ta cavity, composite magnet | 24.9 Hz | 26.7 Hz | $-7\%$ |

The same calculation also reproduces the experimentally observed preferred
orientation of an anisotropic magnet in an anisotropic cavity. A key correctness
detail, documented and tested in the package, is the factor of $\tfrac12$ in the
image self-energy $U = -\tfrac12\,\mathbf{m}\cdot\mathbf{B}_\mathrm{induced}$;
omitting it inflates every stiffness by $2\times$ and every trap frequency by
$\sqrt 2$.

# Acknowledgements

<!-- TODO: acknowledge funding sources, collaborators, and any group whose
experimental data was used for benchmarking. -->

# References
