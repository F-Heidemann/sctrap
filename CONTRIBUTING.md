# Contributing to sctrap

Thanks for your interest in `sctrap`. It is a small, pure-pip scientific
package; contributions that keep it install-clean (NumPy/SciPy/scikit-fem,
optional gmsh) are very welcome.

## Development setup

```bash
git clone <your-fork>
cd sctrap
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # pytest + gmsh + matplotlib
```

## Running the tests

```bash
pytest -q -m "not slow and not benchmark"   # fast unit tests (~1 s)
pytest -q -m slow                           # mesh-building / FEM tests
pytest -q -m benchmark                      # published-experiment benchmarks
```

CI runs the fast suite on Python 3.11 and 3.12. Please make sure it passes
before opening a pull request, and add a test for any new behaviour.

## Conventions

- **SI units everywhere** (metres, tesla, A·m²). The one place a non-SI input
  is accepted is `import_cad(..., scale=...)`, which converts CAD coordinates
  to metres.
- Physics changes should be backed by a verification: an analytic limit, a
  symmetry check, or a published benchmark. The two-plate image-dipole series
  (`sctrap validate`) and the Vinante/Fuchs drivers are the references.
- Keep new geometry generators consistent with the existing ones in
  `generators.py` (named `SC` / `FF` / `air` physical groups).
- Match the surrounding code style; keep public functions documented with a
  short docstring stating units and the returned object.

## Reporting issues

Please include the geometry / parameters file, the `sctrap`, `gmsh`, and
`scikit-fem` versions, and the full traceback. For meshing problems, attach
the `.msh` if you can.
