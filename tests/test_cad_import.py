"""Tests for CAD import (`sctrap.generators.import_cad`).

STEP/IGES/BREP are imported through OpenCASCADE and meshed with a boolean
cut against a far-field sphere; the result must load through `load_msh`
with named ``SC`` and ``FF`` groups. STL is deliberately unsupported and
must raise a clear error rather than crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

gmsh = pytest.importorskip("gmsh")

from sctrap.generators import import_cad
from sctrap.mesh import load_msh

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture()
def step_disc(tmp_path):
    """A small SC disc authored in SI metres, written as STEP."""
    out = tmp_path / "disc.step"
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("disc")
    gmsh.model.occ.addCylinder(0, 0, -2e-3, 0, 0, 2e-3, 5e-3)
    gmsh.model.occ.synchronize()
    gmsh.write(str(out))
    gmsh.model.remove()
    return out


@pytest.mark.slow
def test_import_step_builds_loadable_trap(step_disc, tmp_path):
    msh = import_cad(step_disc, mesh_size_sc=1.0e-3,
                     output_file=tmp_path / "disc.msh")
    sct = load_msh(msh)
    # The air domain must be tetrahedralised with both named boundary groups.
    assert sct.sc_facets is not None and len(sct.sc_facets) > 0
    assert sct.ff_facets is not None and len(sct.ff_facets) > 0
    assert sct.mesh.t.shape[1] > 0


@pytest.mark.slow
def test_import_step_scale_converts_units(step_disc, tmp_path):
    """`scale` multiplies imported coordinates (e.g. mm -> m)."""
    big = load_msh(import_cad(step_disc, mesh_size_sc=1e-3,
                              output_file=tmp_path / "a.msh"))
    small = load_msh(import_cad(step_disc, scale=0.5, mesh_size_sc=0.5e-3,
                                output_file=tmp_path / "b.msh"))
    lo_b, hi_b = big.bounding_box()
    lo_s, hi_s = small.bounding_box()
    # Halving the scale halves the (far-field-dominated) bounding box.
    ratio = (hi_s[2] - lo_s[2]) / (hi_b[2] - lo_b[2])
    assert 0.45 < ratio < 0.55


@pytest.mark.slow
def test_shipped_example_step_imports(tmp_path):
    step = EXAMPLES / "sc_disc.step"
    if not step.exists():
        pytest.skip("example STEP not present")
    sct = load_msh(import_cad(step, mesh_size_sc=1e-3,
                              output_file=tmp_path / "ex.msh"))
    assert len(sct.sc_facets) > 0


def test_stl_raises_not_implemented(tmp_path):
    """STL is not supported; the error must be clear, not a crash."""
    stub = tmp_path / "thing.stl"
    stub.write_text("solid x\nendsolid x\n")
    with pytest.raises(NotImplementedError, match="STEP"):
        import_cad(stub, output_file=tmp_path / "x.msh")


def test_unsupported_format_raises(tmp_path):
    stub = tmp_path / "thing.xyz"
    stub.write_text("nonsense")
    with pytest.raises(ValueError, match="Unsupported"):
        import_cad(stub, output_file=tmp_path / "x.msh")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_cad(tmp_path / "nope.step", output_file=tmp_path / "x.msh")
