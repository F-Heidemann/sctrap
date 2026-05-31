"""Smoke tests for mesh I/O. Skipped if gmsh is not available."""

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")


def test_load_named_groups(tmp_path):
    from sctrap.generators import two_parallel_plates
    from sctrap.mesh import load_msh

    path = tmp_path / "plates.msh"
    two_parallel_plates(
        plate_half_side=10e-3, plate_thickness=0.5e-3,
        plate_separation=4e-3, far_radius=40e-3,
        mesh_size_sc=2.5e-3, mesh_size_ff=15e-3,
        output_file=str(path),
    )
    sct = load_msh(path)

    assert not sct.used_heuristic
    assert sct.sc_name is not None and "SC" in sct.sc_name
    assert sct.ff_name is not None and "FF" in sct.ff_name
    assert sct.sc_facets.size > 0
    assert sct.ff_facets.size > 0

    # FF facet centroids should be farther from origin than any SC centroid
    f = sct.mesh.facets
    p = sct.mesh.p
    ff_r = np.linalg.norm(p[:, f[:, sct.ff_facets]].mean(axis=1), axis=0)
    sc_r = np.linalg.norm(p[:, f[:, sct.sc_facets]].mean(axis=1), axis=0)
    assert ff_r.min() > sc_r.max()
