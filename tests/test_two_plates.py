"""End-to-end FEM accuracy test on a two-plate geometry.

Marked slow because P2 on the validation mesh takes ~30 s.
"""

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")


@pytest.mark.slow
def test_two_plate_midplane_energy(tmp_path):
    from sctrap import MAGNET_MOMENT
    from sctrap.generators import two_parallel_plates
    from sctrap.mesh import load_msh
    from sctrap.potential import U_mag
    from sctrap.validation import U_analytic_two_plates

    path = tmp_path / "plates.msh"
    two_parallel_plates(
        plate_half_side=15e-3, plate_thickness=0.5e-3,
        plate_separation=4e-3, far_radius=60e-3,
        mesh_size_sc=1.5e-3, mesh_size_ff=12e-3,
        output_file=str(path),
    )
    sct = load_msh(path)

    m  = np.array([MAGNET_MOMENT, 0.0, 0.0])
    r0 = np.zeros(3)
    U_fem = U_mag(sct, m, r0, element="P2")
    U_ana = U_analytic_two_plates(0.0, m, 4e-3, n_images=80)

    rel = abs(U_fem - U_ana) / abs(U_ana)
    assert rel < 0.10, f"FEM/analytic disagree: {rel*100:.2f} %  fem={U_fem}  ana={U_ana}"


def test_image_series_symmetry():
    """At z=0 the image series for an x-oriented dipole gives B along -x only."""
    from sctrap import MAGNET_MOMENT
    from sctrap.validation import _image_dipoles
    from sctrap.dipole import B_dipole

    m  = np.array([MAGNET_MOMENT, 0.0, 0.0])
    r0 = np.zeros(3)
    images = _image_dipoles(r0, m, d=4e-3, n_images=20)
    B = np.zeros(3)
    for r_im, m_im in images:
        B += B_dipole(r0[None, :], m_im, r_im)[0]
    assert abs(B[1]) < 1e-12 * abs(B[0])
    assert abs(B[2]) < 1e-12 * abs(B[0])
    assert B[0] < 0   # parallel images attract -> field opposes m
