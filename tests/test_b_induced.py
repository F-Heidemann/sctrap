"""Tests for the element-local (exact) gradient in `B_induced_at`.

The P2 interpolant is a polynomial inside each tetrahedron, so a centred
finite difference whose stencil stays inside one element is *exact* (up to
rounding) for it — the two methods must therefore agree to ~machine
precision away from element faces. The exact method additionally must have
no step-size (h) dependence and must reproduce the known analytic image
field for the two-plate geometry.
"""

import numpy as np
import pytest

from skfem import Basis, ElementTetP2, MeshTet

from sctrap.solver import B_induced_at, PhiSolution


def _interior_points(mesh, cells):
    """Barycentres of the given cells — guaranteed away from element faces."""
    return mesh.p[:, mesh.t[:, cells]].mean(axis=1)   # (3, len(cells))


def test_exact_matches_fd_away_from_faces():
    """Synthetic coeffs on a unit-cube mesh: exact == FD to <1e-6 relative."""
    mesh = MeshTet().refined(2)
    basis = Basis(mesh, ElementTetP2())
    rng = np.random.default_rng(42)
    phi = PhiSolution(coeffs=rng.standard_normal(basis.N), basis=basis)

    pts = _interior_points(mesh, np.array([3, 100, 250]))
    for j in range(pts.shape[1]):
        p = pts[:, j]
        B_exact = B_induced_at(phi, p, method="exact")
        B_fd = B_induced_at(phi, p, method="fd")
        scale = np.max(np.abs(B_exact))
        assert scale > 0
        assert np.max(np.abs(B_exact - B_fd)) / scale < 1e-6, (
            f"exact vs fd mismatch at {p}: {B_exact} vs {B_fd}"
        )


def test_exact_is_h_independent():
    """`h` is an FD knob; the exact method must ignore it entirely."""
    mesh = MeshTet().refined(2)
    basis = Basis(mesh, ElementTetP2())
    rng = np.random.default_rng(7)
    phi = PhiSolution(coeffs=rng.standard_normal(basis.N), basis=basis)

    p = _interior_points(mesh, np.array([10]))[:, 0]
    B1 = B_induced_at(phi, p, h=1e-3, method="exact")
    B2 = B_induced_at(phi, p, h=1e-7, method="exact")
    assert np.array_equal(B1, B2)

    # ... while FD answers genuinely depend on h on a generic (non-interior)
    # point, which is precisely the artefact the exact method removes.


def test_exact_gradient_of_known_polynomial():
    """Project Phi = x^2 + 2 y^2 - 3 z^2 + x y (a P2 function, represented
    exactly) and check grad(Phi) analytically at off-node points."""
    mesh = MeshTet().refined(3)
    basis = Basis(mesh, ElementTetP2())

    def f(x):
        return x[0] ** 2 + 2.0 * x[1] ** 2 - 3.0 * x[2] ** 2 + x[0] * x[1]

    coeffs = basis.project(f)
    phi = PhiSolution(coeffs=coeffs, basis=basis)

    pts = _interior_points(mesh, np.array([5, 333, 1500]))
    g = phi.grad(pts)
    for j in range(pts.shape[1]):
        x, y, z = pts[:, j]
        expected = np.array([2.0 * x + y, 4.0 * y + x, -6.0 * z])
        assert np.allclose(g[:, j], expected, rtol=1e-9, atol=1e-12)


@pytest.mark.slow
def test_two_plate_image_field(tmp_path):
    """On a real two-plate solve the exact B_induced must (a) agree with the
    legacy FD value away from element faces and (b) reproduce the analytic
    image-series field at the midplane within mesh accuracy."""
    pytest.importorskip("gmsh")
    from sctrap import MAGNET_MOMENT
    from sctrap.generators import two_parallel_plates
    from sctrap.mesh import load_msh
    from sctrap.solver import solve_phi
    from sctrap.validation import _image_dipoles
    from sctrap.dipole import B_dipole

    path = tmp_path / "plates.msh"
    two_parallel_plates(
        plate_half_side=15e-3, plate_thickness=0.5e-3,
        plate_separation=4e-3, far_radius=60e-3,
        mesh_size_sc=1.5e-3, mesh_size_ff=12e-3,
        output_file=str(path),
    )
    sct = load_msh(path)

    m = np.array([MAGNET_MOMENT, 0.0, 0.0])
    r0 = np.zeros(3)
    phi = solve_phi(sct, m, r0, subtract_singularity=True)

    # (b) analytic image-series field at the dipole position
    B_ana = np.zeros(3)
    for r_im, m_im in _image_dipoles(r0, m, d=4e-3, n_images=80):
        B_ana += B_dipole(r0[None, :], m_im, r_im)[0]
    B_num = B_induced_at(phi, r0, method="exact")
    rel = np.linalg.norm(B_num - B_ana) / np.linalg.norm(B_ana)
    assert rel < 0.05, f"exact B vs analytic image series: {rel*100:.2f} %"

    # (a) exact == FD at an element barycentre (stencil fully interior)
    mesh = phi.basis.mesh
    cell = mesh.element_finder()(*(np.array([[0.2e-3], [0.1e-3], [0.3e-3]])))
    p = mesh.p[:, mesh.t[:, cell[0]]].mean(axis=1)
    B_exact = B_induced_at(phi, p, method="exact")
    B_fd = B_induced_at(phi, p, h=1e-6, method="fd")
    scale = np.max(np.abs(B_exact))
    assert np.max(np.abs(B_exact - B_fd)) / scale < 1e-6
