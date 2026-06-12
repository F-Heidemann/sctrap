"""Tests for the analytic force/torque (Fix 4): `grad_B_dipole`,
`dipole_moment_derivatives`, `U_mag_and_grad`, and the gradient-based
equilibrium search.

The energy U(r0) = -1/2 m . B_ind(r0; r0) depends on r0 through the
evaluation point AND the source position in the boundary condition; by
reciprocity of the induced Green's function the two derivative
contributions are equal, so the force is the frozen-image form
F = +grad_r [m . B_ind(r; r0)] (no 1/2) and the torque is
dU/dq = -(dm/dq) . B_ind. The FD gradient of the *full* energy (source
moves too) is the ground truth these are checked against.
"""

import numpy as np
import pytest

from sctrap.dipole import (B_dipole, dipole_moment,
                           dipole_moment_derivatives, grad_B_dipole)


def test_grad_B_dipole_vs_fd():
    m = np.array([0.3, -0.5, 0.8]) * 1e-7
    r0 = np.array([0.1e-3, -0.2e-3, 0.05e-3])
    pts = np.array([[0.4e-3, 0.3e-3, 0.6e-3],
                    [-0.2e-3, 0.5e-3, -0.3e-3]])
    G = grad_B_dipole(pts, m, r0)
    h = 1e-9
    for n in range(pts.shape[0]):
        for i in range(3):
            e = np.zeros(3); e[i] = h
            dB = (B_dipole(pts[n] + e, m, r0)[0]
                  - B_dipole(pts[n] - e, m, r0)[0]) / (2 * h)
            assert np.allclose(G[n, i, :], dB, rtol=1e-6), (n, i)


def test_grad_B_dipole_symmetric_traceless():
    """curl B = 0 away from the source -> dB_j/dx_i symmetric;
    div B = 0 -> traceless."""
    m = np.array([1.0, 2.0, -1.5]) * 1e-8
    r0 = np.zeros(3)
    pts = np.random.default_rng(3).uniform(-1e-3, 1e-3, (5, 3)) + 2e-3
    G = grad_B_dipole(pts, m, r0)
    assert np.allclose(G, np.transpose(G, (0, 2, 1)), rtol=1e-12)
    assert np.allclose(np.trace(G, axis1=1, axis2=2), 0.0,
                       atol=1e-12 * np.abs(G).max())


def test_dipole_moment_derivatives_vs_fd():
    m_mag, theta, phi = 3.7e-8, 0.9, -1.3
    dth, dph = dipole_moment_derivatives(m_mag, theta, phi)
    h = 1e-7
    dth_fd = (dipole_moment(m_mag, theta + h, phi)
              - dipole_moment(m_mag, theta - h, phi)) / (2 * h)
    dph_fd = (dipole_moment(m_mag, theta, phi + h)
              - dipole_moment(m_mag, theta, phi - h)) / (2 * h)
    assert np.allclose(dth, dth_fd, rtol=1e-7)
    assert np.allclose(dph, dph_fd, rtol=1e-7)


# ---------------------------------------------------------------------------
# FEM-backed gate tests (slow; need gmsh)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_cavity(tmp_path_factory):
    gmsh = pytest.importorskip("gmsh")
    from sctrap.generators import elliptical_cavity
    from sctrap.mesh import load_msh
    path = tmp_path_factory.mktemp("cav") / "cavity.msh"
    elliptical_cavity(2.0e-3, 2.0e-3, 4.0e-3, mesh_size_sc=0.3e-3,
                      output_file=str(path))
    return load_msh(path)


M_MAG = 4.6583e-8      # Vinante-like sphere
MASS = 6.126e-10


@pytest.mark.slow
def test_analytic_gradient_matches_fd_energy(coarse_cavity):
    """THE gate: single-solve analytic 5D gradient vs the FD gradient of the
    full energy (source moves too), <1 % of the gradient norm where the FD
    is itself converged (near the equilibrium region, subtraction on)."""
    from sctrap.potential import U_mag, U_mag_and_grad
    sct = coarse_cavity
    kw = dict(element="P2", subtract_singularity=True)

    # Tilted, off-axis points: every gradient component (force AND torque)
    # has a genuine non-zero value to compare against. At symmetric points
    # the true torque vanishes and the check would compare noise with noise.
    for r0, th, ph in [
        (np.array([0.05e-3, -0.04e-3, 0.40e-3]), np.pi / 2 - 0.1, 0.2),
        (np.array([0.02e-3, -0.03e-3, 0.34e-3]), np.pi / 2 - 0.05, 0.1),
    ]:
        _, g = U_mag_and_grad(sct, M_MAG, th, ph, r0, **kw)
        h_pos, h_ang = 2e-6, 2e-3
        g_fd = np.empty(5)
        m_vec = dipole_moment(M_MAG, th, ph)
        for i in range(3):
            e = np.zeros(3); e[i] = h_pos
            g_fd[i] = (U_mag(sct, m_vec, r0 + e, **kw)
                       - U_mag(sct, m_vec, r0 - e, **kw)) / (2 * h_pos)
        g_fd[3] = (U_mag(sct, dipole_moment(M_MAG, th + h_ang, ph), r0, **kw)
                   - U_mag(sct, dipole_moment(M_MAG, th - h_ang, ph), r0, **kw)
                   ) / (2 * h_ang)
        g_fd[4] = (U_mag(sct, dipole_moment(M_MAG, th, ph + h_ang), r0, **kw)
                   - U_mag(sct, dipole_moment(M_MAG, th, ph - h_ang), r0, **kw)
                   ) / (2 * h_ang)

        # spatial: vector-relative (components 1000x below the dominant one
        # are noise in BOTH methods); angular: per-component
        rel_r = np.linalg.norm(g[:3] - g_fd[:3]) / np.linalg.norm(g_fd[:3])
        assert rel_r < 0.01, f"force vs FD energy gradient: {rel_r*100:.2f} %"
        ang_scale = max(np.abs(g_fd[3:]).max(), 1e-30)
        rel_a = np.abs(g[3:] - g_fd[3:]) / ang_scale
        assert np.all(rel_a < 0.01), f"torque vs FD: {rel_a*100} %"


@pytest.mark.slow
def test_gradient_equilibrium_matches_nelder_mead(coarse_cavity):
    """From a deliberately bad seed both methods must find the same
    equilibrium (within the documented noise-floor resolution), with the
    gradient method using several-fold fewer solves."""
    import sctrap.potential as P
    import sctrap.solver as S
    from sctrap.frequencies import find_equilibrium_5d

    counts = {}
    orig = S.solve_phi

    def counted(*a, **k):
        counts[key] = counts.get(key, 0) + 1
        return orig(*a, **k)

    S.solve_phi = counted
    P.solve_phi = counted
    try:
        seed = np.array([0.15e-3, -0.10e-3, 0.60e-3])
        results = {}
        for key in ("nelder-mead", "gradient"):
            results[key] = find_equilibrium_5d(
                coarse_cavity, M_MAG, seed, theta_guess=1.2, phi_guess=0.7,
                mass=MASS, method=key,
                element="P2", subtract_singularity=True)
    finally:
        S.solve_phi = orig
        P.solve_phi = orig

    r_nm, th_nm, _ = results["nelder-mead"]
    r_gr, th_gr, _ = results["gradient"]
    # z is the stiff DOF -> tight; x/y sit in a ~1 Hz flat basin at the
    # noise floor -> loose. (phi is exactly neutral for a sphere: excluded.)
    assert abs(r_gr[2] - r_nm[2]) < 5e-6
    assert np.linalg.norm(r_gr[:2] - r_nm[:2]) < 100e-6
    assert abs(th_gr - th_nm) < 0.05
    assert counts["gradient"] * 3 < counts["nelder-mead"], counts
