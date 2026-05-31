"""Tests for volumetric (finite-size) magnet particle support.

Three groups:

1. Far-field convergence — each new particle's `B_field` should approach the
   point-dipole field of its total magnetic moment at distances large
   compared to the particle's largest dimension.
2. Solver interface pinning — wrapping `B_dipole` as a `B_source` callable
   should reproduce the existing `solve_phi` output to within the FEM noise
   floor (this protects the new `B_source` plumbing without introducing new
   physics).
3. End-to-end benchmarks — the Fuchs 2024 z-mode should improve once the bar
   is modelled volumetrically; the Vinante 2020 sphere benchmark must remain
   unchanged (it still goes through the point-dipole path).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Group 1: far-field reduction to a point dipole
# ---------------------------------------------------------------------------

magpylib = pytest.importorskip("magpylib")


def _max_extent_bar(p):
    return max(p.length, p.width, p.height)


def _max_extent_cyl(p):
    return max(2.0 * p.radius, p.length)


def _max_extent_ring(p):
    return max(2.0 * p.outer_radius, p.length)


def _far_field_grid(scale: float, multiplier: float = 30.0, n: int = 12):
    """A handful of probe points uniformly far from the magnet's COM.

    The leading finite-size correction to a uniformly magnetised body of
    extent L falls as ~(L/(2r))^2.  At r = 10 L the on-axis residual is
    ~0.25 %; we therefore probe at r = 30 L (residual ~0.03 %), so the
    0.1 % test threshold sits comfortably above the analytic floor.
    """
    r = multiplier * scale
    rng = np.random.RandomState(0)
    # Random unit directions, plus a few hand-picked axes for coverage.
    axes = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    ], dtype=float)
    rand = rng.randn(n - len(axes), 3)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    dirs = np.vstack([axes, rand])
    return r * dirs


@pytest.mark.parametrize("theta,phi", [(np.pi / 2, 0.0),
                                       (np.pi / 2, np.pi / 2),
                                       (1.0, 2.0),
                                       (np.pi, 0.0)])
def test_bar_far_field_matches_dipole(theta, phi):
    from sctrap.dipole import B_dipole, dipole_moment
    from sctrap.particle import BarMagnetParticle

    p = BarMagnetParticle(length=0.75e-3, width=0.25e-3, height=0.25e-3,
                          density=7500.0, B_remanence=1.4)
    pts = _far_field_grid(_max_extent_bar(p), multiplier=30.0)
    r0 = np.array([0.0, 0.0, 0.0])

    B_vol = p.B_field(pts, r0, theta, phi)
    m_vec = dipole_moment(p.moment, theta, phi)
    B_dip = B_dipole(pts, m_vec, r0)

    rel = np.linalg.norm(B_vol - B_dip, axis=1) / np.linalg.norm(B_dip, axis=1)
    assert rel.max() < 1e-3, (
        f"bar far-field disagrees with point dipole: max rel err = {rel.max():.3e}"
    )


@pytest.mark.parametrize("theta,phi", [(np.pi / 2, 0.0),
                                       (0.0, 0.0),
                                       (1.2, -0.7)])
def test_cylinder_far_field_matches_dipole(theta, phi):
    from sctrap.dipole import B_dipole, dipole_moment
    from sctrap.particle import CylinderMagnetParticle

    p = CylinderMagnetParticle(radius=0.125e-3, length=0.75e-3,
                               density=7500.0, B_remanence=1.4)
    pts = _far_field_grid(_max_extent_cyl(p), multiplier=30.0)
    r0 = np.zeros(3)

    B_vol = p.B_field(pts, r0, theta, phi)
    m_vec = dipole_moment(p.moment, theta, phi)
    B_dip = B_dipole(pts, m_vec, r0)

    rel = np.linalg.norm(B_vol - B_dip, axis=1) / np.linalg.norm(B_dip, axis=1)
    assert rel.max() < 1e-3, (
        f"cylinder far-field disagrees with point dipole: max rel err = {rel.max():.3e}"
    )


@pytest.mark.parametrize("theta,phi", [(np.pi / 2, 0.0),
                                       (np.pi / 3, 1.4),
                                       (np.pi, 0.0)])
def test_ring_far_field_matches_dipole(theta, phi):
    from sctrap.dipole import B_dipole, dipole_moment
    from sctrap.particle import RingMagnetParticle

    p = RingMagnetParticle(outer_radius=0.30e-3, inner_radius=0.10e-3,
                           length=0.50e-3, density=7500.0, B_remanence=1.4)
    pts = _far_field_grid(_max_extent_ring(p), multiplier=30.0)
    r0 = np.zeros(3)

    B_vol = p.B_field(pts, r0, theta, phi)
    m_vec = dipole_moment(p.moment, theta, phi)
    B_dip = B_dipole(pts, m_vec, r0)

    rel = np.linalg.norm(B_vol - B_dip, axis=1) / np.linalg.norm(B_dip, axis=1)
    assert rel.max() < 1e-3, (
        f"ring far-field disagrees with point dipole: max rel err = {rel.max():.3e}"
    )


# ---------------------------------------------------------------------------
# Group 2: B_dipole-as-B_source produces the same Phi as the legacy path
# ---------------------------------------------------------------------------

gmsh = pytest.importorskip("gmsh")


@pytest.mark.slow
def test_B_source_wrapping_dipole_is_bit_identical(tmp_path):
    """Wrapping B_dipole as a B_source callable should reproduce solve_phi.

    This pins the new code path without introducing new physics: any drift
    would mean the Neumann assembly has changed semantics.
    """
    from sctrap import MAGNET_MOMENT
    from sctrap.dipole import B_dipole
    from sctrap.generators import elliptical_cavity
    from sctrap.mesh import load_msh
    from sctrap.solver import solve_phi, B_induced_at

    mpath = tmp_path / "cav.msh"
    elliptical_cavity(
        2.0e-3, 2.0e-3, 4.0e-3,
        mesh_size_sc=0.6e-3,
        output_file=str(mpath),
        verbose=False,
    )
    sct = load_msh(mpath)

    m  = np.array([MAGNET_MOMENT, 0.0, 0.0])
    r0 = np.array([0.0, 0.0, 0.3e-3])

    # Legacy path.
    phi_a = solve_phi(sct, m, r0, element="P2")
    B_a = B_induced_at(phi_a, r0)

    # New path: identical analytic source, routed through B_source.
    def B_src(points):
        return B_dipole(points, m, r0)
    phi_b = solve_phi(sct, m, r0, element="P2", B_source=B_src)
    B_b = B_induced_at(phi_b, r0)

    # Coefficient vectors should be identical up to FP roundoff in the
    # linear solve (the only difference is one function indirection).
    diff = np.linalg.norm(phi_a.coeffs - phi_b.coeffs)
    ref  = np.linalg.norm(phi_a.coeffs) + 1e-30
    assert diff / ref < 1e-10, f"Phi coefficient drift = {diff/ref:.3e}"

    # And the induced field at the dipole must match.
    rel_B = np.linalg.norm(B_a - B_b) / (np.linalg.norm(B_a) + 1e-30)
    assert rel_B < 1e-9, f"B_induced drift = {rel_B:.3e}"


def test_B_source_rejects_subtract_singularity():
    """The half-space image construction is only meaningful for a point
    dipole; combining it with B_source is a programming error."""
    from sctrap.solver import solve_phi

    # We don't need a real mesh — the kwargs guard fires before any work.
    class _DummySct: ...

    with pytest.raises(ValueError):
        solve_phi(_DummySct(),
                  np.zeros(3), np.zeros(3),
                  subtract_singularity=True,
                  B_source=lambda p: np.zeros_like(p))


# ---------------------------------------------------------------------------
# Group 3: end-to-end benchmarks
# ---------------------------------------------------------------------------

def _z_mode_freq(nm):
    """Pick the mode whose mass-weighted eigenvector is dominated by z."""
    z_idx = int(np.argmax(np.abs(nm.eigvecs_mw[2, :])))
    return float(nm.f_Hz[z_idx])


@pytest.mark.slow
@pytest.mark.benchmark
def test_vinante_sphere_unchanged(tmp_path):
    """The Vinante 2020 sphere benchmark uses the point-dipole path; the new
    volumetric plumbing must not perturb it.  Tolerance 10 % of 56.5 Hz."""
    from sctrap.dipole import dipole_moment
    from sctrap.frequencies import find_equilibrium
    from sctrap.generators import elliptical_cavity
    from sctrap.mesh import load_msh
    from sctrap.modes import normal_modes
    from sctrap.particle import SphericalParticle

    mpath = tmp_path / "vinante.msh"
    elliptical_cavity(
        2.0e-3, 2.0e-3, 4.0e-3,
        mesh_size_sc=0.30e-3,
        output_file=str(mpath),
        verbose=False,
    )
    sct = load_msh(mpath)

    p = SphericalParticle(radius=27e-6, density=7430.0, B_remanence=0.71)
    m_vec = dipole_moment(p.moment, np.pi / 2.0, 0.0)
    eq_r = find_equilibrium(
        sct, m_vec, np.array([0.0, 0.0, 0.30e-3]),
        mass=p.mass, element="P2", subtract_singularity=True,
    )
    nm = normal_modes(
        sct, eq_r, p.moment,
        eq_theta=np.pi / 2.0, eq_phi=0.0,
        mass=p.mass, inertia=p.inertia,
        h_trans=5e-6, h_ang=5e-3,
        element="P2", subtract_singularity=True,
    )
    f_z = _z_mode_freq(nm)
    rel = abs(f_z - 56.5) / 56.5
    assert rel < 0.10, (
        f"Vinante z-mode drifted: f_z = {f_z:.3f} Hz vs 56.5 Hz "
        f"(rel err = {rel*100:.2f} %)"
    )


@pytest.mark.slow
@pytest.mark.benchmark
def test_fuchs_bar_volumetric(tmp_path):
    """Fuchs 2024 bar (3 cubes stuck end-to-end = 0.75 mm long, 0.25 mm
    square cross-section) inside the 4.5 x 3.5 x 4.7 mm elliptical cavity.

    The point-dipole baseline overshoots the measured 26.7 Hz z-mode by
    ~63 % (-> 43.6 Hz).  With volumetric source field the z-mode should
    move closer to 26.7 Hz; we require it within 20 %.
    """
    from sctrap.dipole import dipole_moment
    from sctrap.frequencies import find_equilibrium
    from sctrap.generators import elliptical_cavity
    from sctrap.mesh import load_msh
    from sctrap.modes import normal_modes
    from sctrap.particle import BarMagnetParticle

    mpath = tmp_path / "fuchs_bar.msh"
    elliptical_cavity(
        4.5e-3 / 2.0, 3.5e-3 / 2.0, 4.7e-3,
        mesh_size_sc=0.5e-3,
        output_file=str(mpath),
        verbose=False,
    )
    sct = load_msh(mpath)

    p = BarMagnetParticle(length=0.75e-3, width=0.25e-3, height=0.25e-3,
                          density=7500.0, B_remanence=1.4)
    m_vec = dipole_moment(p.moment, np.pi / 2.0, 0.0)
    eq_r = find_equilibrium(
        sct, m_vec, np.array([0.0, 0.0, 4.7e-3 * 0.5]),
        mass=p.mass, particle=p, element="P2",
    )
    nm = normal_modes(
        sct, eq_r, p.moment,
        eq_theta=np.pi / 2.0, eq_phi=0.0,
        mass=p.mass, inertia=(p.inertia_yy, p.inertia_zz),
        h_trans=20e-6, h_ang=1e-3,
        particle=p, element="P2",
    )
    f_z = _z_mode_freq(nm)
    rel = abs(f_z - 26.7) / 26.7
    assert rel < 0.20, (
        f"Fuchs bar z-mode out of range: f_z = {f_z:.3f} Hz vs 26.7 Hz "
        f"(rel err = {rel*100:.2f} %)"
    )
