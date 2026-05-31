"""Unit tests for the analytic dipole field."""

import numpy as np
import pytest

from sctrap import B_dipole, MU_0_OVER_4PI


def test_axis_field_along_moment():
    """Field on the moment axis: B = (mu0/4pi) * 2m / r^3 along m_hat."""
    m = np.array([0.0, 0.0, 1.0])
    r = np.array([0.0, 0.0, 0.5])
    B = B_dipole(r[None, :], m, np.zeros(3))[0]
    expected = MU_0_OVER_4PI * 2.0 * 1.0 / 0.5 ** 3 * np.array([0, 0, 1])
    assert B == pytest.approx(expected, rel=1e-12)


def test_equator_field_opposite_moment():
    """Field on the equatorial plane: B = -(mu0/4pi) * m / r^3."""
    m = np.array([0.0, 0.0, 1.0])
    r = np.array([1.0, 0.0, 0.0])
    B = B_dipole(r[None, :], m, np.zeros(3))[0]
    expected = -MU_0_OVER_4PI * 1.0 / 1.0 ** 3 * np.array([0, 0, 1])
    assert B == pytest.approx(expected, rel=1e-12)


def test_translation_invariance():
    """Translating dipole and field point by the same vector leaves B unchanged."""
    m = np.array([1.0, -2.0, 0.5])
    r0  = np.array([0.1, -0.3, 0.2])
    pts = np.random.RandomState(0).randn(20, 3) * 0.5
    delta = np.array([1.7, -0.4, 0.9])
    B0 = B_dipole(pts,         m, r0)
    B1 = B_dipole(pts + delta, m, r0 + delta)
    assert B0 == pytest.approx(B1, rel=1e-13, abs=1e-30)


def test_divergenceless_in_finite_difference():
    """div B = 0 anywhere away from the source."""
    m, r0 = np.array([0.7, 0.1, 1.3]), np.array([0.0, 0.0, 0.0])
    p = np.array([0.4, -0.2, 0.6])
    h = 1e-6
    div = 0.0
    for axis in range(3):
        e = np.zeros(3); e[axis] = h
        B_p = B_dipole((p + e)[None, :], m, r0)[0, axis]
        B_m = B_dipole((p - e)[None, :], m, r0)[0, axis]
        div += (B_p - B_m) / (2 * h)
    # divergence should be tiny (limited by FD truncation + roundoff)
    assert abs(div) < 1e-3


def test_curl_free_outside_source():
    """curl B = 0 in vacuum away from the source."""
    m, r0 = np.array([0.7, 0.1, 1.3]), np.array([0.0, 0.0, 0.0])
    p = np.array([0.4, -0.2, 0.6])
    h = 1e-6

    def Bv(q):
        return B_dipole(q[None, :], m, r0)[0]

    # curl_x = dBz/dy - dBy/dz, etc.
    def d(comp, axis):
        e = np.zeros(3); e[axis] = h
        return (Bv(p + e)[comp] - Bv(p - e)[comp]) / (2 * h)

    curl = np.array([
        d(2, 1) - d(1, 2),
        d(0, 2) - d(2, 0),
        d(1, 0) - d(0, 1),
    ])
    assert np.linalg.norm(curl) < 1e-3
