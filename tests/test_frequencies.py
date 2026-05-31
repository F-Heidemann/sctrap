"""Cheap unit test for the parabola fit used in frequency extraction."""

import numpy as np

from sctrap.frequencies import _parabola_freq


def test_parabola_recovers_known_frequency():
    """U(q) = (1/2) k q^2 with k = M (2*pi*f)^2 should yield f back."""
    M = 1.0e-6
    f_true = 12.34
    k = M * (2 * np.pi * f_true) ** 2
    qs = np.linspace(-1e-3, 1e-3, 11)
    Us = 0.5 * k * qs ** 2

    f = _parabola_freq(qs, Us, M)
    assert abs(f - f_true) / f_true < 1e-9
