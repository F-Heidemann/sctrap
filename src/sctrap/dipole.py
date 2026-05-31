"""Magnetic dipole field, vectorised pure-numpy.

Formula (SI):
    B(r) = (mu_0 / 4 pi) * [ 3 (m . r_hat) r_hat - m ] / |r|^3
         = (mu_0 / 4 pi) * [ 3 (m . d) d - |d|^2 m ] / |d|^5

where d = r - r0 is the displacement from the dipole position to the field point.
"""

from __future__ import annotations

import numpy as np

from .config import MU_0_OVER_4PI


def B_dipole(points: np.ndarray, m: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """Magnetic field of a point dipole at every requested field point.

    Parameters
    ----------
    points : (N, 3) array of field point coordinates [m]
    m      : (3,)  magnetic moment vector [A m^2]
    r0     : (3,)  dipole position [m]

    Returns
    -------
    B : (N, 3) array, magnetic field at each row of `points` [T]
    """
    points = np.asarray(points, dtype=float)
    m  = np.asarray(m, dtype=float).reshape(3)
    r0 = np.asarray(r0, dtype=float).reshape(3)

    if points.ndim == 1:
        points = points[None, :]

    d  = points - r0                               # (N, 3)
    r2 = np.einsum("ij,ij->i", d, d)               # (N,)
    # Guard against evaluation exactly at the dipole location
    safe = r2 > 0
    r2_safe = np.where(safe, r2, 1.0)
    inv_r3 = np.where(safe, r2_safe ** (-1.5), 0.0)
    inv_r5 = np.where(safe, r2_safe ** (-2.5), 0.0)

    m_dot_d = d @ m                                # (N,)
    B = MU_0_OVER_4PI * (
        3.0 * (m_dot_d * inv_r5)[:, None] * d - inv_r3[:, None] * m[None, :]
    )
    return B


def dipole_moment(magnitude: float, theta: float, phi: float) -> np.ndarray:
    """Build a magnetic-moment vector from spherical angles.

    theta : polar angle from +z [rad]
    phi   : azimuth from +x   [rad]
    """
    return magnitude * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
