"""Closed-form image-dipole field for a single SC half-space (singularity
subtraction support for the Neumann scalar-potential problem).

Idea
----
The dipole field `B_dip` has a |r-r0|^-3 singularity on the SC facet directly
under the dipole. Even with high-order surface quadrature, the FEM cannot
integrate `-n · B_dip` accurately across that one facet — every nearby
collocation point dominates the integral.

Trick: split

    Phi  =  Phi_image  +  Phi_residual

where `Phi_image` is the closed-form scalar potential (i.e. analytically
integrable B-field) of the image dipole obtained by mirroring `r0, m` through
the local tangent plane at the closest SC point. By construction:

    n_tp · grad(Phi_image)  =  -n_tp · B_dip   on the tangent plane,

so the *residual* RHS

    -n · (B_dip + B_image)

vanishes on the tangent plane and is smooth on neighbouring SC facets that
are nearly parallel to it. The residual problem is therefore quadrature-
friendly. The induced field on the dipole is reconstructed as

    B_induced(r0) = grad(Phi_residual)(r0) + B_image(r0).

This module provides the geometric helpers; `solver.solve_phi` wires it in.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .dipole import B_dipole
from .mesh import SCTrapMesh


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def reflect_point(point: np.ndarray, plane_point: np.ndarray,
                  plane_normal: np.ndarray) -> np.ndarray:
    """Mirror `point` through the plane (plane_point, plane_normal)."""
    n = plane_normal
    d = float(np.dot(point - plane_point, n))
    return point - 2.0 * d * n


def reflect_moment(m: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """SC mirror image of a magnetic moment.

    A reflection through a plane with unit normal n flips the moment's
    component along n and keeps the tangential components — equivalently

        m_image = m - 2 (m . n) n

    This is the rule that makes B . n = 0 on the SC plane.
    """
    n = plane_normal
    return m - 2.0 * float(np.dot(m, n)) * n


def find_nearest_sc_facet(
    sctmesh: SCTrapMesh, r0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (centroid, outward_normal_into_sc, facet_index) of the SC facet
    whose centroid is closest to `r0`.

    The returned normal points *from r0 into the SC* (i.e. the air-domain
    outward normal at that facet) so that mirroring `r0` through the plane
    places the image on the SC side.
    """
    mesh = sctmesh.mesh
    sc_f = sctmesh.sc_facets
    if sc_f is None or len(sc_f) == 0:
        raise ValueError("Mesh has no SC facets to project onto.")

    facet_verts = mesh.facets[:, sc_f]                     # (3, M)
    pts = mesh.p[:, facet_verts]                           # (3, 3, M)
    centroids = pts.mean(axis=1)                           # (3, M)

    r0 = np.asarray(r0, dtype=float).reshape(3)
    d2 = ((centroids - r0[:, None]) ** 2).sum(axis=0)
    k = int(np.argmin(d2))

    centroid = centroids[:, k].copy()
    v0, v1, v2 = pts[:, 0, k], pts[:, 1, k], pts[:, 2, k]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 0.0:
        raise ValueError(f"Degenerate SC facet at index {sc_f[k]}.")
    n = n / n_norm

    # Orient the normal to point from r0 toward the SC (so the image lands on
    # the far side of the plane, inside the SC).
    if float(np.dot(n, centroid - r0)) < 0.0:
        n = -n

    return centroid, n, int(sc_f[k])


# ---------------------------------------------------------------------------
# Image-dipole field
# ---------------------------------------------------------------------------

def B_halfspace_image(
    points: np.ndarray,
    m: np.ndarray,
    r0: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """B-field of the SC image of (m, r0) reflected through the tangent plane.

    Parameters
    ----------
    points       : (N, 3) field points
    m            : (3,)  source moment
    r0           : (3,)  source position
    plane_point  : (3,)  any point on the mirror plane
    plane_normal : (3,)  unit normal of the mirror plane

    Returns
    -------
    B : (N, 3) array — image-dipole field at each field point [T]
    """
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    n = n / float(np.linalg.norm(n))
    r0_im = reflect_point(np.asarray(r0,  dtype=float).reshape(3),
                          np.asarray(plane_point, dtype=float).reshape(3), n)
    m_im  = reflect_moment(np.asarray(m, dtype=float).reshape(3), n)
    return B_dipole(points, m_im, r0_im)
