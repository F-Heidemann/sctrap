"""Load gmsh `.msh` files into a scikit-fem tetrahedral mesh with
labelled SC (Neumann) and far-field (Dirichlet) facet sets.

Mesh contract
-------------
The user supplies a `.msh` containing two named *surface* physical groups —
one whose name contains ``"SC"`` (the superconductor surface) and one whose
name contains ``"FF"`` (the far-field truncation surface) — plus a single
volume group (any name) that is meshed with tetrahedra.

If named groups are missing, ``load_msh`` falls back to a geometric heuristic:
the largest distance from the origin defines the far-field boundary; all
other boundary facets are treated as the superconductor surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import warnings

import numpy as np
from skfem import MeshTet


# Substrings used to recognise physical-group names (case-insensitive).
_SC_KEYS = ("sc", "superconductor")
_FF_KEYS = ("ff", "farfield", "far_field", "far-field", "outer", "infinity")


@dataclass
class SCTrapMesh:
    """A scikit-fem tetrahedral mesh with SC / far-field facet labels."""

    mesh: MeshTet
    sc_facets: np.ndarray            # facet indices on SC surface
    ff_facets: np.ndarray            # facet indices on far-field
    sc_name: Optional[str] = None    # physical-group name used (if any)
    ff_name: Optional[str] = None
    used_heuristic: bool = False

    @property
    def points(self) -> np.ndarray:
        """(N, 3) array of vertex coordinates."""
        return self.mesh.p.T

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.mesh.p
        return p.min(axis=1), p.max(axis=1)

    def __repr__(self) -> str:
        nv = self.mesh.p.shape[1]
        nt = self.mesh.t.shape[1]
        nf_sc = len(self.sc_facets)
        nf_ff = len(self.ff_facets)
        tag = "heuristic" if self.used_heuristic else "named"
        return (f"<SCTrapMesh nverts={nv} ntets={nt} "
                f"sc_facets={nf_sc} ff_facets={nf_ff} ({tag})>")


def _match(name: str, keys: Sequence[str]) -> bool:
    n = name.lower()
    return any(k in n for k in keys)


def _heuristic_split(mesh: MeshTet) -> tuple[np.ndarray, np.ndarray]:
    """Classify boundary facets by distance of their centroids from the origin.

    The single farthest boundary component is taken as the far-field surface;
    all remaining boundary facets are SC.
    """
    bnd = mesh.boundary_facets()
    facets = mesh.facets[:, bnd]              # (3, N) vertex indices per facet
    centroids = mesh.p[:, facets].mean(axis=1)  # (3, N)
    radii = np.linalg.norm(centroids, axis=0)

    r_max = radii.max()
    if r_max <= 0:
        raise ValueError("Could not locate far-field facets via heuristic.")

    # Far-field = facets within 5 % of r_max from the outer envelope.
    ff_mask = radii > 0.95 * r_max
    sc_mask = ~ff_mask
    if not sc_mask.any() or not ff_mask.any():
        raise ValueError("Heuristic could not separate SC and far-field facets.")
    return bnd[sc_mask], bnd[ff_mask]


def load_msh(path: str | Path) -> SCTrapMesh:
    """Load a gmsh `.msh` file and identify SC / far-field facet sets.

    Parameters
    ----------
    path : path to the gmsh ``.msh`` file (any version meshio understands).

    Returns
    -------
    SCTrapMesh
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    mesh = MeshTet.load(str(path))

    sc_name = ff_name = None
    sc_facets = ff_facets = None
    used_heuristic = False

    boundaries = getattr(mesh, "boundaries", None) or {}
    for name in boundaries:
        if sc_name is None and _match(name, _SC_KEYS):
            sc_name = name
        elif ff_name is None and _match(name, _FF_KEYS):
            ff_name = name

    if sc_name is not None and ff_name is not None:
        # Open trap: explicit SC + FF groups.
        sc_facets = np.asarray(boundaries[sc_name], dtype=np.int64)
        ff_facets = np.asarray(boundaries[ff_name], dtype=np.int64)
    elif sc_name is not None and ff_name is None:
        # Fully enclosed (closed-trap) mode: only an SC group is provided.
        # The Laplace problem is pure-Neumann; the solver pins one interior
        # DOF to fix the gauge.
        sc_facets = np.asarray(boundaries[sc_name], dtype=np.int64)
        ff_facets = np.array([], dtype=np.int64)
    else:
        warnings.warn(
            "Mesh has no recognised SC / far-field physical groups; "
            "using geometric heuristic (outermost boundary = far field).",
            stacklevel=2,
        )
        sc_facets, ff_facets = _heuristic_split(mesh)
        used_heuristic = True

    return SCTrapMesh(
        mesh=mesh,
        sc_facets=sc_facets,
        ff_facets=ff_facets,
        sc_name=sc_name,
        ff_name=ff_name,
        used_heuristic=used_heuristic,
    )
