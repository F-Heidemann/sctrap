"""Magnetic interaction energy and 1-D parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import G_GRAV, MAGNET_MASS
from .solver import solve_phi, B_induced_at
from .mesh import SCTrapMesh


def U_mag(sctmesh: SCTrapMesh, m: np.ndarray, r0: np.ndarray, **kwargs) -> float:
    """Magnetic trap (self-)energy U_mag = -1/2 m . B_induced(r0).

    The induced field is created *by* the dipole's own image currents, so the
    trap potential is the image self-energy with the standard factor 1/2
    (Jackson Sec. 2.2). Omitting it makes every stiffness 2x and every trap
    frequency a factor sqrt(2) too high.

    Solves the Laplace problem once and evaluates the induced field at r0.
    Extra kwargs are forwarded to `solve_phi` (e.g. ``element="P1"``).
    """
    phi  = solve_phi(sctmesh, m, r0, **kwargs)
    Bind = B_induced_at(phi, r0)
    return -0.5 * float(np.dot(np.asarray(m, dtype=float).reshape(3), Bind))


DEFAULT_GRAVITY = np.array([0.0, 0.0, -G_GRAV])


def gravity_potential(r0: np.ndarray, mass: float,
                      g_vec: np.ndarray = DEFAULT_GRAVITY) -> float:
    """Gravitational PE = -m * g_vec . r0.

    With the default `g_vec = (0, 0, -g)` this is `+m g z` (PE rises with
    height). For a trap tilted by angle theta about the y-axis use
    `g_vec = g * (sin theta, 0, -cos theta)`; the +x component of gravity
    pushes the equilibrium downhill.
    """
    g = np.asarray(g_vec, dtype=float).reshape(3)
    return -float(mass) * float(np.dot(g, np.asarray(r0, dtype=float).reshape(3)))


def U_total(sctmesh: SCTrapMesh,
            m: np.ndarray,
            r0: np.ndarray,
            mass: float = MAGNET_MASS,
            g_vec: np.ndarray = DEFAULT_GRAVITY,
            **kwargs) -> float:
    """U_mag + gravitational PE in `g_vec`."""
    return U_mag(sctmesh, m, r0, **kwargs) + gravity_potential(r0, mass, g_vec)


# ---------------------------------------------------------------------------
# Volumetric-particle energy path
# ---------------------------------------------------------------------------
#
# These functions are additive: they leave U_mag / U_total untouched.  They
# route the inducing source field through `particle.B_field(...)` (analytic
# closed form for the volumetric magnet, via magpylib) instead of a point
# dipole, then evaluate the interaction energy via `particle.interaction_energy`.

def U_mag_particle(sctmesh: SCTrapMesh,
                   particle,
                   r0: np.ndarray,
                   theta: float,
                   phi: float,
                   **kwargs) -> float:
    """Magnetic interaction energy for a finite-size magnet.

    `particle` must expose:
      - `B_field(points, r0, theta, phi) -> (N, 3) Tesla` (used to build the
        Neumann RHS on the SC facets), and
      - `interaction_energy(phi_sol, r0, theta, phi) -> float` (used to turn
        the solved potential into the energy).

    Extra kwargs go to `solve_phi`. `subtract_singularity=True` is rejected
    (the half-space image is only valid for a point dipole).
    """
    if kwargs.get("subtract_singularity", False):
        raise ValueError(
            "U_mag_particle does not support subtract_singularity=True "
            "(image-method subtraction is point-dipole only)."
        )
    kwargs.pop("subtract_singularity", None)

    r0 = np.asarray(r0, dtype=float).reshape(3)

    def B_source(points: np.ndarray) -> np.ndarray:
        return particle.B_field(points, r0, theta, phi)

    # `m` and `r0` are still required positional args of solve_phi for
    # bookkeeping/debugging; the actual source field comes from B_source.
    phi_sol = solve_phi(sctmesh, np.zeros(3), r0,
                        subtract_singularity=False,
                        B_source=B_source, **kwargs)
    return particle.interaction_energy(phi_sol, r0, theta, phi)


def U_total_particle(sctmesh: SCTrapMesh,
                     particle,
                     r0: np.ndarray,
                     theta: float,
                     phi: float,
                     g_vec: np.ndarray = DEFAULT_GRAVITY,
                     **kwargs) -> float:
    """U_mag_particle + gravitational PE for a finite-size magnet."""
    return (U_mag_particle(sctmesh, particle, r0, theta, phi, **kwargs)
            + gravity_potential(r0, particle.mass, g_vec))


def gravity_from_tilt(angle: float, axis: str = "y",
                      g: float = G_GRAV) -> np.ndarray:
    """Build a gravity vector for a trap tilted by `angle` rad about `axis`.

    `axis="y"` rotates the trap so that +x becomes downhill (default),
    `axis="x"` rotates so +y becomes downhill. `angle=0` gives -z gravity.
    """
    s, c = np.sin(angle), np.cos(angle)
    if axis == "y":
        return np.array([g * s, 0.0, -g * c])
    if axis == "x":
        return np.array([0.0, g * s, -g * c])
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


@dataclass
class Sweep1D:
    axis: str               # one of "x","y","z"
    coords: np.ndarray      # (N,)
    U_mag: np.ndarray       # (N,)
    U_grav: np.ndarray      # (N,)

    @property
    def U_total(self) -> np.ndarray:
        return self.U_mag + self.U_grav


def sweep_axis(
    sctmesh: SCTrapMesh,
    axis: str,
    coords: np.ndarray,
    m: np.ndarray,
    fixed: Optional[np.ndarray] = None,
    mass: float = MAGNET_MASS,
    verbose: bool = False,
    **kwargs,
) -> Sweep1D:
    """Compute U_mag along one Cartesian axis with the other two coordinates fixed.

    Parameters
    ----------
    axis   : "x", "y", or "z"
    coords : (N,) values of the swept coordinate [m]
    m      : (3,) dipole moment [A m^2]
    fixed  : (3,) baseline position; the swept axis component is overwritten.
             Defaults to the origin.
    """
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    base = np.asarray(fixed, dtype=float).reshape(3) if fixed is not None else np.zeros(3)
    coords = np.asarray(coords, dtype=float).ravel()

    Um = np.empty(coords.size, dtype=float)
    Ug = np.empty(coords.size, dtype=float)

    for i, c in enumerate(coords):
        r0 = base.copy()
        r0[idx] = c
        Um[i] = U_mag(sctmesh, m, r0, **kwargs)
        Ug[i] = mass * G_GRAV * float(r0[2])
        if verbose:
            print(f"  [{i+1}/{coords.size}] {axis}={c:.4e}  "
                  f"U_mag={Um[i]:.4e}  U_tot={Um[i]+Ug[i]:.4e}", flush=True)

    return Sweep1D(axis=axis, coords=coords, U_mag=Um, U_grav=Ug)
