"""sctrap — superconducting trap simulator."""

from .config import (
    MU_0,
    MU_0_OVER_4PI,
    G_GRAV,
    MAGNET_REMANENCE,
    MAGNET_VOLUME,
    MAGNET_MASS,
    MAGNET_RADIUS,
    MAGNET_MOMENT,
    MAGNET_INERTIA,
)
from .dipole import B_dipole, dipole_moment
from .mesh import load_msh, SCTrapMesh
from .solver import solve_phi, B_induced_at
from .potential import U_mag, sweep_axis
from .frequencies import find_equilibrium, frequencies_at, TrapFrequencies
from .modes import normal_modes, NormalModes
from .uncertainty import stencil_richardson, FrequencyWithUncertainty

__version__ = "0.1.0"

__all__ = [
    "MU_0", "MU_0_OVER_4PI", "G_GRAV",
    "MAGNET_REMANENCE", "MAGNET_VOLUME", "MAGNET_MASS",
    "MAGNET_RADIUS", "MAGNET_MOMENT", "MAGNET_INERTIA",
    "B_dipole", "dipole_moment",
    "load_msh", "SCTrapMesh",
    "solve_phi", "B_induced_at",
    "U_mag", "sweep_axis",
    "find_equilibrium", "frequencies_at", "TrapFrequencies",
    "normal_modes", "NormalModes",
    "stencil_richardson", "FrequencyWithUncertainty",
]
