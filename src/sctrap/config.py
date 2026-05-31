"""Physical constants and default magnet parameters (SI units throughout)."""

import numpy as np

# Fundamental constants
MU_0 = 1.25663706212e-6          # Permeability of free space [H/m]
MU_0_OVER_4PI = 1e-7             # mu_0 / (4 pi)  [H/m]
G_GRAV = 9.80665                 # [m/s^2]

# Default magnet (0.25 mm cube, NdFeB-like)
MAGNET_REMANENCE = 1.4                              # B_rem [T]
MAGNET_VOLUME    = (0.25e-3) ** 3                   # [m^3]
MAGNET_MASS      = 0.43e-6                          # [kg]
MAGNET_RADIUS    = 0.25e-3                          # [m]

# Dipole moment magnitude  m = B_rem * V / mu_0   [A m^2]
MAGNET_MOMENT  = MAGNET_REMANENCE * MAGNET_VOLUME / MU_0

# Moment of inertia (solid sphere approximation)  I = (2/5) m R^2
MAGNET_INERTIA = (2.0 / 5.0) * MAGNET_MASS * MAGNET_RADIUS ** 2
