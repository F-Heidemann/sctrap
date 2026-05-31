"""Vinante et al. PRApplied 13, 064027 (2020) preset.

Closed cylindrical Pb cavity 4 mm dia x 4 mm deep. NdFeB microsphere
R = 27 um. Targets: z-mode 56.5 Hz, beta-mode 377 Hz.
"""
from __future__ import annotations
import numpy as np

# Geometry (circular -> CAVITY_A == CAVITY_B)
CAVITY_A      = 2.0e-3
CAVITY_B      = 2.0e-3
CAVITY_HEIGHT = 4.0e-3

# Particle
PARTICLE_KIND = "sphere"
SPHERE_PARTICLE_RADIUS  = 27e-6
SPHERE_PARTICLE_DENSITY = 7430.0
SPHERE_PARTICLE_BREM    = 0.71

# (composite knobs ignored, but defined for import safety)
N_CUBES = 3; CUBE_EDGE = 0.25e-3; BEAD_RADIUS = 0.125e-3
BEAD_POSITION = np.array([0., 0., -(0.5*CUBE_EDGE + BEAD_RADIUS)])
B_REMANENCE = 1.4
DENSITY_NDFEB = 7500.0; DENSITY_BOROSILICATE = 2230.0
RESCALE_TO_MASS = None

EQ_THETA = np.pi / 2.0
EQ_PHI   = 0.0

TILT_ANGLE = 0.0
TILT_AXIS  = "y"

MESH_SIZE        = 0.30e-3
MESH_SIZE_NEAR   = None
MESH_REFINE_DIST = None

ELEMENT              = "P2"
SUBTRACT_SINGULARITY = True

H_TRANS = 5e-6
H_ANG   = 5e-3

EQ_GUESS = np.array([0.0, 0.0, 0.30e-3])     # ~ image-method z0

OUT_DIR = "vinante_2020"

COMPARE_F_HZ  = 56.5
COMPARE_LABEL = "Vinante 2020 (z-mode)"
