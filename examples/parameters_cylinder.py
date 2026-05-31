"""Cylindrical-magnet preset.

Single solid NdFeB cylindrical magnet, magnetised along its long axis, in a
closed elliptical Ta cavity.  Default cavity matches Fuchs 2024 (4.5 x 3.5 x
4.7 mm); default rod is 0.75 mm long with 0.125 mm radius.

Copy this file and edit to fit your own rod geometry.
"""
from __future__ import annotations
import numpy as np

# ----- cavity ---------------------------------------------------------------
CAVITY_A      = 4.5e-3 / 2.0
CAVITY_B      = 3.5e-3 / 2.0
CAVITY_HEIGHT = 4.7e-3

# ----- particle -------------------------------------------------------------
PARTICLE_KIND = "cylinder"

# body-x = cylinder axis (= magnetisation axis)
CYL_RADIUS  = 0.125e-3
CYL_LENGTH  = 0.75e-3
CYL_DENSITY = 7500.0
CYL_BREM    = 1.4

# Defaults below are unused for kind="cylinder" but defined for import safety.
SPHERE_PARTICLE_RADIUS  = 27e-6
SPHERE_PARTICLE_DENSITY = 7430.0
SPHERE_PARTICLE_BREM    = 0.71
N_CUBES = 3; CUBE_EDGE = 0.25e-3; BEAD_RADIUS = 0.125e-3
BEAD_POSITION = np.array([0., 0., -(0.5*CUBE_EDGE + BEAD_RADIUS)])
B_REMANENCE = 1.4
DENSITY_NDFEB = 7500.0; DENSITY_BOROSILICATE = 2230.0
RESCALE_TO_MASS = None
BAR_LENGTH = 0.75e-3; BAR_WIDTH = 0.25e-3; BAR_HEIGHT = 0.25e-3
BAR_DENSITY = 7500.0; BAR_BREM = 1.4
RING_OUTER_RADIUS = 0.30e-3; RING_INNER_RADIUS = 0.10e-3
RING_LENGTH = 0.75e-3; RING_DENSITY = 7500.0; RING_BREM = 1.4

# ----- orientation (equilibrium) -------------------------------------------
EQ_THETA = np.pi / 2.0           # cylinder lies in the cavity equatorial plane
EQ_PHI   = 0.0

# ----- gravity tilt --------------------------------------------------------
TILT_ANGLE = 0.0
TILT_AXIS  = "y"

# ----- mesh ----------------------------------------------------------------
MESH_SIZE        = 0.5e-3
MESH_SIZE_NEAR   = None
MESH_REFINE_DIST = None

# ----- solver --------------------------------------------------------------
ELEMENT              = "P2"
SUBTRACT_SINGULARITY = False

H_TRANS = 20e-6
H_ANG   = 1e-3

EQ_GUESS = np.array([0.0, 0.0, CAVITY_HEIGHT * 0.5])

OUT_DIR = "cylinder_magnet"

COMPARE_F_HZ  = None
COMPARE_LABEL = ""
