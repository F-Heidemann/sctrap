"""Fuchs et al. Sci. Adv. 10, eadk2949 (2024) preset.

Closed elliptical Ta cavity 4.5 x 3.5 mm cross-section, 4.7 mm tall.
Composite particle: 3x 250-um NdFeB cubes + 250-um glass bead, total 0.43 mg.
Target z-mode: 26.7 Hz measured, 27 Hz analytic estimate.
"""
from __future__ import annotations
import numpy as np

# Geometry
CAVITY_A      = 4.5e-3 / 2.0
CAVITY_B      = 3.5e-3 / 2.0
CAVITY_HEIGHT = 4.7e-3

# Particle
PARTICLE_KIND = "composite"

# (sphere knobs ignored, but defined)
SPHERE_PARTICLE_RADIUS  = 27e-6
SPHERE_PARTICLE_DENSITY = 7430.0
SPHERE_PARTICLE_BREM    = 0.71

N_CUBES         = 3
CUBE_EDGE       = 0.25e-3
BEAD_RADIUS     = 0.125e-3
BEAD_POSITION   = np.array([0., 0., -(0.5 * CUBE_EDGE + BEAD_RADIUS)])

B_REMANENCE          = 1.4
DENSITY_NDFEB        = 7500.0
DENSITY_BOROSILICATE = 2230.0
RESCALE_TO_MASS      = 0.43e-6

# Orientation is the *initial guess* only. With FIND_ORIENTATION = True the
# quickstart jointly relaxes position AND orientation (find_equilibrium_5d),
# so it discovers the preferred orientation rather than assuming it. Starting
# from the long axis (EQ_PHI = 0), the solver settles onto the SHORT (y) axis
# -- the orientation energy minimum, the short-axis-preference result of the
# cuboidal-trap analysis -- and the Hessian there has all five modes stable.
# Set FIND_ORIENTATION = False to instead pin (EQ_THETA, EQ_PHI).
EQ_THETA        = np.pi / 2.0
EQ_PHI          = 0.0
FIND_ORIENTATION = True

TILT_ANGLE = 0.0
TILT_AXIS  = "y"

# Finer mesh than the 0.5 mm default: the z-stiffness is a small curvature on
# a large baseline energy, so it benefits from resolution. 0.3 mm is a good
# balance (~minutes per run); drop toward 0.2 mm for a convergence check.
MESH_SIZE        = 0.3e-3
MESH_SIZE_NEAR   = None
MESH_REFINE_DIST = None

ELEMENT              = "P2"
SUBTRACT_SINGULARITY = True

H_TRANS = 20e-6
H_ANG   = 1e-3

EQ_GUESS = np.array([0.0, 0.0, CAVITY_HEIGHT * 0.5])

OUT_DIR = "fuchs_2024"

COMPARE_F_HZ  = 26.7
COMPARE_LABEL = "Fuchs 2024 (z-mode, measured)"
