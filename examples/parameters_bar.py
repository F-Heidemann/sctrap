"""Bar-magnet preset.

Single rectangular NdFeB bar inside a closed elliptical Ta cavity, modelled
with magpylib's analytic cuboid field on the SC facets (i.e. *not* a point
dipole).  The defaults reproduce the geometric scale of the Fuchs 2024 bar
(0.75 mm long, 0.25 x 0.25 mm cross-section) in the same 4.5 x 3.5 x 4.7 mm
elliptical cavity.

Copy this file and edit to fit your own bar geometry.
"""
from __future__ import annotations
import numpy as np

# ----- cavity ---------------------------------------------------------------
CAVITY_A      = 4.5e-3 / 2.0     # ellipse semi-axis along x
CAVITY_B      = 3.5e-3 / 2.0     # ellipse semi-axis along y
CAVITY_HEIGHT = 4.7e-3

# ----- particle -------------------------------------------------------------
PARTICLE_KIND = "bar"

# body-x = length (= magnetisation axis), body-y = width, body-z = height
BAR_LENGTH  = 0.75e-3
BAR_WIDTH   = 0.25e-3
BAR_HEIGHT  = 0.25e-3
# Density chosen so the bar matches the Fuchs 2024 composite total mass
# (3 NdFeB cubes + glass bead + glue, rescaled to 0.43 mg in the paper):
#   m_total = 0.43e-6 kg ;  V = L*w*h = 4.6875e-11 m^3
#   density = m_total / V = 9173.3 kg/m^3
# Use the bare NdFeB density (7500) if you want the un-rescaled mass.
BAR_DENSITY = 9173.3
BAR_BREM    = 1.4             # remanent flux density [T]

# Sphere/composite knobs are unused but defined for import safety:
SPHERE_PARTICLE_RADIUS  = 27e-6
SPHERE_PARTICLE_DENSITY = 7430.0
SPHERE_PARTICLE_BREM    = 0.71
N_CUBES = 3; CUBE_EDGE = 0.25e-3; BEAD_RADIUS = 0.125e-3
BEAD_POSITION = np.array([0., 0., -(0.5*CUBE_EDGE + BEAD_RADIUS)])
B_REMANENCE = 1.4
DENSITY_NDFEB = 7500.0; DENSITY_BOROSILICATE = 2230.0
RESCALE_TO_MASS = None

# Other volumetric defaults (unused for kind="bar" but defined for safety):
CYL_RADIUS = 0.125e-3; CYL_LENGTH = 0.75e-3
CYL_DENSITY = 7500.0; CYL_BREM = 1.4
RING_OUTER_RADIUS = 0.30e-3; RING_INNER_RADIUS = 0.10e-3
RING_LENGTH = 0.75e-3; RING_DENSITY = 7500.0; RING_BREM = 1.4

# ----- orientation (equilibrium) -------------------------------------------
EQ_THETA = np.pi / 2.0           # bar lies in the cavity equatorial plane
EQ_PHI   = 0.0                   # magnetisation along +x

# ----- gravity tilt --------------------------------------------------------
TILT_ANGLE = 0.0
TILT_AXIS  = "y"

# ----- mesh ----------------------------------------------------------------
MESH_SIZE        = 0.5e-3
MESH_SIZE_NEAR   = None
MESH_REFINE_DIST = None

# ----- solver --------------------------------------------------------------
ELEMENT              = "P2"
# Image-subtraction is point-dipole only; the quickstart turns this off
# automatically when the particle is volumetric.
SUBTRACT_SINGULARITY = False

# Hessian stencil widths
H_TRANS = 20e-6
H_ANG   = 1e-3

EQ_GUESS = np.array([0.0, 0.0, CAVITY_HEIGHT * 0.5])

OUT_DIR = "bar_magnet"

# Optional reference for the diagnostic agreement print at the end of the run
COMPARE_F_HZ  = 26.7
COMPARE_LABEL = "Fuchs 2024 (z-mode, measured) — bar-only baseline"
