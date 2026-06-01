"""Composite-particle geometry and inertia.

The Fuchs et al. 2024 trap holds a small composite particle:

    [cube][cube][cube]      <- N NdFeB cubes glued along their easy axis
         ●                  <- a borosilicate glass sphere glued to the
                                long underside of the bar (= -z face)

Body-frame conventions (used throughout):

    x  =  bar (= dipole-moment) axis. Cubes lie centred on this axis.
    y  =  one transverse axis.
    z  =  other transverse axis. The glass bead sits at z < 0 by default
          (Fuchs geometry: bead glued underneath the bar).

The particle is treated as a rigid body. We compute the full 3x3 inertia
tensor in the body frame, about the centre of mass; for an asymmetric
composite (bar + bead-below) the principal moments around the two
transverse axes (y, z) are different, which matters for libration modes.

This module replaces the old "thin-bar / solid-sphere" guesses that lived
inline in scripts/fuchs_2024.py and config.py.

Usage
-----
>>> p = composite_particle()                 # Fuchs defaults
>>> print(p)
>>> p.mass, p.inertia_yy, p.inertia_zz, p.moment

Sanity check (Fuchs defaults at NdFeB density 7500 / borosilicate 2230):

    3 cubes 250 um             -> 0.352 mg
    1 sphere 250 um diameter   -> 0.018 mg
    total                      ~ 0.370 mg

Paper says 0.43 mg (the published number includes glue / coating); use
`from_total_mass(0.43e-6)` to rescale densities so the total matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import MU_0


DENSITY_NDFEB        = 7500.0
DENSITY_BOROSILICATE = 2230.0


# ---------------------------------------------------------------------------
# Inertia primitives
# ---------------------------------------------------------------------------

def _inertia_cube_self(mass: float, edge: float) -> np.ndarray:
    """3x3 inertia tensor of a uniform cube about axes through its centre,
    aligned with the cube edges. All three principal moments equal m a^2 / 6."""
    I = np.eye(3) * (mass * edge ** 2 / 6.0)
    return I


def _inertia_sphere_self(mass: float, radius: float) -> np.ndarray:
    return np.eye(3) * (0.4 * mass * radius ** 2)


def _parallel_axis_shift(mass: float, r: np.ndarray) -> np.ndarray:
    """Steiner term: I_shifted = I_self + m * (|r|^2 I - r r^T)."""
    r = np.asarray(r, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    return mass * (r2 * np.eye(3) - np.outer(r, r))


# ---------------------------------------------------------------------------
# Particle dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompositeParticle:
    """Rigid composite of N cubic magnets in a row plus 1 spherical bead.

    Geometry is specified relative to the cube-stack centre.  Cube i sits at
    x_i = (i - (N-1)/2) * cube_edge along +x;  the bead sits at
    `sphere_position` (a 3-vector).  Default Fuchs geometry has the bead
    tangent to the underside of the bar:
        sphere_position = (0, 0, -(0.5 * cube_edge + sphere_radius)).
    """

    cube_edge:        float
    n_cubes:          int
    sphere_radius:    float
    sphere_position:  np.ndarray         # (3,) bead centre rel. to cube-stack centre

    magnet_density:   float
    sphere_density:   float
    B_remanence:      float

    # Derived fields
    cube_mass:        float = 0.0
    sphere_mass:      float = 0.0
    mass:             float = 0.0
    com:              np.ndarray = field(default_factory=lambda: np.zeros(3))
    I_tensor:         np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    I_principal:      np.ndarray = field(default_factory=lambda: np.zeros(3))
    I_axes:           np.ndarray = field(default_factory=lambda: np.eye(3))
    moment:           float = 0.0
    bar_length:       float = 0.0

    def __post_init__(self) -> None:
        a   = self.cube_edge
        N   = int(self.n_cubes)
        R   = float(self.sphere_radius)
        rsp = np.asarray(self.sphere_position, dtype=float).reshape(3)
        self.sphere_position = rsp

        # ---- Sphere-cube interpenetration check ----------------------
        # For each cube, find the closest point ON the cube to the sphere
        # centre, then check |closest - sphere_centre| >= sphere_radius.
        if R > 0:
            x_cubes_check = (np.arange(N) - 0.5 * (N - 1)) * a
            half = 0.5 * a
            for i, xi in enumerate(x_cubes_check):
                cube_min = np.array([xi - half, -half, -half])
                cube_max = np.array([xi + half, +half, +half])
                closest = np.minimum(np.maximum(rsp, cube_min), cube_max)
                gap = float(np.linalg.norm(rsp - closest)) - R
                if gap < -1e-12:
                    raise ValueError(
                        f"Sphere overlaps cube #{i} (centre at x={xi*1e3:.3f} mm). "
                        f"Penetration depth = {-gap*1e6:.2f} um. "
                        "Adjust `sphere_position` or `sphere_radius`."
                    )

        Vc = a ** 3
        Vs = (4.0 / 3.0) * np.pi * R ** 3 if R > 0 else 0.0
        mc = self.magnet_density * Vc
        ms = self.sphere_density * Vs

        # Cube positions
        x_cubes = (np.arange(N) - 0.5 * (N - 1)) * a
        cube_positions = np.stack(
            [x_cubes, np.zeros(N), np.zeros(N)], axis=1)        # (N, 3)

        # Centre of mass
        M_total = N * mc + ms
        com = (mc * cube_positions.sum(axis=0) + ms * rsp) / M_total

        # Inertia tensor about origin (cube-stack centre)
        I_origin = np.zeros((3, 3))
        for r in cube_positions:
            I_origin += _inertia_cube_self(mc, a) + _parallel_axis_shift(mc, r)
        if R > 0:
            I_origin += _inertia_sphere_self(ms, R) + _parallel_axis_shift(ms, rsp)

        # Shift to COM via parallel-axis theorem (subtract M_total * com_outer)
        I_com = I_origin - _parallel_axis_shift(M_total, com)
        I_com = 0.5 * (I_com + I_com.T)             # symmetrise

        # Principal moments
        evals, evecs = np.linalg.eigh(I_com)

        moment = N * self.B_remanence * Vc / MU_0

        # Populate
        self.cube_mass   = mc
        self.sphere_mass = ms
        self.mass        = M_total
        self.com         = com
        self.I_tensor    = I_com
        self.I_principal = evals
        self.I_axes      = evecs
        self.moment      = moment
        self.bar_length  = N * a

    # -------- Convenience principal moments (body axes) ------------------
    @property
    def inertia_xx(self) -> float:        # spin about bar (= dipole) axis
        return float(self.I_tensor[0, 0])

    @property
    def inertia_yy(self) -> float:        # libration about y (tilts in xz plane)
        return float(self.I_tensor[1, 1])

    @property
    def inertia_zz(self) -> float:        # libration about z (yaw in xy plane)
        return float(self.I_tensor[2, 2])

    @property
    def inertia_perp_max(self) -> float:
        """Worst-case transverse moment — the larger of yy and zz.
        Useful when downstream code only takes a single scalar."""
        return float(max(self.I_tensor[1, 1], self.I_tensor[2, 2]))

    # -------- Factories ---------------------------------------------------

    @classmethod
    def fuchs_default(cls) -> "CompositeParticle":
        a = 0.25e-3
        R = 0.5 * 0.25e-3
        # Bead glued tangent to underside of bar -> below in z.
        sphere_pos = np.array([0.0, 0.0, -(0.5 * a + R)])
        return cls(
            cube_edge       = a,
            n_cubes         = 3,
            sphere_radius   = R,
            sphere_position = sphere_pos,
            magnet_density  = DENSITY_NDFEB,
            sphere_density  = DENSITY_BOROSILICATE,
            B_remanence     = 1.4,
        )

    @classmethod
    def from_total_mass(cls, total_mass: float, **kwargs) -> "CompositeParticle":
        """Build with given densities, then rescale both densities uniformly
        so the total mass matches ``total_mass``. Inertia and COM ratios are
        preserved (since both densities scale by the same factor)."""
        defaults = cls._defaults_dict()
        defaults.update(kwargs)
        if defaults["sphere_position"] is None:
            defaults["sphere_position"] = cls._default_sphere_pos(
                defaults["cube_edge"], defaults["sphere_radius"])
        p = cls(**defaults)
        scale = total_mass / p.mass
        defaults["magnet_density"] *= scale
        defaults["sphere_density"] *= scale
        return cls(**defaults)

    @staticmethod
    def _defaults_dict() -> dict:
        return dict(
            cube_edge       = 0.25e-3,
            n_cubes         = 3,
            sphere_radius   = 0.125e-3,
            sphere_position = None,
            magnet_density  = DENSITY_NDFEB,
            sphere_density  = DENSITY_BOROSILICATE,
            B_remanence     = 1.4,
        )

    @staticmethod
    def _default_sphere_pos(cube_edge: float, sphere_radius: float) -> np.ndarray:
        return np.array([0.0, 0.0, -(0.5 * cube_edge + sphere_radius)])

    # -------- I/O ---------------------------------------------------------

    def __str__(self) -> str:
        x_com_um = self.com * 1e6
        return "\n".join([
            "CompositeParticle",
            f"  geometry           : {self.n_cubes} cubes @ {self.cube_edge*1e3:.3f} mm "
            f"+ sphere R={self.sphere_radius*1e3:.3f} mm",
            f"  sphere position    : ({self.sphere_position[0]*1e3:+.3f}, "
            f"{self.sphere_position[1]*1e3:+.3f}, "
            f"{self.sphere_position[2]*1e3:+.3f}) mm",
            f"  bar length (=N*a)  : {self.bar_length*1e3:.3f} mm",
            f"  cube mass (each)   : {self.cube_mass*1e6:.4f} mg",
            f"  sphere mass        : {self.sphere_mass*1e6:.4f} mg",
            f"  total mass         : {self.mass*1e6:.4f} mg",
            f"  COM (rel. bar ctr) : ({x_com_um[0]:+.2f}, {x_com_um[1]:+.2f}, "
            f"{x_com_um[2]:+.2f}) um",
            f"  I_xx (spin/bar)    : {self.inertia_xx:.4e} kg m^2",
            f"  I_yy (theta-libr.) : {self.inertia_yy:.4e} kg m^2",
            f"  I_zz (phi-libr.)   : {self.inertia_zz:.4e} kg m^2",
            f"  principal moments  : {self.I_principal}",
            f"  dipole moment |m|  : {self.moment:.4e} A m^2",
        ])

    def as_dict(self) -> dict:
        return {
            "cube_edge_m":         self.cube_edge,
            "n_cubes":             self.n_cubes,
            "sphere_radius_m":     self.sphere_radius,
            "sphere_position_m":   self.sphere_position.tolist(),
            "magnet_density":      self.magnet_density,
            "sphere_density":      self.sphere_density,
            "B_remanence_T":       self.B_remanence,
            "mass_kg":             self.mass,
            "cube_mass_kg":        self.cube_mass,
            "sphere_mass_kg":      self.sphere_mass,
            "com_m":               self.com.tolist(),
            "I_tensor_kgm2":       self.I_tensor.tolist(),
            "I_principal_kgm2":    self.I_principal.tolist(),
            "moment_Am2":          self.moment,
            "bar_length_m":        self.bar_length,
        }


@dataclass
class SphericalParticle:
    """Homogeneous magnetised sphere (Vinante 2020 / image-method benchmark).

    The whole particle is a single ferromagnetic sphere with uniform
    magnetisation. The dipole moment vector lies along the body-x axis
    (chart: theta=pi/2, phi=0 puts m along world-x at equilibrium).
    """

    radius:        float          # [m]
    density:       float          # [kg/m^3]
    B_remanence:   float          # [T]   (paper convention: mu_0 M)

    # Derived
    volume:        float = 0.0
    mass:          float = 0.0
    moment:        float = 0.0
    inertia:       float = 0.0    # solid sphere about any axis through centre
    com:           np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        R = self.radius
        V = (4.0 / 3.0) * np.pi * R ** 3
        m = self.density * V
        self.volume  = V
        self.mass    = m
        self.moment  = self.B_remanence * V / MU_0
        self.inertia = 0.4 * m * R ** 2          # I = (2/5) m R^2

    # Same accessors as CompositeParticle for API compatibility ------------
    @property
    def inertia_xx(self) -> float: return self.inertia
    @property
    def inertia_yy(self) -> float: return self.inertia
    @property
    def inertia_zz(self) -> float: return self.inertia

    def __str__(self) -> str:
        return "\n".join([
            "SphericalParticle",
            f"  radius            : {self.radius*1e6:.2f} um",
            f"  density           : {self.density:.0f} kg/m^3",
            f"  mu_0 M (Brem)     : {self.B_remanence:.3f} T",
            f"  mass              : {self.mass*1e9:.4f} ug   ({self.mass:.3e} kg)",
            f"  inertia (any axis): {self.inertia:.4e} kg m^2",
            f"  dipole moment |m| : {self.moment:.4e} A m^2",
        ])

    def as_dict(self) -> dict:
        return {
            "kind":           "SphericalParticle",
            "radius_m":       self.radius,
            "density_kgm3":   self.density,
            "B_remanence_T":  self.B_remanence,
            "mass_kg":        self.mass,
            "inertia_kgm2":   self.inertia,
            "moment_Am2":     self.moment,
        }


def composite_particle(**kwargs) -> CompositeParticle:
    """Convenience wrapper. With no args, returns the Fuchs default."""
    if not kwargs:
        return CompositeParticle.fuchs_default()
    defaults = CompositeParticle._defaults_dict()
    defaults.update(kwargs)
    if defaults["sphere_position"] is None:
        defaults["sphere_position"] = CompositeParticle._default_sphere_pos(
            defaults["cube_edge"], defaults["sphere_radius"])
    return CompositeParticle(**defaults)


# ---------------------------------------------------------------------------
# Volumetric (finite-size) magnet particles
# ---------------------------------------------------------------------------
#
# These particle types expose an analytic source field B_field(...) computed
# by magpylib (closed-form integration over the magnetised volume).  The
# Laplace solver consumes that field through the optional `B_source` kwarg
# in `solve_phi`, replacing the point-dipole field on the SC facets.
#
# Convention (matches CompositeParticle / SphericalParticle):
#     body-x  =  magnetisation (dipole) axis at theta=pi/2, phi=0.
#     world-frame unit moment vector  hat_m = (sin t cos p, sin t sin p, cos t).
#
# Each class derives its scalar inertia moments analytically, so callers can
# read off inertia_xx (spin about magnetisation axis), inertia_yy, inertia_zz
# from the same API as CompositeParticle.

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector cannot be normalised")
    return v / n


def _rotation_aligning(src: np.ndarray, dst: np.ndarray):
    """scipy.spatial.transform.Rotation that rotates `src` to `dst`.

    Handles the antiparallel edge case (where the naive Rodrigues cross
    product vanishes) by picking a fixed perpendicular axis.
    """
    from scipy.spatial.transform import Rotation

    s = _unit(src)
    d = _unit(dst)
    c = float(np.dot(s, d))
    if c > 1.0 - 1e-12:
        return Rotation.identity()
    if c < -1.0 + 1e-12:
        # Antiparallel: choose any axis perpendicular to s.
        # If s is along +x or -x, rotate about y; otherwise about x.
        if abs(s[0]) < 0.9:
            axis = np.cross(s, np.array([1.0, 0.0, 0.0]))
        else:
            axis = np.cross(s, np.array([0.0, 1.0, 0.0]))
        axis = _unit(axis)
        return Rotation.from_rotvec(axis * np.pi)
    axis = np.cross(s, d)
    axis = _unit(axis)
    angle = float(np.arccos(np.clip(c, -1.0, 1.0)))
    return Rotation.from_rotvec(axis * angle)


def _hat_m(theta: float, phi: float) -> np.ndarray:
    """Unit magnetisation direction in the world frame.

    Mirrors `dipole_moment(1.0, theta, phi)` from `dipole.py` exactly so the
    source field and the energy evaluation never disagree on orientation.
    """
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], dtype=float)


def _interaction_energy_from_phi(moment: float,
                                 phi_sol,
                                 r0: np.ndarray,
                                 theta: float,
                                 phi: float) -> float:
    """U = -1/2 m . B_induced(r0); shared by all volumetric particles.

    The 1/2 is the image self-energy factor (Jackson Sec. 2.2); see `U_mag`.
    """
    from .dipole import dipole_moment
    from .solver import B_induced_at
    m_vec = dipole_moment(moment, theta, phi)
    B_ind = B_induced_at(phi_sol, np.asarray(r0, dtype=float).reshape(3))
    return -0.5 * float(np.dot(m_vec, B_ind))


@dataclass
class BarMagnetParticle:
    """Solid rectangular bar magnet.

    Body-frame geometry:
        body-x = length (= magnetisation/dipole axis)
        body-y = width
        body-z = height

    The COM sits at the centre of the bar.
    """

    length:        float          # along body-x (magnetisation axis) [m]
    width:         float          # along body-y                       [m]
    height:        float          # along body-z                       [m]
    density:       float          # [kg/m^3]
    B_remanence:   float          # [T]

    # Derived
    volume:        float = 0.0
    mass:          float = 0.0
    moment:        float = 0.0
    com:           np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        L, w, h = float(self.length), float(self.width), float(self.height)
        if min(L, w, h) <= 0:
            raise ValueError("length, width and height must be > 0")
        V = L * w * h
        m = self.density * V
        self.volume = V
        self.mass   = m
        self.moment = self.B_remanence * V / MU_0

    # --- Inertia (uniform cuboid about COM, body axes) -------------------
    @property
    def inertia_xx(self) -> float:
        return self.mass * (self.width ** 2 + self.height ** 2) / 12.0
    @property
    def inertia_yy(self) -> float:
        return self.mass * (self.length ** 2 + self.height ** 2) / 12.0
    @property
    def inertia_zz(self) -> float:
        return self.mass * (self.length ** 2 + self.width ** 2) / 12.0

    # --- magpylib source --------------------------------------------------
    def _magpy_source(self, r0: np.ndarray, theta: float, phi: float):
        """Build a configured magpylib Cuboid for the current pose."""
        import magpylib as magpy
        # Magnetisation in *local* frame is along +x; the orientation rotates
        # body-x onto hat_m.  Magnitude = B_rem / mu_0  [A/m].
        M = self.B_remanence / MU_0
        rot = _rotation_aligning(np.array([1.0, 0.0, 0.0]),
                                 _hat_m(theta, phi))
        src = magpy.magnet.Cuboid(
            magnetization=(M, 0.0, 0.0),
            dimension=(self.length, self.width, self.height),
        )
        src.orientation = rot
        src.position = np.asarray(r0, dtype=float).reshape(3)
        return src

    def B_field(self, points: np.ndarray, r0: np.ndarray,
                theta: float, phi: float) -> np.ndarray:
        """Source field B(points) in Tesla, with COM at r0, body axes set
        by (theta, phi). Vectorised over an (N, 3) array of observers."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        src = self._magpy_source(r0, theta, phi)
        return src.getB(pts)

    def interaction_energy(self, phi_sol, r0: np.ndarray,
                           theta: float, phi: float) -> float:
        return _interaction_energy_from_phi(self.moment, phi_sol,
                                            r0, theta, phi)

    def __str__(self) -> str:
        return "\n".join([
            "BarMagnetParticle",
            f"  L x w x h         : {self.length*1e3:.3f} x "
            f"{self.width*1e3:.3f} x {self.height*1e3:.3f} mm",
            f"  density           : {self.density:.0f} kg/m^3",
            f"  mu_0 M (Brem)     : {self.B_remanence:.3f} T",
            f"  mass              : {self.mass*1e9:.4f} ug",
            f"  I_xx (spin/bar)   : {self.inertia_xx:.4e} kg m^2",
            f"  I_yy              : {self.inertia_yy:.4e} kg m^2",
            f"  I_zz              : {self.inertia_zz:.4e} kg m^2",
            f"  dipole moment |m| : {self.moment:.4e} A m^2",
        ])

    def as_dict(self) -> dict:
        return {
            "kind":           "BarMagnetParticle",
            "length_m":       self.length,
            "width_m":        self.width,
            "height_m":       self.height,
            "density_kgm3":   self.density,
            "B_remanence_T":  self.B_remanence,
            "mass_kg":        self.mass,
            "moment_Am2":     self.moment,
            "inertia_xx_kgm2": self.inertia_xx,
            "inertia_yy_kgm2": self.inertia_yy,
            "inertia_zz_kgm2": self.inertia_zz,
        }


@dataclass
class CylinderMagnetParticle:
    """Solid cylindrical magnet, magnetised along its long axis.

    Body-frame geometry:
        body-x = cylinder axis (= magnetisation/dipole axis), length `length`
        body-y, body-z = transverse plane, radius `radius`
    """

    radius:        float          # [m]
    length:        float          # along body-x                       [m]
    density:       float          # [kg/m^3]
    B_remanence:   float          # [T]

    # Derived
    volume:        float = 0.0
    mass:          float = 0.0
    moment:        float = 0.0
    com:           np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        R, L = float(self.radius), float(self.length)
        if R <= 0 or L <= 0:
            raise ValueError("radius and length must be > 0")
        V = np.pi * R ** 2 * L
        m = self.density * V
        self.volume = V
        self.mass   = m
        self.moment = self.B_remanence * V / MU_0

    @property
    def inertia_xx(self) -> float:        # spin about cylinder axis
        return 0.5 * self.mass * self.radius ** 2
    @property
    def inertia_yy(self) -> float:        # transverse
        return self.mass * (3.0 * self.radius ** 2 + self.length ** 2) / 12.0
    @property
    def inertia_zz(self) -> float:
        return self.inertia_yy            # axisymmetric about body-x

    def _magpy_source(self, r0: np.ndarray, theta: float, phi: float):
        import magpylib as magpy
        # magpylib Cylinder's geometric axis is local +z, magnetisation also
        # in local frame; rotate local-z onto hat_m and put magnetisation
        # along that same axis.
        M = self.B_remanence / MU_0
        rot = _rotation_aligning(np.array([0.0, 0.0, 1.0]),
                                 _hat_m(theta, phi))
        src = magpy.magnet.Cylinder(
            magnetization=(0.0, 0.0, M),
            dimension=(2.0 * self.radius, self.length),
        )
        src.orientation = rot
        src.position = np.asarray(r0, dtype=float).reshape(3)
        return src

    def B_field(self, points: np.ndarray, r0: np.ndarray,
                theta: float, phi: float) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        src = self._magpy_source(r0, theta, phi)
        return src.getB(pts)

    def interaction_energy(self, phi_sol, r0: np.ndarray,
                           theta: float, phi: float) -> float:
        return _interaction_energy_from_phi(self.moment, phi_sol,
                                            r0, theta, phi)

    def __str__(self) -> str:
        return "\n".join([
            "CylinderMagnetParticle",
            f"  R x L             : {self.radius*1e3:.3f} x {self.length*1e3:.3f} mm",
            f"  density           : {self.density:.0f} kg/m^3",
            f"  mu_0 M (Brem)     : {self.B_remanence:.3f} T",
            f"  mass              : {self.mass*1e9:.4f} ug",
            f"  I_xx (spin/bar)   : {self.inertia_xx:.4e} kg m^2",
            f"  I_yy = I_zz       : {self.inertia_yy:.4e} kg m^2",
            f"  dipole moment |m| : {self.moment:.4e} A m^2",
        ])

    def as_dict(self) -> dict:
        return {
            "kind":           "CylinderMagnetParticle",
            "radius_m":       self.radius,
            "length_m":       self.length,
            "density_kgm3":   self.density,
            "B_remanence_T":  self.B_remanence,
            "mass_kg":        self.mass,
            "moment_Am2":     self.moment,
            "inertia_xx_kgm2": self.inertia_xx,
            "inertia_yy_kgm2": self.inertia_yy,
            "inertia_zz_kgm2": self.inertia_zz,
        }


@dataclass
class RingMagnetParticle:
    """Hollow cylindrical (ring) magnet, magnetised along its long axis.

    Body-frame geometry:
        body-x = ring axis (= magnetisation/dipole axis), length `length`
        outer/inner radii in the transverse plane
    """

    outer_radius:  float          # [m]
    inner_radius:  float          # [m]   (must be < outer_radius)
    length:        float          # along body-x [m]
    density:       float          # [kg/m^3]
    B_remanence:   float          # [T]

    # Derived
    volume:        float = 0.0
    mass:          float = 0.0
    moment:        float = 0.0
    com:           np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        Ro, Ri, L = (float(self.outer_radius),
                     float(self.inner_radius),
                     float(self.length))
        if Ro <= 0 or L <= 0:
            raise ValueError("outer_radius and length must be > 0")
        if Ri < 0 or Ri >= Ro:
            raise ValueError("inner_radius must be in [0, outer_radius)")
        V = np.pi * (Ro ** 2 - Ri ** 2) * L
        m = self.density * V
        self.volume = V
        self.mass   = m
        self.moment = self.B_remanence * V / MU_0

    @property
    def inertia_xx(self) -> float:
        return 0.5 * self.mass * (self.outer_radius ** 2
                                  + self.inner_radius ** 2)
    @property
    def inertia_yy(self) -> float:
        return self.mass * (
            3.0 * (self.outer_radius ** 2 + self.inner_radius ** 2)
            + self.length ** 2
        ) / 12.0
    @property
    def inertia_zz(self) -> float:
        return self.inertia_yy

    def _magpy_source(self, r0: np.ndarray, theta: float, phi: float):
        import magpylib as magpy
        M = self.B_remanence / MU_0
        rot = _rotation_aligning(np.array([0.0, 0.0, 1.0]),
                                 _hat_m(theta, phi))
        # CylinderSegment: dimension = (r1, r2, h, phi1_deg, phi2_deg).
        # r1 = inner, r2 = outer, h = length along local z, full ring -> 0..360.
        src = magpy.magnet.CylinderSegment(
            magnetization=(0.0, 0.0, M),
            dimension=(self.inner_radius, self.outer_radius,
                       self.length, 0.0, 360.0),
        )
        src.orientation = rot
        src.position = np.asarray(r0, dtype=float).reshape(3)
        return src

    def B_field(self, points: np.ndarray, r0: np.ndarray,
                theta: float, phi: float) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        src = self._magpy_source(r0, theta, phi)
        return src.getB(pts)

    def interaction_energy(self, phi_sol, r0: np.ndarray,
                           theta: float, phi: float) -> float:
        return _interaction_energy_from_phi(self.moment, phi_sol,
                                            r0, theta, phi)

    def __str__(self) -> str:
        return "\n".join([
            "RingMagnetParticle",
            f"  R_outer x R_inner : {self.outer_radius*1e3:.3f} x "
            f"{self.inner_radius*1e3:.3f} mm",
            f"  length            : {self.length*1e3:.3f} mm",
            f"  density           : {self.density:.0f} kg/m^3",
            f"  mu_0 M (Brem)     : {self.B_remanence:.3f} T",
            f"  mass              : {self.mass*1e9:.4f} ug",
            f"  I_xx (spin/bar)   : {self.inertia_xx:.4e} kg m^2",
            f"  I_yy = I_zz       : {self.inertia_yy:.4e} kg m^2",
            f"  dipole moment |m| : {self.moment:.4e} A m^2",
        ])

    def as_dict(self) -> dict:
        return {
            "kind":            "RingMagnetParticle",
            "outer_radius_m":  self.outer_radius,
            "inner_radius_m":  self.inner_radius,
            "length_m":        self.length,
            "density_kgm3":    self.density,
            "B_remanence_T":   self.B_remanence,
            "mass_kg":         self.mass,
            "moment_Am2":      self.moment,
            "inertia_xx_kgm2": self.inertia_xx,
            "inertia_yy_kgm2": self.inertia_yy,
            "inertia_zz_kgm2": self.inertia_zz,
        }
