"""Quickstart: closed elliptical / circular superconducting trap.

Edit `scripts/parameters.py` (or pass another parameters file as the first
CLI arg) and run:

    cd sctrap
    python3 scripts/quickstart_elliptical.py
    python3 scripts/quickstart_elliptical.py examples/parameters_vinante.py
    python3 scripts/quickstart_elliptical.py examples/parameters_fuchs.py

Pipeline (one CLI invocation does all 6 steps):

    1. Build the particle (sphere or composite).
    2. Generate a closed elliptical-cavity gmsh mesh.
    3. Find equilibrium under gravity (with optional tilt).
    4. Build the 5x5 Hessian and diagonalise -> normal modes.
    5. Standard report (JSON, CSV, summary).
    6. Sanity plots (mesh, particle, modes, Hessian, B-field, U scans).

All artifacts go to  sctrap/results/<OUT_DIR>/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_ROOT = REPO / "results"


# ---- parameters loading ----------------------------------------------------
def _load_parameters(arg: str | None):
    if arg:
        path = Path(arg).resolve()
    else:
        path = HERE / "parameters.py"
    spec = importlib.util.spec_from_file_location("user_parameters", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load parameters from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


# ---------------------------------------------------------------------------
from sctrap.analytic_box import analytic_box_estimate
from sctrap.dipole import dipole_moment
from sctrap.frequencies import find_equilibrium, find_equilibrium_5d
from sctrap.generators import elliptical_cavity
from sctrap.half_space import image_equilibrium_height
from sctrap.mesh import load_msh
from sctrap.modes import normal_modes
from sctrap.particle import (
    BarMagnetParticle,
    CompositeParticle,
    CylinderMagnetParticle,
    RingMagnetParticle,
    SphericalParticle,
)
from sctrap.potential import gravity_from_tilt
from sctrap.report import write_text_artifacts, plot_U_along_dofs, plot_mode_shapes
from sctrap.sanity_plots import (
    plot_B_induced_slice, plot_equilibrium_residual,
    plot_frequency_bar, plot_mesh_overview,
    plot_particle_diagram, plot_U_long_axes,
)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def build_particle(P):
    kind = getattr(P, "PARTICLE_KIND", "composite").lower()
    if kind == "sphere":
        return SphericalParticle(
            radius      = P.SPHERE_PARTICLE_RADIUS,
            density     = P.SPHERE_PARTICLE_DENSITY,
            B_remanence = P.SPHERE_PARTICLE_BREM,
        )
    if kind == "composite":
        common = dict(
            cube_edge       = P.CUBE_EDGE,
            n_cubes         = P.N_CUBES,
            sphere_radius   = P.BEAD_RADIUS,
            sphere_position = np.asarray(P.BEAD_POSITION, dtype=float),
            magnet_density  = P.DENSITY_NDFEB,
            sphere_density  = P.DENSITY_BOROSILICATE,
            B_remanence     = P.B_REMANENCE,
        )
        if P.RESCALE_TO_MASS is not None:
            return CompositeParticle.from_total_mass(
                total_mass=P.RESCALE_TO_MASS, **common)
        return CompositeParticle(**common)
    if kind == "bar":
        return BarMagnetParticle(
            length      = P.BAR_LENGTH,
            width       = P.BAR_WIDTH,
            height      = P.BAR_HEIGHT,
            density     = P.BAR_DENSITY,
            B_remanence = P.BAR_BREM,
        )
    if kind == "cylinder":
        return CylinderMagnetParticle(
            radius      = P.CYL_RADIUS,
            length      = P.CYL_LENGTH,
            density     = P.CYL_DENSITY,
            B_remanence = P.CYL_BREM,
        )
    if kind == "ring":
        return RingMagnetParticle(
            outer_radius = P.RING_OUTER_RADIUS,
            inner_radius = P.RING_INNER_RADIUS,
            length       = P.RING_LENGTH,
            density      = P.RING_DENSITY,
            B_remanence  = P.RING_BREM,
        )
    raise ValueError(
        "PARTICLE_KIND must be 'sphere', 'composite', 'bar', "
        f"'cylinder', or 'ring'; got {kind!r}"
    )


def particle_inertia_for_modes(particle):
    """Return scalar or 2-tuple (I_theta, I_phi) for normal_modes()."""
    if isinstance(particle, SphericalParticle):
        return particle.inertia
    return (particle.inertia_yy, particle.inertia_zz)


def _is_volumetric(particle) -> bool:
    """True if the particle is one of the magpylib-backed finite-size types."""
    return isinstance(particle, (BarMagnetParticle,
                                 CylinderMagnetParticle,
                                 RingMagnetParticle))


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    P, params_path = _load_parameters(arg)
    print(f"  parameters from : {params_path}")

    out_dir = RESULTS_ROOT / P.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "_meshes"
    mesh_dir.mkdir(exist_ok=True)
    mesh_path = mesh_dir / "cavity.msh"

    # ---------------------------------------------------------------- 1
    banner("[1/6]  Particle")
    particle = build_particle(P)
    print(particle)
    if isinstance(particle, CompositeParticle):
        plot_particle_diagram(particle, out_dir / "particle.png")

    # ---------------------------------------------------------------- 2
    banner("[2/6]  Mesh")
    print(f"  cavity      : {2*P.CAVITY_A*1e3:.2f} x {2*P.CAVITY_B*1e3:.2f} mm  "
          f"x  H={P.CAVITY_HEIGHT*1e3:.2f} mm  (closed)")
    print(f"  mesh size   : {P.MESH_SIZE*1e3:.3f} mm "
          f"(near-SC: {P.MESH_SIZE_NEAR})")
    t0 = time.time()
    elliptical_cavity(
        P.CAVITY_A, P.CAVITY_B, P.CAVITY_HEIGHT,
        mesh_size_sc    = P.MESH_SIZE,
        mesh_size_near  = P.MESH_SIZE_NEAR,
        refine_distance = P.MESH_REFINE_DIST,
        output_file     = str(mesh_path),
        verbose         = False,
    )
    sct = load_msh(mesh_path)
    print(f"  built in {time.time()-t0:.1f} s   ->   {sct}")
    plot_mesh_overview(sct, out_dir / "mesh.png")

    # ---------------------------------------------------------------- 3
    banner("[3/6]  Equilibrium")
    tilt_angle = float(getattr(P, "TILT_ANGLE", 0.0))
    tilt_axis  = getattr(P, "TILT_AXIS", "y")
    g_vec = gravity_from_tilt(tilt_angle, tilt_axis)
    if abs(tilt_angle) > 1e-12:
        print(f"  tilt        : {np.rad2deg(tilt_angle):.3f} deg about {tilt_axis}-axis")
        print(f"  g_vec       : ({g_vec[0]:+.4f}, {g_vec[1]:+.4f}, {g_vec[2]:+.4f}) m/s^2")
    else:
        print(f"  tilt        : none (gravity = -z)")

    # For volumetric particles the point-dipole singularity-subtraction is
    # not applicable; the source field is computed from the magnet volume.
    volumetric = _is_volumetric(particle)
    subtract_sing = bool(P.SUBTRACT_SINGULARITY) and not volumetric
    if volumetric and P.SUBTRACT_SINGULARITY:
        print("  (volumetric particle) -> disabling subtract_singularity")
    solver_kwargs = dict(element=P.ELEMENT,
                         subtract_singularity=subtract_sing)
    # Only the point-dipole code path knows about subtract_singularity; the
    # volumetric path silently drops it (the kwarg is incompatible there).
    volumetric_kw = dict(element=P.ELEMENT)

    # Analytic seed + benchmark from the cuboidal image-lattice model.
    # Approximating the elliptical bore by the rectangular box [-a,a]x[-b,b]x
    # [0,H] gives, in closed form, the equilibrium height AND the full diagonal
    # Hessian (k_x,k_y,k_z, k_theta,k_phi) -> all five mode frequencies. We use
    # it to (a) seed z_eq so the FEM optimiser starts at the minimum, and (b)
    # print an independent benchmark for the FEM frequencies that follow.
    #
    # The optimiser's energy tolerance (fatol) is handled separately, inside
    # find_equilibrium_5d, as ftol_rel * |U0| with ftol_rel safely above the
    # FEM objective's relative noise floor (~4e-5 for the singularity-subtracted
    # solve). We deliberately do NOT match fatol to the analytic curvature: that
    # scale (~1e-15 J for a 0.2 um target) sits ~1000x below the noise floor, so
    # the f-test could never be met and the search would grind to maxiter. The
    # floor caps the achievable equilibrium resolution at ~10 um regardless --
    # which is fine, since the frequencies come from the Hessian window, not the
    # equilibrium position. Set AUTO_Z_GUESS = False to keep the hand-set
    # EQ_GUESS[2].
    g_mag = float(np.linalg.norm(g_vec))
    r0_guess = np.asarray(P.EQ_GUESS, dtype=float).copy()
    have_box = all(hasattr(P, k) for k in ("CAVITY_A", "CAVITY_B", "CAVITY_HEIGHT"))
    if have_box:
        est = analytic_box_estimate(
            P.CAVITY_A, P.CAVITY_B, P.CAVITY_HEIGHT,
            particle.moment, particle.mass,
            theta0=P.EQ_THETA, phi0=P.EQ_PHI, g=g_mag,
            I_theta=particle.inertia_yy, I_phi=particle.inertia_zz,
        )
        print("  analytic box estimate (cuboidal image lattice) [benchmark]:")
        print(f"    z_eq = {est['z_eq']*1e3:.4f} mm    "
              f"k = (x {est['k_x']*1e3:.3f}, y {est['k_y']*1e3:.3f}, "
              f"z {est['k_z']*1e3:.3f}) mN/m")
        print(f"    f_x={est['f_x']:.2f}  f_y={est['f_y']:.2f}  "
              f"f_z={est['f_z']:.2f} Hz   |   "
              f"k_theta={est['k_theta']*1e9:.3f}  k_phi={est['k_phi']*1e9:.3f} nJ/rad^2"
              + (f"  (f_theta={est['f_theta']:.2f}  f_phi={est['f_phi']:.2f} Hz)"
                 if est['f_theta'] and est['f_phi'] else ""))
        if bool(getattr(P, "AUTO_Z_GUESS", True)):
            r0_guess[2] = est["z_eq"]
    elif bool(getattr(P, "AUTO_Z_GUESS", True)):
        # No cuboidal dims: fall back to the single-image half-space height.
        z_above_floor = image_equilibrium_height(
            particle.moment, P.EQ_THETA, particle.mass, g=g_mag)
        sc_centroids_z = sct.mesh.p[2, sct.mesh.facets[:, sct.sc_facets]].mean(axis=0)
        floor_z = float(sc_centroids_z.min())
        r0_guess[2] = floor_z + z_above_floor
        print(f"  analytic z_eq (image dipole) = {z_above_floor*1e3:+.3f} mm "
              f"above floor (z={floor_z*1e3:+.3f} mm)")

    print(f"  initial guess r0 = ({r0_guess[0]*1e3:+.3f}, "
          f"{r0_guess[1]*1e3:+.3f}, {r0_guess[2]*1e3:+.3f}) mm")
    # By default, jointly relax position AND orientation (find_equilibrium_5d):
    # in an anisotropic cavity the preferred orientation is found rather than
    # assumed, so the Hessian is taken at a true minimum (no spurious negative
    # libration modes). Set FIND_ORIENTATION = False to pin (EQ_THETA, EQ_PHI).
    find_orientation = bool(getattr(P, "FIND_ORIENTATION", True))
    eq_kw = dict(particle=particle, **volumetric_kw) if volumetric \
        else dict(**solver_kwargs)
    t0 = time.time()
    if find_orientation:
        print(f"  orientation guess (theta, phi) = "
              f"({P.EQ_THETA:+.4f}, {P.EQ_PHI:+.4f}) rad  -> relaxing")
        eq_r, eq_theta, eq_phi = find_equilibrium_5d(
            sct, particle.moment, r0_guess,
            theta_guess=P.EQ_THETA, phi_guess=P.EQ_PHI,
            mass=particle.mass, g_vec=g_vec,
            progress=True, **eq_kw,
        )
    else:
        m_vec = dipole_moment(particle.moment, P.EQ_THETA, P.EQ_PHI)
        eq_r = find_equilibrium(
            sct, m_vec, r0_guess,
            mass=particle.mass, g_vec=g_vec, progress=True, **eq_kw,
        )
        eq_theta, eq_phi = float(P.EQ_THETA), float(P.EQ_PHI)
    print(f"  r_eq = ({eq_r[0]*1e3:+.4f}, {eq_r[1]*1e3:+.4f}, "
          f"{eq_r[2]*1e3:+.4f}) mm   ({time.time()-t0:.1f} s)")
    print(f"  orientation (theta, phi) = ({eq_theta:+.4f}, {eq_phi:+.4f}) rad"
          + ("   [relaxed]" if find_orientation else "   [pinned]"))
    plot_mesh_overview(sct, out_dir / "mesh.png", eq_r=eq_r)

    # ---------------------------------------------------------------- 4
    banner("[4/6]  Normal modes")
    t0 = time.time()
    if volumetric:
        nm = normal_modes(
            sct, eq_r, particle.moment,
            eq_theta = eq_theta, eq_phi = eq_phi,
            mass     = particle.mass,
            inertia  = particle_inertia_for_modes(particle),
            h_trans  = P.H_TRANS, h_ang = P.H_ANG,
            g_vec    = g_vec,
            particle = particle,
            progress = True,
            **volumetric_kw,
        )
    else:
        nm = normal_modes(
            sct, eq_r, particle.moment,
            eq_theta = eq_theta, eq_phi = eq_phi,
            mass     = particle.mass,
            inertia  = particle_inertia_for_modes(particle),
            h_trans  = P.H_TRANS, h_ang = P.H_ANG,
            g_vec    = g_vec,
            progress = True,
            **solver_kwargs,
        )
    print(f"  ({time.time()-t0:.1f} s)")
    print()
    print(nm)

    # ---------------------------------------------------------------- 5
    banner("[5/6]  Standard report")
    write_text_artifacts(nm, out_dir)
    plot_mode_shapes(nm, out_dir / "mode_shapes.png")
    plot_U_along_dofs(sct, nm, particle.moment,
                       out_dir / "U_along_DOFs.png",
                       n_pts=13, span_factor=4.0,
                       mass=particle.mass, **solver_kwargs)
    print(f"  ->  {out_dir}/(summary.txt, result.json, modes.csv, *.png)")

    # ---------------------------------------------------------------- 6
    banner("[6/6]  Sanity plots")
    pmin, pmax = sct.bounding_box()
    safe_span = 0.85 * min(P.CAVITY_A, P.CAVITY_B) / np.sqrt(2.0)

    plot_frequency_bar(nm, out_dir / "frequencies.png",
                       compare_f_Hz=getattr(P, "COMPARE_F_HZ", None),
                       compare_label=getattr(P, "COMPARE_LABEL", ""))
    plot_equilibrium_residual(nm, out_dir / "hessian_card.png")
    print("  + B-field xz slice")
    plot_B_induced_slice(
        sct, eq_r, eq_theta, eq_phi, particle.moment,
        out_dir / "B_induced_xz.png",
        span=float(safe_span), n_pts=21, **solver_kwargs)
    print("  + U(x), U(y), U(z) long-range scans")
    long_axes = plot_U_long_axes(
        sct, eq_r, eq_theta, eq_phi, particle.moment, particle.mass,
        span=float(safe_span), n_pts=11,
        out_path=out_dir / "U_long_axes.png", **solver_kwargs)
    (out_dir / "U_long_axes.json").write_text(json.dumps(long_axes, indent=2))

    # Snapshot
    snapshot = {
        "parameters_path": str(params_path),
        "particle":        particle.as_dict(),
        "tilt_angle_rad":  tilt_angle,
        "tilt_axis":       tilt_axis,
        "g_vec":           g_vec.tolist(),
        "eq_r_m":          eq_r.tolist(),
    }
    (out_dir / "run_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, default=lambda o: o.tolist()
                    if hasattr(o, "tolist") else str(o)))

    banner("DONE")
    print(f"  artifacts:  {out_dir}")
    if getattr(P, "COMPARE_F_HZ", None) is not None:
        z_idx = int(np.argmax(np.abs(nm.eigvecs_mw[2, :])))
        f_z = float(nm.f_Hz[z_idx])
        diff = (f_z - P.COMPARE_F_HZ) / P.COMPARE_F_HZ * 100.0
        print(f"  z-dominant mode f_z = {f_z:8.3f} Hz")
        print(f"  reference          = {P.COMPARE_F_HZ:8.3f} Hz "
              f"({P.COMPARE_LABEL})")
        print(f"  agreement          = {diff:+.2f} %")


if __name__ == "__main__":
    main()
