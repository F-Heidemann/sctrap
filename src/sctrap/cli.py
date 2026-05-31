"""``sctrap`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import (
    G_GRAV, MAGNET_MASS, MAGNET_INERTIA, MAGNET_MOMENT,
)
from .dipole import dipole_moment
from .mesh import load_msh
from .frequencies import find_equilibrium, frequencies_at
from .modes import normal_modes
from .report import write_report
from .uncertainty import stencil_richardson


# ---------------------------------------------------------------------------
# `simulate`
# ---------------------------------------------------------------------------

def cmd_simulate(args: argparse.Namespace) -> int:
    sctmesh = load_msh(args.mesh)
    print(f"Loaded mesh: {sctmesh}")
    bbox_lo, bbox_hi = sctmesh.bounding_box()
    print(f"  bbox: lo={bbox_lo}  hi={bbox_hi}")

    # Initial position guess: a few mm above the origin (or user-supplied)
    if args.r0 is not None:
        r0 = np.array(args.r0, dtype=float)
    else:
        r0 = np.array([0.0, 0.0, 0.5 * (bbox_hi[2] - bbox_lo[2]) * 0.05])
        # ~5 % of the vertical extent above the origin
    print(f"  initial position guess r0 = {r0} m")

    m = dipole_moment(MAGNET_MOMENT, theta=args.theta, phi=args.phi)
    print(f"  dipole moment |m| = {np.linalg.norm(m):.4e} A m^2  (theta={args.theta}, phi={args.phi})")

    solver_kwargs = dict(
        element=args.element,
        subtract_singularity=args.subtract_singularity,
    )

    print("\n[1/2] Finding equilibrium...")
    eq = find_equilibrium(
        sctmesh, m, r0,
        mass=MAGNET_MASS,
        verbose=args.verbose,
        **solver_kwargs,
    )
    print(f"  equilibrium r_eq = {eq} m")
    # Levitation height: vertical position of the equilibrium above the
    # bottom of the geometry's bounding box (the SC floor for a bowl/cavity,
    # or the lowest CAD point for an imported solid).
    lev_height = float(eq[2] - bbox_lo[2])
    print(f"  levitation height (z_eq - z_min) = {lev_height*1e3:.4f} mm "
          f"  [z_eq = {eq[2]*1e3:.4f} mm]")

    if args.modes:
        print("\n[2/2] Building 5x5 Hessian and diagonalising...")
        band = None
        if args.converge:
            print("       (with --converge: running stencils h and h/sqrt(2))")
            _, nm, band = stencil_richardson(
                sctmesh, eq, float(np.linalg.norm(m)),
                eq_theta=args.theta, eq_phi=args.phi,
                mass=MAGNET_MASS, inertia=MAGNET_INERTIA,
                h_trans=args.h_trans, h_ang=args.h_ang,
                compute_anharmonic=args.anharmonic,
                verbose=args.verbose,
                **solver_kwargs,
            )
        else:
            nm = normal_modes(
                sctmesh, eq, float(np.linalg.norm(m)),
                eq_theta=args.theta, eq_phi=args.phi,
                mass=MAGNET_MASS, inertia=MAGNET_INERTIA,
                h_trans=args.h_trans, h_ang=args.h_ang,
                compute_anharmonic=args.anharmonic,
                verbose=args.verbose,
                **solver_kwargs,
            )
        print()
        print(nm)
        if band is not None:
            print("\n  Stencil-Richardson uncertainty band  f ± df  [Hz]:")
            for k, (f, df) in enumerate(zip(band.f_Hz, band.df_Hz)):
                print(f"    mode {k+1}:  {f:11.4f}  ±  {df:8.4f}")

        if args.report is not None:
            print(f"\nWriting report to {args.report} ...")
            write_report(
                sctmesh, nm, float(np.linalg.norm(m)),
                args.report,
                plot_slice_span=args.slice_span,
                plot_slice_pts=args.slice_pts,
                mass=MAGNET_MASS,
                skip_plots=args.no_plots,
                uncertainty=band,
                **solver_kwargs,
            )
            print("Done.")

        if args.out is not None:
            Path(args.out).write_text(json.dumps(nm.as_dict(), indent=2))
            print(f"\nWrote {args.out}")

        return 0

    print("\n[2/2] Fitting trap frequencies (per-DOF parabola)...")
    tf = frequencies_at(
        sctmesh, eq, m,
        mass=MAGNET_MASS, inertia=MAGNET_INERTIA,
        h_trans=args.h_trans, h_ang=args.h_ang, n_pts=args.n_pts,
        verbose=args.verbose,
        **solver_kwargs,
    )
    print()
    print(tf)

    if args.out is not None:
        Path(args.out).write_text(json.dumps(tf.as_dict(), indent=2))
        print(f"\nWrote {args.out}")

    return 0


# ---------------------------------------------------------------------------
# `mesh`
# ---------------------------------------------------------------------------

def cmd_mesh(args: argparse.Namespace) -> int:
    from . import generators as G

    if args.shape == "ellipse":
        out = G.elliptical_cylinder(
            a=args.a, b=args.b, half_height=args.h, far_radius=args.R,
            mesh_size_sc=args.size_sc, mesh_size_ff=args.size_ff,
            mesh_size_near=args.size_near, refine_distance=args.refine_distance,
            output_file=args.out, verbose=args.verbose,
        )
    elif args.shape == "plates":
        out = G.two_parallel_plates(
            plate_half_side=args.L, plate_thickness=args.t,
            plate_separation=args.d, far_radius=args.R,
            mesh_size_sc=args.size_sc, mesh_size_ff=args.size_ff,
            mesh_size_near=args.size_near, refine_distance=args.refine_distance,
            output_file=args.out, verbose=args.verbose,
        )
    elif args.shape == "bowl":
        out = G.hemispherical_bowl(
            radius=args.r, far_radius=args.R,
            mesh_size_sc=args.size_sc, mesh_size_ff=args.size_ff,
            mesh_size_near=args.size_near, refine_distance=args.refine_distance,
            output_file=args.out, verbose=args.verbose,
        )
    elif args.shape == "import":
        out = G.import_cad(
            args.cad, far_radius=args.R,
            mesh_size_sc=args.size_sc, mesh_size_ff=args.size_ff,
            mesh_size_near=args.size_near, refine_distance=args.refine_distance,
            scale=args.scale, output_file=args.out, verbose=args.verbose,
        )
    else:                                                    # pragma: no cover
        print(f"Unknown shape '{args.shape}'", file=sys.stderr)
        return 2

    print(f"Wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# `validate`
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    """Two-plate analytic image-dipole benchmark.

    Compares the FEM U_mag(z) for a dipole between two large parallel SC
    plates against the closed-form image series.
    """
    from .validation import two_plate_benchmark

    return two_plate_benchmark(
        plate_separation=args.separation,
        plate_half_side=args.L,
        far_radius=args.R,
        mesh_size_sc=args.size_sc,
        mesh_size_ff=args.size_ff,
        mesh_size_near=args.size_near,
        refine_distance=args.refine_distance,
        subtract_singularity=args.subtract_singularity,
        n_z=args.n_z,
        n_images=args.n_images,
        tol_pct=args.tol,
        verbose=args.verbose,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sctrap", description=__doc__)
    p.add_argument("--version", action="version", version=f"sctrap {__version__}")

    sub = p.add_subparsers(dest="command", required=True)

    # ---- simulate -------------------------------------------------------
    s = sub.add_parser("simulate", help="Run equilibrium + frequency fit on a mesh.")
    s.add_argument("--mesh", required=True, help="Path to a gmsh .msh file.")
    s.add_argument("--out", default=None, help="Write JSON results to this path.")
    s.add_argument("--r0", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   default=None, help="Initial position guess [m].")
    s.add_argument("--theta", type=float, default=0.0,
                   help="Dipole polar angle [rad] (default: 0, |m| along +z).")
    s.add_argument("--phi", type=float, default=0.0,
                   help="Dipole azimuthal angle [rad] (default: 0).")
    s.add_argument("--element", choices=("P1", "P2"), default="P2",
                   help="Lagrange element order (default: P2).")
    s.add_argument("--subtract-singularity", dest="subtract_singularity",
                   action="store_true",
                   help="Use image-dipole singularity subtraction in the "
                        "Neumann RHS (recommended near the SC surface).")
    s.add_argument("--h-trans", dest="h_trans", type=float, default=50e-6,
                   help="Translational stencil half-width [m] (default: 5e-5).")
    s.add_argument("--h-ang", dest="h_ang", type=float, default=1e-3,
                   help="Angular stencil half-width [rad] (default: 1e-3).")
    s.add_argument("--n-pts", dest="n_pts", type=int, default=5,
                   help="Stencil points per DOF (odd, default: 5).")
    s.add_argument("--modes", action="store_true",
                   help="Use full 5x5 Hessian eigendecomposition (true normal "
                        "modes, mixes chart DOFs). Recommended.")
    s.add_argument("--anharmonic", action="store_true",
                   help="With --modes, also fit a quartic per DOF and report "
                        "the diagonal anharmonic coefficient k4.")
    s.add_argument("--converge", action="store_true",
                   help="With --modes, run the Hessian a second time at "
                        "stencil width h/sqrt(2) and Richardson-extrapolate "
                        "to give an f ± df band per mode.")
    s.add_argument("--report", default=None,
                   help="With --modes, write JSON+CSV+TXT and matplotlib "
                        "plots into this directory.")
    s.add_argument("--no-plots", dest="no_plots", action="store_true",
                   help="With --report, skip the plots (text artifacts only).")
    s.add_argument("--slice-span", dest="slice_span", type=float, default=1.5e-3,
                   help="Half-width of the 2-D U slice plots [m] (default 1.5e-3).")
    s.add_argument("--slice-pts", dest="slice_pts", type=int, default=21,
                   help="Grid points per side of the 2-D U slice plots.")
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_simulate)

    # ---- mesh -----------------------------------------------------------
    m = sub.add_parser("mesh", help="Generate a built-in trap mesh via gmsh.")
    msub = m.add_subparsers(dest="shape", required=True)

    me = msub.add_parser("ellipse", help="Solid elliptical-cylinder SC.")
    me.add_argument("--a", type=float, required=True, help="x semi-axis [m]")
    me.add_argument("--b", type=float, required=True, help="y semi-axis [m]")
    me.add_argument("--h", type=float, required=True, help="half-height [m]")
    me.add_argument("--R", type=float, default=None,
                    help="far-field sphere radius [m] (default 5*max(a,b,h))")
    me.add_argument("--size-sc", dest="size_sc", type=float, default=1e-3)
    me.add_argument("--size-ff", dest="size_ff", type=float, default=5e-3)
    me.add_argument("--size-near", dest="size_near", type=float, default=None,
                    help="Local element size on/near SC surfaces (Distance+Threshold field).")
    me.add_argument("--refine-distance", dest="refine_distance", type=float, default=None,
                    help="Distance over which the size ramps from --size-near to --size-ff.")
    me.add_argument("--out", required=True)
    me.add_argument("--verbose", "-v", action="store_true")
    me.set_defaults(func=cmd_mesh)

    mp = msub.add_parser("plates", help="Two parallel SC plates.")
    mp.add_argument("--L", type=float, required=True, help="plate half-side [m]")
    mp.add_argument("--t", type=float, required=True, help="plate thickness [m]")
    mp.add_argument("--d", type=float, required=True, help="plate separation [m]")
    mp.add_argument("--R", type=float, default=None,
                    help="far-field sphere radius [m] (default 5*L)")
    mp.add_argument("--size-sc", dest="size_sc", type=float, default=1e-3)
    mp.add_argument("--size-ff", dest="size_ff", type=float, default=5e-3)
    mp.add_argument("--size-near", dest="size_near", type=float, default=None,
                    help="Local element size on/near SC surfaces (Distance+Threshold field).")
    mp.add_argument("--refine-distance", dest="refine_distance", type=float, default=None,
                    help="Distance over which the size ramps from --size-near to --size-ff.")
    mp.add_argument("--out", required=True)
    mp.add_argument("--verbose", "-v", action="store_true")
    mp.set_defaults(func=cmd_mesh)

    mb = msub.add_parser("bowl", help="Hemispherical SC bowl.")
    mb.add_argument("--r", type=float, required=True, help="bowl radius [m]")
    mb.add_argument("--R", type=float, default=None, help="far-field radius [m]")
    mb.add_argument("--size-sc", dest="size_sc", type=float, default=1e-3)
    mb.add_argument("--size-ff", dest="size_ff", type=float, default=5e-3)
    mb.add_argument("--size-near", dest="size_near", type=float, default=None,
                    help="Local element size on/near SC surfaces (Distance+Threshold field).")
    mb.add_argument("--refine-distance", dest="refine_distance", type=float, default=None,
                    help="Distance over which the size ramps from --size-near to --size-ff.")
    mb.add_argument("--out", required=True)
    mb.add_argument("--verbose", "-v", action="store_true")
    mb.set_defaults(func=cmd_mesh)

    mi = msub.add_parser(
        "import",
        help="Import a CAD solid (STEP/IGES/BREP) as the SC and build a trap mesh.",
    )
    mi.add_argument("--cad", required=True,
                    help="CAD file (.step/.stp/.iges/.igs/.brep). Treated as the SC body.")
    mi.add_argument("--R", type=float, default=None,
                    help="far-field sphere radius [m] (default 5x the solid's "
                         "half bounding-box diagonal).")
    mi.add_argument("--scale", type=float, default=1.0,
                    help="Multiply imported coordinates, e.g. 1e-3 to convert "
                         "a CAD file authored in mm to SI metres.")
    mi.add_argument("--size-sc", dest="size_sc", type=float, default=1e-3,
                    help="Element size on the SC surface [m].")
    mi.add_argument("--size-ff", dest="size_ff", type=float, default=None,
                    help="Element size on the far field [m] (default far_radius/6).")
    mi.add_argument("--size-near", dest="size_near", type=float, default=None,
                    help="Local element size on/near SC surfaces (Distance+Threshold field).")
    mi.add_argument("--refine-distance", dest="refine_distance", type=float, default=None,
                    help="Distance over which the size ramps from --size-near to --size-ff.")
    mi.add_argument("--out", required=True)
    mi.add_argument("--verbose", "-v", action="store_true")
    mi.set_defaults(func=cmd_mesh)

    # ---- validate -------------------------------------------------------
    v = sub.add_parser("validate", help="Run the two-plate analytic benchmark.")
    v.add_argument("--separation", type=float, default=4e-3,
                   help="Plate separation [m] (default 4 mm).")
    v.add_argument("--L", type=float, default=15e-3, help="plate half-side [m]")
    v.add_argument("--R", type=float, default=60e-3, help="far-field radius [m]")
    v.add_argument("--size-sc", dest="size_sc", type=float, default=0.6e-3)
    v.add_argument("--size-ff", dest="size_ff", type=float, default=8e-3)
    v.add_argument("--size-near", dest="size_near", type=float, default=None,
                   help="Local element size on/near SC surfaces.")
    v.add_argument("--refine-distance", dest="refine_distance", type=float, default=None,
                   help="Ramp distance from --size-near to --size-ff.")
    v.add_argument("--subtract-singularity", dest="subtract_singularity",
                   action="store_true",
                   help="Use image-dipole singularity subtraction in the Neumann RHS.")
    v.add_argument("--n-z", dest="n_z", type=int, default=11,
                   help="number of z samples between plates")
    v.add_argument("--n-images", dest="n_images", type=int, default=40,
                   help="image-dipole truncation order")
    v.add_argument("--tol", type=float, default=10.0,
                   help="pass tolerance, %% (default 10 %%)")
    v.add_argument("--verbose", "-v", action="store_true")
    v.set_defaults(func=cmd_validate)

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Defaults that depend on other args
    if args.command == "mesh":
        if args.shape == "ellipse" and args.R is None:
            args.R = 5.0 * max(args.a, args.b, args.h)
        elif args.shape == "plates" and args.R is None:
            args.R = 5.0 * args.L
        elif args.shape == "bowl" and args.R is None:
            args.R = 5.0 * args.r

    return args.func(args)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
