"""Built-in gmsh mesh generators for common SC trap shapes.

These are optional helpers — the first-class workflow is loading a `.msh`
the user has prepared themselves. Generators here exist so `sctrap mesh` and
`sctrap validate` can produce reference geometries with one command.

`gmsh` is an optional dependency; importing this module raises a clear error
if gmsh is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import gmsh
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "The `gmsh` package is required for `sctrap.generators`. "
        "Install with: pip install gmsh"
    ) from exc


def _start(name: str, verbose: bool):
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add(name)


def _finish(out: str | Path) -> str:
    gmsh.model.mesh.generate(3)
    out = str(out)
    gmsh.write(out)
    gmsh.model.remove()
    return out


def _add_sc_distance_field(
    sc_surface_tags: list[int],
    size_near: float,
    size_far: float,
    dist_min: float,
    dist_max: float,
) -> None:
    """Distance+Threshold field: size_near on SC surfaces, ramps to size_far
    over the range [dist_min, dist_max]. Combined as the global mesh-size field
    via a Min field if a previous field is already set."""
    if size_near is None or size_near <= 0 or size_near >= size_far:
        return
    mf = gmsh.model.mesh.field
    fd = mf.add("Distance")
    mf.setNumbers(fd, "SurfacesList", sc_surface_tags)
    mf.setNumber(fd, "Sampling", 100)

    ft = mf.add("Threshold")
    mf.setNumber(ft, "InField", fd)
    mf.setNumber(ft, "SizeMin", size_near)
    mf.setNumber(ft, "SizeMax", size_far)
    mf.setNumber(ft, "DistMin", dist_min)
    mf.setNumber(ft, "DistMax", dist_max)

    mf.setAsBackgroundMesh(ft)
    # NOTE: leave Mesh.MeshSizeFromPoints / FromCurvature / ExtendFromBoundary
    # at their defaults. Gmsh combines the background field with point-based
    # size hints by taking the minimum, so the FF sphere's far-field point
    # sizes still drive the mesh away from the SC. Disabling those sources
    # produced singular stiffness matrices because the FF region degenerated.


def two_parallel_plates(
    plate_half_side: float,
    plate_thickness: float,
    plate_separation: float,
    far_radius: float,
    mesh_size_sc: float = 1e-3,
    mesh_size_ff: float = 5e-3,
    mesh_size_near: float | None = None,
    refine_distance: float | None = None,
    output_file: str | Path = "plates.msh",
    verbose: bool = False,
) -> str:
    """Two square SC plates separated along z, embedded in a far-field sphere.

    Physical groups: ``"SC"`` (plates), ``"FF"`` (sphere), ``"air"`` (volume).
    """
    _start("two_plates", verbose)
    occ = gmsh.model.occ

    L = plate_half_side
    t = plate_thickness
    d = plate_separation / 2.0

    bot = occ.addBox(-L, -L, -d - t, 2 * L, 2 * L, t)
    top = occ.addBox(-L, -L,  d,      2 * L, 2 * L, t)
    sph = occ.addSphere(0, 0, 0, far_radius)

    air, _ = occ.cut([(3, sph)], [(3, bot), (3, top)], removeTool=False)
    occ.synchronize()

    air_tag = air[0][1]

    all_surfs = [s[1] for s in gmsh.model.getEntities(2)]
    plate_surfs = list(set(
        occ.getSurfaceLoops(bot)[1][0].tolist()
        + occ.getSurfaceLoops(top)[1][0].tolist()
    ))
    ff_surfs = [s for s in all_surfs if s not in plate_surfs]

    gmsh.model.addPhysicalGroup(2, plate_surfs, name="SC")
    gmsh.model.addPhysicalGroup(2, ff_surfs,    name="FF")
    gmsh.model.addPhysicalGroup(3, [air_tag],   name="air")

    _set_size_at_box((-L, -L, -d - t, L, L, d + t), mesh_size_sc)
    _set_size_at_radius(far_radius, mesh_size_ff)

    if mesh_size_near is not None:
        rd = refine_distance if refine_distance is not None else plate_separation
        _add_sc_distance_field(
            plate_surfs,
            size_near=mesh_size_near,
            size_far=mesh_size_ff,
            dist_min=0.5 * mesh_size_near,
            dist_max=rd,
        )

    return _finish(output_file)


def elliptical_cavity(
    a: float,
    b: float,
    height: float,
    mesh_size_sc: float = 0.5e-3,
    mesh_size_near: float | None = None,
    refine_distance: float | None = None,
    output_file: str | Path = "cavity.msh",
    verbose: bool = False,
) -> str:
    """Closed elliptical cylindrical cavity carved out of bulk SC.

    The mesh's *air* volume is the interior of the cylinder with horizontal
    semi-axes ``a, b`` and total height ``height`` (z = 0 at the bottom).
    All bounding surfaces (floor, ceiling, side wall) are tagged as ``"SC"``;
    no ``"FF"`` group is created. ``load_msh`` then runs the pure-Neumann
    closed-trap branch (the solver pins one interior DOF).

    Used for benchmarks against trap geometries where the magnet is fully
    enclosed (e.g. Fuchs et al. 2024).
    """
    _start("cavity", verbose)
    occ = gmsh.model.occ

    cyl = occ.addCylinder(0, 0, 0, 0, 0, height, a)
    occ.dilate([(3, cyl)], 0, 0, 0, 1.0, b / a, 1.0)
    occ.synchronize()

    sc_surfs = list(set(occ.getSurfaceLoops(cyl)[1][0].tolist()))

    gmsh.model.addPhysicalGroup(2, sc_surfs, name="SC")
    gmsh.model.addPhysicalGroup(3, [cyl],    name="air")

    # Apply mesh size to all points of the cylinder
    pts = gmsh.model.getEntities(0)
    if pts:
        gmsh.model.mesh.setSize(pts, mesh_size_sc)

    if mesh_size_near is not None:
        rd = refine_distance if refine_distance is not None else min(a, b, height) / 2.0
        _add_sc_distance_field(
            sc_surfs,
            size_near=mesh_size_near,
            size_far=mesh_size_sc,
            dist_min=0.5 * mesh_size_near,
            dist_max=rd,
        )

    return _finish(output_file)


def rectangular_cavity(
    a: float,
    b: float,
    height: float,
    mesh_size_sc: float = 0.5e-3,
    mesh_size_near: float | None = None,
    refine_distance: float | None = None,
    output_file: str | Path = "rect_cavity.msh",
    verbose: bool = False,
) -> str:
    """Closed rectangular cuboid cavity carved out of bulk SC.

    The air domain is the box ``[-a, a] x [-b, b] x [0, height]``. All six
    bounding faces are tagged ``"SC"``; no far-field group is created.
    Useful for analytic image-method validation: a rectangular SC box has
    an exact, periodic-lattice image expansion for an interior dipole.
    """
    _start("rect_cavity", verbose)
    occ = gmsh.model.occ

    box = occ.addBox(-a, -b, 0, 2 * a, 2 * b, height)
    occ.synchronize()

    sc_surfs = list(set(occ.getSurfaceLoops(box)[1][0].tolist()))
    gmsh.model.addPhysicalGroup(2, sc_surfs, name="SC")
    gmsh.model.addPhysicalGroup(3, [box],    name="air")

    pts = gmsh.model.getEntities(0)
    if pts:
        gmsh.model.mesh.setSize(pts, mesh_size_sc)

    if mesh_size_near is not None:
        rd = refine_distance if refine_distance is not None else min(a, b, height) / 2.0
        _add_sc_distance_field(
            sc_surfs,
            size_near=mesh_size_near,
            size_far=mesh_size_sc,
            dist_min=0.5 * mesh_size_near,
            dist_max=rd,
        )

    return _finish(output_file)


def elliptical_cylinder(
    a: float,
    b: float,
    half_height: float,
    far_radius: float,
    mesh_size_sc: float = 1e-3,
    mesh_size_ff: float = 5e-3,
    mesh_size_near: float | None = None,
    refine_distance: float | None = None,
    output_file: str | Path = "ellipse.msh",
    verbose: bool = False,
) -> str:
    """Solid elliptical-cylinder SC inside a far-field sphere."""
    _start("ellipse", verbose)
    occ = gmsh.model.occ

    cyl = occ.addCylinder(0, 0, -half_height, 0, 0, 2 * half_height, a)
    occ.dilate([(3, cyl)], 0, 0, 0, 1.0, b / a, 1.0)
    sph = occ.addSphere(0, 0, 0, far_radius)

    air, _ = occ.cut([(3, sph)], [(3, cyl)], removeTool=False)
    occ.synchronize()

    air_tag  = air[0][1]
    sc_surfs = list(set(occ.getSurfaceLoops(cyl)[1][0].tolist()))
    all_surfs = [s[1] for s in gmsh.model.getEntities(2)]
    ff_surfs = [s for s in all_surfs if s not in sc_surfs]

    gmsh.model.addPhysicalGroup(2, sc_surfs, name="SC")
    gmsh.model.addPhysicalGroup(2, ff_surfs, name="FF")
    gmsh.model.addPhysicalGroup(3, [air_tag], name="air")

    _set_size_at_box((-a, -b, -half_height, a, b, half_height), mesh_size_sc)
    _set_size_at_radius(far_radius, mesh_size_ff)

    if mesh_size_near is not None:
        rd = refine_distance if refine_distance is not None else max(a, b, half_height)
        _add_sc_distance_field(
            sc_surfs,
            size_near=mesh_size_near,
            size_far=mesh_size_ff,
            dist_min=0.5 * mesh_size_near,
            dist_max=rd,
        )

    return _finish(output_file)


def hemispherical_bowl(
    radius: float,
    far_radius: float,
    mesh_size_sc: float = 1e-3,
    mesh_size_ff: float = 5e-3,
    mesh_size_near: float | None = None,
    refine_distance: float | None = None,
    output_file: str | Path = "bowl.msh",
    verbose: bool = False,
) -> str:
    """A hemispherical SC bowl (lower half-sphere) inside a far-field sphere."""
    _start("bowl", verbose)
    occ = gmsh.model.occ

    bowl = occ.addSphere(0, 0, 0, radius)
    cut  = occ.addBox(-2 * radius, -2 * radius, 0, 4 * radius, 4 * radius, 2 * radius)
    bowl_low, _ = occ.cut([(3, bowl)], [(3, cut)])

    sph = occ.addSphere(0, 0, 0, far_radius)
    air, _ = occ.cut([(3, sph)], bowl_low, removeTool=False)
    occ.synchronize()

    air_tag = air[0][1]
    sc_tags = []
    for dim, tag in bowl_low:
        sc_tags.extend(occ.getSurfaceLoops(tag)[1][0].tolist())
    sc_tags = list(set(sc_tags))
    all_surfs = [s[1] for s in gmsh.model.getEntities(2)]
    ff_surfs = [s for s in all_surfs if s not in sc_tags]

    gmsh.model.addPhysicalGroup(2, sc_tags,  name="SC")
    gmsh.model.addPhysicalGroup(2, ff_surfs, name="FF")
    gmsh.model.addPhysicalGroup(3, [air_tag], name="air")

    _set_size_at_radius(far_radius, mesh_size_ff)

    if mesh_size_near is not None:
        rd = refine_distance if refine_distance is not None else radius
        _add_sc_distance_field(
            sc_tags,
            size_near=mesh_size_near,
            size_far=mesh_size_ff,
            dist_min=0.5 * mesh_size_near,
            dist_max=rd,
        )

    return _finish(output_file)


# ---------------------------------------------------------------------------
# CAD import
# ---------------------------------------------------------------------------

# OpenCASCADE-readable solid formats (gmsh `occ.importShapes`).
_OCC_CAD_SUFFIXES = {".step", ".stp", ".iges", ".igs", ".brep"}
# Faceted-surface formats (gmsh `merge` + reclassification).
_STL_SUFFIXES = {".stl"}


def import_cad(
    cad_file: str | Path,
    far_radius: float | None = None,
    mesh_size_sc: float = 1e-3,
    mesh_size_ff: float | None = None,
    mesh_size_near: float | None = None,
    refine_distance: float | None = None,
    scale: float = 1.0,
    classify_angle_deg: float = 40.0,
    output_file: str | Path = "cad_trap.msh",
    verbose: bool = False,
) -> str:
    """Build a trap mesh from a CAD file of the superconductor.

    The imported solid is treated as the superconductor (the magnet levitates
    in the air *outside* it). The solid is embedded in a far-field sphere; the
    air domain is the region between the solid surface and the sphere. Physical
    groups written: ``"SC"`` (solid surface, Neumann), ``"FF"`` (far-field
    boundary, Dirichlet ``Φ = 0``), ``"air"`` (the volume). The resulting
    ``.msh`` is consumed verbatim by :func:`sctrap.mesh.load_msh`.

    Supported formats
    -----------------
    - **STEP / IGES / BREP** (``.step .stp .iges .igs .brep``) — imported as a
      solid through OpenCASCADE and meshed with a robust boolean cut. STEP is
      the recommended format: every major CAD package exports it.
    - **STL** (``.stl``) — *not yet supported*. STL is a faceted mesh format
      and gmsh's reconstruction of a meshable solid from arbitrary STL is not
      robust; ``import_cad`` raises ``NotImplementedError`` pointing you to
      STEP. (Tracked for a future release.)

    Parameters
    ----------
    cad_file
        Path to the CAD file.
    far_radius
        Radius of the far-field sphere [m]. Defaults to ``5 ×`` the solid's
        half bounding-box diagonal, centred on the solid.
    mesh_size_sc
        Target element size on the SC surface [m].
    mesh_size_ff
        Target element size on the far-field boundary [m]. Defaults to
        ``far_radius / 6``.
    mesh_size_near, refine_distance
        Optional local refinement near the SC surface; see
        :func:`_add_sc_distance_field`.
    scale
        Multiply all imported coordinates by this factor, e.g. ``1e-3`` to
        convert a CAD file authored in millimetres to SI metres. All of
        ``sctrap`` works in metres.
    classify_angle_deg
        Reserved for future STL support; currently unused.
    output_file
        Where to write the ``.msh``.
    verbose
        Forward gmsh's terminal output.

    Returns
    -------
    str
        The path to the written ``.msh`` file.
    """
    cad_file = Path(cad_file)
    if not cad_file.exists():
        raise FileNotFoundError(f"CAD file not found: {cad_file}")
    suffix = cad_file.suffix.lower()

    if suffix in _OCC_CAD_SUFFIXES:
        return _import_cad_occ(
            cad_file, far_radius, mesh_size_sc, mesh_size_ff,
            mesh_size_near, refine_distance, scale, output_file, verbose,
        )
    if suffix in _STL_SUFFIXES:
        raise NotImplementedError(
            "STL import is not supported yet. STL is a faceted mesh format, "
            "and gmsh's reconstruction of a meshable solid from arbitrary STL "
            "is not robust enough to ship. Please export your geometry as "
            "STEP (.step) — every major CAD package (FreeCAD, SolidWorks, "
            "Fusion 360, Inventor, Onshape) writes STEP, and sctrap meshes it "
            "robustly. (Tracked for a future release.)"
        )
    raise ValueError(
        f"Unsupported CAD format '{suffix}'. Supported: "
        f"{sorted(_OCC_CAD_SUFFIXES)} (STEP recommended)."
    )


def _solid_bbox_centre_halfextent():
    """(centre, half_diagonal_extent) of all model entities, after sync."""
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    centre = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    half = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    return centre, half


def _import_cad_occ(
    cad_file, far_radius, mesh_size_sc, mesh_size_ff,
    mesh_size_near, refine_distance, scale, output_file, verbose,
) -> str:
    _start("cad_occ", verbose)
    occ = gmsh.model.occ

    ents = occ.importShapes(str(cad_file))
    vols = [e for e in ents if e[0] == 3]
    if not vols:
        gmsh.model.remove()
        raise ValueError(
            f"{cad_file.name}: no solid volume found on import "
            f"(got entities {ents}). The CAD file must contain a closed solid."
        )
    occ.synchronize()

    # Mesh in the CAD file's native units, then scale the finished node
    # coordinates by `scale` (geometry-level dilation of imported shapes is
    # unreliable in gmsh). All user-facing sizes/radii are in the final
    # (post-scale, SI) units, so convert them into native units here.
    inv = 1.0 / scale
    centre, half = _solid_bbox_centre_halfextent()         # native units
    far_radius_n = 5.0 * half if far_radius is None else far_radius * inv
    size_ff_n = far_radius_n / 6.0 if mesh_size_ff is None else mesh_size_ff * inv
    size_sc_n = mesh_size_sc * inv

    sph = occ.addSphere(centre[0], centre[1], centre[2], far_radius_n)
    air, _ = occ.cut([(3, sph)], vols, removeTool=False)
    occ.synchronize()
    air_tag = air[0][1]

    sc_surfs: list[int] = []
    for _, t in vols:
        sc_surfs += occ.getSurfaceLoops(t)[1][0].tolist()
    sc_surfs = sorted(set(sc_surfs))
    all_surfs = [s[1] for s in gmsh.model.getEntities(2)]
    ff_surfs = [s for s in all_surfs if s not in sc_surfs]

    gmsh.model.addPhysicalGroup(2, sc_surfs, name="SC")
    gmsh.model.addPhysicalGroup(2, ff_surfs, name="FF")
    gmsh.model.addPhysicalGroup(3, [air_tag], name="air")

    # SC-surface element size: set on the points bounding the solid.
    cx, cy, cz = centre
    _set_size_at_box((cx - half, cy - half, cz - half,
                      cx + half, cy + half, cz + half), size_sc_n)
    _set_size_at_radius(far_radius_n, size_ff_n)

    if mesh_size_near is not None:
        size_near_n = mesh_size_near * inv
        rd = half if refine_distance is None else refine_distance * inv
        _add_sc_distance_field(
            sc_surfs, size_near=size_near_n, size_far=size_ff_n,
            dist_min=0.5 * size_near_n, dist_max=rd,
        )

    gmsh.model.mesh.generate(3)
    if scale != 1.0:
        gmsh.model.mesh.affineTransform(
            [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0]
        )
    out = str(output_file)
    gmsh.write(out)
    gmsh.model.remove()
    return out


# ---------------------------------------------------------------------------
# Mesh-size helpers
# ---------------------------------------------------------------------------

def _set_size_at_box(bbox, size):
    x0, y0, z0, x1, y1, z1 = bbox
    pad = 1e-6
    pts = gmsh.model.getEntitiesInBoundingBox(
        x0 - pad, y0 - pad, z0 - pad,
        x1 + pad, y1 + pad, z1 + pad, 0,
    )
    if pts:
        gmsh.model.mesh.setSize(pts, size)


def _set_size_at_radius(radius, size, frac=0.9):
    pts = []
    for p in gmsh.model.getEntities(0):
        if np.linalg.norm(gmsh.model.getValue(0, p[1], [])) > frac * radius:
            pts.append(p)
    if pts:
        gmsh.model.mesh.setSize(pts, size)
