"""FastAPI app: a local-only browser UI for sctrap mesh generation.

API contract
------------
POST /api/mesh
    body: MeshRequest (see schema below)
    returns: MeshResponse with a `token` referencing a saved .msh file
             and a slim payload of boundary triangles for client rendering.

GET  /api/mesh/{token}/download
    returns: the on-disk .msh file as application/octet-stream.

The schema uses a `trap_kind` discriminator so we can grow into open traps
(with a Far-Field surface and surface tagging) without breaking the v1 shape.

Concurrency: gmsh keeps process-global state and is not thread-safe; an
asyncio.Lock serialises calls.
"""
from __future__ import annotations

import asyncio
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Importing gmsh-using modules is deferred until first request so the
# server can start up without gmsh installed (useful for static asset
# serving + giving a clear error if gmsh is missing).


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ClosedEllipticalRequest(BaseModel):
    trap_kind: Literal["closed_elliptical"] = "closed_elliptical"
    a: float = Field(..., gt=0, le=50e-3,
                     description="Semi-axis along x [m]")
    b: float = Field(..., gt=0, le=50e-3,
                     description="Semi-axis along y [m]")
    height: float = Field(..., gt=0, le=50e-3,
                          description="Cavity height [m]")
    mesh_size: float = Field(..., gt=0, le=10e-3,
                             description="Bulk tet edge target [m]")
    mesh_size_near: float | None = Field(
        None, gt=0, le=10e-3,
        description="Tet edge near SC walls [m]; None for uniform")
    refine_distance: float | None = Field(
        None, gt=0, le=50e-3,
        description="Distance over which the size grades from "
                    "mesh_size_near to mesh_size [m]")


# Future: OpenEllipticalRequest with `far_field_radius` etc.
MeshRequest = ClosedEllipticalRequest


class SurfaceFacets(BaseModel):
    """Slim Three.js-friendly geometry for a tagged surface group."""
    name: str                            # "SC", "FF", ...
    color: str                           # hex, "#3171b5"
    n_facets: int
    indices: list[int]                   # flat triangle index buffer (3 * n_facets)


class MeshResponse(BaseModel):
    token: str
    n_pts: int
    n_tets: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    positions: list[float]               # flat XYZ, length 3 * n_pts (in mm for Three.js)
    surfaces:  list[SurfaceFacets]       # one entry per physical surface group
    timing_s:  float                     # gmsh wall time
    msh_bytes: int


# ---------------------------------------------------------------------------
# Mesh generation + parsing
# ---------------------------------------------------------------------------

# Cap to avoid OOM
MAX_TETS = 2_000_000
SURFACE_COLORS = {"SC": "#3171b5", "FF": "#d6604d"}


def _estimate_tet_count(req: ClosedEllipticalRequest) -> int:
    h_min = min(req.mesh_size, req.mesh_size_near or req.mesh_size)
    vol   = np.pi * req.a * req.b * req.height
    return int(vol / (h_min ** 3) * 6.0)        # ~6 tets per cube of side h


def _generate_closed_elliptical(req: ClosedEllipticalRequest, out_dir: Path) -> Path:
    """Run gmsh and return the path to the .msh file."""
    from sctrap.generators import elliptical_cavity        # local import
    out = out_dir / "cavity.msh"
    elliptical_cavity(
        req.a, req.b, req.height,
        mesh_size_sc    = req.mesh_size,
        mesh_size_near  = req.mesh_size_near,
        refine_distance = req.refine_distance,
        output_file     = str(out), verbose = False,
    )
    return out


def _read_mesh_payload(msh_path: Path) -> dict:
    """Load .msh with meshio and extract a Three.js-ready payload."""
    import meshio
    m = meshio.read(str(msh_path))

    pts = np.asarray(m.points, dtype=np.float64)            # (N, 3) in metres
    n_pts = pts.shape[0]

    # tetra count
    n_tets = 0
    for blk in m.cells:
        if blk.type in ("tetra", "tetra4", "tetra10"):
            n_tets += blk.data.shape[0]

    bbox_min = pts.min(axis=0).tolist()
    bbox_max = pts.max(axis=0).tolist()

    # Triangle blocks (boundary). meshio stores physical-tag info in
    # `cell_sets` / `cell_data["gmsh:physical"]`. We map physical tag ids back
    # to names via field_data.
    phys_id_to_name: dict[int, str] = {}
    for name, (tag, dim) in (m.field_data or {}).items():
        if dim == 2:
            phys_id_to_name[int(tag)] = name

    triangles_by_group: dict[str, list[list[int]]] = {}
    physical_arrays = m.cell_data.get("gmsh:physical")

    for blk_idx, blk in enumerate(m.cells):
        if blk.type not in ("triangle", "triangle3", "triangle6"):
            continue
        tris = blk.data[:, :3]                               # take corner verts only
        if physical_arrays is not None and blk_idx < len(physical_arrays):
            tags = np.asarray(physical_arrays[blk_idx], dtype=np.int64)
            for tag in np.unique(tags):
                name = phys_id_to_name.get(int(tag), f"phys_{tag}")
                mask = tags == tag
                triangles_by_group.setdefault(name, []).extend(
                    tris[mask].tolist())
        else:
            triangles_by_group.setdefault("untagged", []).extend(tris.tolist())

    # Convert millimetre scale for the front-end (less floating-point glitter)
    positions_mm = (pts * 1e3).flatten().tolist()

    surfaces: list[dict] = []
    for name, tris in triangles_by_group.items():
        flat = np.asarray(tris, dtype=np.int64).flatten().tolist()
        surfaces.append({
            "name":     name,
            "color":    SURFACE_COLORS.get(name, "#888888"),
            "n_facets": len(tris),
            "indices":  flat,
        })

    return {
        "n_pts":     int(n_pts),
        "n_tets":    int(n_tets),
        "bbox_min":  [b * 1e3 for b in bbox_min],            # mm
        "bbox_max":  [b * 1e3 for b in bbox_max],            # mm
        "positions": positions_mm,
        "surfaces":  surfaces,
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="sctrap mesh studio", version="0.1")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# token -> mesh path (in a temp dir under STORAGE_ROOT)
_STORAGE_ROOT = Path(tempfile.gettempdir()) / "sctrap-web-meshes"
_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
_TOKEN_TO_PATH: dict[str, Path] = {}
_TOKEN_AGE: dict[str, float] = {}
_TOKEN_TTL_S = 60 * 60                                        # 1 h
_GMSH_LOCK = asyncio.Lock()


def _gc_old_tokens() -> None:
    now = time.time()
    expired = [t for t, age in _TOKEN_AGE.items() if now - age > _TOKEN_TTL_S]
    for t in expired:
        path = _TOKEN_TO_PATH.pop(t, None)
        _TOKEN_AGE.pop(t, None)
        if path is not None:
            shutil.rmtree(path.parent, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/api/mesh", response_model=MeshResponse)
async def make_mesh(req: ClosedEllipticalRequest) -> MeshResponse:
    if req.mesh_size_near is not None and req.mesh_size_near >= req.mesh_size:
        raise HTTPException(400, "mesh_size_near must be < mesh_size")

    est = _estimate_tet_count(req)
    if est > MAX_TETS:
        raise HTTPException(
            400,
            f"Predicted ~{est:,} tets exceeds cap of {MAX_TETS:,}. "
            "Increase mesh_size or relax the near-SC refinement.")

    _gc_old_tokens()
    token = secrets.token_urlsafe(8)
    work_dir = _STORAGE_ROOT / token
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # gmsh installs signal handlers and only works on the main thread, so we
    # run it synchronously under the lock rather than in an executor.
    try:
        async with _GMSH_LOCK:
            msh_path = _generate_closed_elliptical(req, work_dir)
            payload = _read_mesh_payload(msh_path)
    except ImportError as exc:
        raise HTTPException(
            500, f"gmsh not available on the server: {exc}") from exc
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(500, f"gmsh failed: {exc}") from exc
    timing = time.time() - t0

    _TOKEN_TO_PATH[token] = msh_path
    _TOKEN_AGE[token]     = time.time()

    return MeshResponse(
        token = token,
        timing_s = timing,
        msh_bytes = msh_path.stat().st_size,
        **payload,
    )


@app.get("/api/mesh/{token}/download")
async def download(token: str = PathParam(..., min_length=4, max_length=64)):
    path = _TOKEN_TO_PATH.get(token)
    if path is None or not path.exists():
        raise HTTPException(404, "unknown or expired mesh token")
    return FileResponse(
        path = path,
        media_type = "application/octet-stream",
        filename = "cavity.msh",
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Console-script entrypoint: `sctrap-serve`."""
    import uvicorn
    print(f"\n  sctrap mesh studio  ->  http://{host}:{port}\n")
    uvicorn.run("sctrap.web.app:app", host=host, port=port,
                reload=reload, log_level="info")
