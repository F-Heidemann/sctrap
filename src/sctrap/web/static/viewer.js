/* sctrap mesh studio — viewer + controls */
console.log("viewer.js v3 loaded at", new Date().toISOString());
import * as THREE from "/static/three.module.min.js";
import { OrbitControls } from "/static/OrbitControls.js";

// ---- DOM ------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const inputs = {
    a:               $("a_mm"),
    b:               $("b_mm"),
    h:               $("height_mm"),
    mesh:            $("mesh_size_mm"),
    refine_on:       $("refine_on"),
    refine_block:    $("refine_block"),
    near:            $("mesh_size_near_mm"),
    refdist:         $("refine_distance_mm"),
    auto:            $("auto_update"),
};
const ui = {
    status:          $("status"),
    stat_pts:        $("stat_pts"),
    stat_tets:       $("stat_tets"),
    stat_sc:         $("stat_sc"),
    stat_ff:         $("stat_ff"),
    stat_bytes:      $("stat_bytes"),
    stat_time:       $("stat_time"),
    legend:          $("legend"),
    btn_generate:    $("generate_btn"),
    btn_download:    $("download_btn"),
    btn_snippet:     $("snippet_btn"),
    snippet_pre:     $("snippet_text"),
};
let currentToken = null;

inputs.refine_on.addEventListener("change", () => {
    inputs.refine_block.style.display = inputs.refine_on.checked ? "" : "none";
    scheduleUpdate();
});

// ---- Three.js scene -------------------------------------------------------
function step(label, fn) {
    try { return fn(); }
    catch (e) {
        const msg = `[init: ${label}] ${e.message || e}`;
        const s = document.getElementById("status");
        if (s) { s.textContent = msg; s.className = "error"; }
        console.error(msg, e);
        throw e;
    }
}

const canvas   = $("canvas");
const renderer = step("WebGLRenderer", () => new THREE.WebGLRenderer({ canvas, antialias: true }));
step("setPixelRatio", () => renderer.setPixelRatio(window.devicePixelRatio));

const scene = step("Scene", () => new THREE.Scene());
step("scene.background", () => { scene.background = new THREE.Color(0xf4f4f6); });

const camera = step("PerspectiveCamera", () => new THREE.PerspectiveCamera(45, 1, 0.001, 1000));
step("camera.position.set", () => camera.position.set(10, 8, 10));

const controls = step("OrbitControls", () => new OrbitControls(camera, renderer.domElement));
step("controls.enableDamping", () => { controls.enableDamping = true; });
step("controls.dampingFactor", () => { controls.dampingFactor = 0.08; });

step("ambient", () => scene.add(new THREE.AmbientLight(0xffffff, 0.55)));
const key  = step("dir-key",  () => { const l = new THREE.DirectionalLight(0xffffff, 0.65); l.position.set(5,10,7); scene.add(l); return l; });
const fill = step("dir-fill", () => { const l = new THREE.DirectionalLight(0xffffff, 0.25); l.position.set(-6,3,-5); scene.add(l); return l; });

const grid = step("GridHelper", () => { const g = new THREE.GridHelper(20, 20, 0xcccccc, 0xeeeeee); scene.add(g); return g; });

const meshGroup = step("meshGroup", () => { const g = new THREE.Group(); g.rotateX(-Math.PI / 2); scene.add(g); return g; });
step("axes", () => { const a = new THREE.AxesHelper(2); meshGroup.add(a); });

function fitCanvas() {
    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
}
window.addEventListener("resize", fitCanvas);
fitCanvas();

(function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
})();

// ---- Mesh rendering -------------------------------------------------------

function clearMesh() {
    while (meshGroup.children.length) {
        const obj = meshGroup.children.pop();
        obj.geometry?.dispose();
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material?.dispose();
    }
}

function buildMesh(payload) {
    clearMesh();
    const positions = new Float32Array(payload.positions);   // Nx3 already in mm
    const posAttr   = new THREE.BufferAttribute(positions, 3);

    let legend = "";
    for (const surf of payload.surfaces) {
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", posAttr);
        geom.setIndex(surf.indices);
        geom.computeVertexNormals();

        const baseColor = new THREE.Color(surf.color);
        const fillMat = new THREE.MeshLambertMaterial({
            color: baseColor, transparent: true, opacity: 0.45,
            side: THREE.DoubleSide, depthWrite: false,
        });
        const fillMesh = new THREE.Mesh(geom, fillMat);
        meshGroup.add(fillMesh);

        const wireMat = new THREE.LineBasicMaterial({
            color: baseColor.clone().multiplyScalar(0.55),
            transparent: true, opacity: 0.7,
        });
        const wire = new THREE.LineSegments(
            new THREE.WireframeGeometry(geom), wireMat);
        meshGroup.add(wire);

        legend += `<div><span class="swatch" style="background:${surf.color}"></span>`
               +  `${surf.name}: ${surf.n_facets.toLocaleString()} facets</div>`;
    }
    ui.legend.innerHTML = legend;

    fitView(payload.bbox_min, payload.bbox_max);
}

function fitView(bmin, bmax) {
    // bbox is in physics coordinates (x, y, z). The mesh group is rotated
    // by -90 deg about x, so physics (x, y, z) -> world (x, z, -y).
    const c_phys = [
        0.5 * (bmin[0] + bmax[0]),
        0.5 * (bmin[1] + bmax[1]),
        0.5 * (bmin[2] + bmax[2]),
    ];
    const dx = bmax[0] - bmin[0], dy = bmax[1] - bmin[1], dz = bmax[2] - bmin[2];
    const span = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const c_world = new THREE.Vector3(c_phys[0], c_phys[2], -c_phys[1]);
    controls.target.copy(c_world);
    const r = Math.max(span * 1.5, 5);
    camera.position.set(c_world.x + r * 0.9, c_world.y + r * 0.7, c_world.z + r * 0.6);
    camera.lookAt(c_world);
    controls.update();
}

// ---- API call -------------------------------------------------------------

function gatherRequest() {
    const req = {
        trap_kind: "closed_elliptical",
        a:          parseFloat(inputs.a.value)   * 1e-3,
        b:          parseFloat(inputs.b.value)   * 1e-3,
        height:     parseFloat(inputs.h.value)   * 1e-3,
        mesh_size:  parseFloat(inputs.mesh.value) * 1e-3,
    };
    if (inputs.refine_on.checked) {
        req.mesh_size_near  = parseFloat(inputs.near.value)    * 1e-3;
        req.refine_distance = parseFloat(inputs.refdist.value) * 1e-3;
    }
    return req;
}

function setStatus(text, cls) {
    ui.status.textContent = text;
    ui.status.className = cls || "idle";
}

let inflight = null;
async function generate() {
    if (inflight) inflight.abort();
    inflight = new AbortController();

    let req;
    try { req = gatherRequest(); }
    catch (e) { setStatus(e.message, "error"); return; }

    setStatus("meshing…", "busy");
    ui.btn_generate.disabled = true;

    try {
        const r = await fetch("/api/mesh", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(req),
            signal: inflight.signal,
        });
        if (!r.ok) {
            const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
            throw new Error(detail);
        }
        const payload = await r.json();
        buildMesh(payload);
        currentToken = payload.token;
        ui.stat_pts.textContent   = payload.n_pts.toLocaleString();
        ui.stat_tets.textContent  = payload.n_tets.toLocaleString();
        ui.stat_sc.textContent    = (payload.surfaces.find(s=>s.name==="SC")?.n_facets ?? 0).toLocaleString();
        ui.stat_ff.textContent    = (payload.surfaces.find(s=>s.name==="FF")?.n_facets ?? 0).toLocaleString();
        ui.stat_bytes.textContent = (payload.msh_bytes / 1024).toFixed(1) + " KB";
        ui.stat_time.textContent  = payload.timing_s.toFixed(2) + " s";
        ui.btn_download.disabled  = false;
        ui.btn_snippet.disabled   = false;
        setStatus(`ready — ${payload.n_tets.toLocaleString()} tets`, "ok");
    } catch (e) {
        if (e.name === "AbortError") return;
        setStatus("error: " + e.message, "error");
    } finally {
        ui.btn_generate.disabled = false;
        inflight = null;
    }
}

// ---- live updates ---------------------------------------------------------
let debounceId = null;
function scheduleUpdate() {
    if (!inputs.auto.checked) return;
    if (debounceId) clearTimeout(debounceId);
    debounceId = setTimeout(generate, 350);
}

[inputs.a, inputs.b, inputs.h, inputs.mesh, inputs.near, inputs.refdist].forEach(el => {
    el.addEventListener("input", scheduleUpdate);
});
inputs.auto.addEventListener("change", () => { if (inputs.auto.checked) scheduleUpdate(); });

ui.btn_generate.addEventListener("click", generate);
ui.btn_download.addEventListener("click", () => {
    if (!currentToken) return;
    const a = document.createElement("a");
    a.href = `/api/mesh/${currentToken}/download`;
    a.download = "cavity.msh";
    document.body.appendChild(a); a.click(); a.remove();
});

ui.btn_snippet.addEventListener("click", () => {
    const a = parseFloat(inputs.a.value);
    const b = parseFloat(inputs.b.value);
    const h = parseFloat(inputs.h.value);
    const ms = parseFloat(inputs.mesh.value);
    let extra = "MESH_SIZE_NEAR     = None\nMESH_REFINE_DIST   = None";
    if (inputs.refine_on.checked) {
        const near = parseFloat(inputs.near.value);
        const dist = parseFloat(inputs.refdist.value);
        extra = `MESH_SIZE_NEAR     = ${(near*1e-3).toExponential(3)}\n`
              + `MESH_REFINE_DIST   = ${(dist*1e-3).toExponential(3)}`;
    }
    const text = `# from sctrap mesh studio
CAVITY_A      = ${(a*1e-3).toExponential(3)}
CAVITY_B      = ${(b*1e-3).toExponential(3)}
CAVITY_HEIGHT = ${(h*1e-3).toExponential(3)}

MESH_SIZE          = ${(ms*1e-3).toExponential(3)}
${extra}
`;
    ui.snippet_pre.textContent = text;
    ui.snippet_pre.classList.add("visible");
    navigator.clipboard?.writeText(text).then(
        () => setStatus("snippet copied to clipboard", "ok"),
        () => setStatus("snippet shown below — copy manually", "ok"));
});

// initial run
window.addEventListener("load", () => setTimeout(generate, 100));
