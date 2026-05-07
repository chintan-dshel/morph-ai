"""
MorphAI — FastAPI backend
Wraps the existing optimizer / materials / geometry pipeline and serves the React frontend.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from materials import MATERIALS
from optimizer import simp_core, build_filter, build_load_cases
from geometry import to_stl_bytes
from utils import estimate_print

# ── Storage paths ──────────────────────────────────────────────────────────────
MORPHAI_DIR = Path.home() / ".morphai"
HISTORY_FILE = MORPHAI_DIR / "history.json"
STL_DIR = MORPHAI_DIR / "stls"


def _ensure_dirs():
    MORPHAI_DIR.mkdir(parents=True, exist_ok=True)
    STL_DIR.mkdir(parents=True, exist_ok=True)


def _load_history() -> dict:
    _ensure_dirs()
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"runs": []}


def _save_history(data: dict):
    _ensure_dirs()
    HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Material name mapping (frontend short → backend full key) ──────────────────
_MAT_MAP = {
    "PLA":            "PLA (Bioplastic)",
    "PETG":           "PETG (Engineering Plastic)",
    "ABS":            "ABS (Acrylonitrile Butadiene Styrene)",
    "Nylon":          "Nylon PA12",
    "CF-PETG":        "Carbon Fiber PETG (Composite)",
    "Aluminum (ref)": "Aluminum 6061 (Reference)",
    "Titanium (ref)": "Titanium Ti-6Al-4V (Reference)",
}

# ── Face name mapping (frontend short → backend full) ─────────────────────────
_FACE_MAP = {
    "Left":   "Left (X=0)",
    "Right":  "Right (X=W)",
    "Top":    "Top (Y=H)",
    "Bottom": "Bottom (Y=0)",
    "Front":  "Front (Z=0)",
    "Back":   "Back (Z=D)",
}

# ── Force direction unit vectors (frontend dir → (fx, fy)) ────────────────────
_DIR_UNIT = {
    "-X": (-1.0,  0.0),
    "+X": ( 1.0,  0.0),
    "-Y": ( 0.0, -1.0),
    "+Y": ( 0.0,  1.0),
    "-Z": ( 0.0,  0.0),
    "+Z": ( 0.0,  0.0),
}


# ── Metrics helpers ────────────────────────────────────────────────────────────

def _compute_bimodality(xPhys: np.ndarray) -> float:
    flat = xPhys.flatten()
    return float(((flat < 0.2) | (flat > 0.8)).sum() / len(flat))


def _check_load_path(xPhys: np.ndarray, fixed_face: str, load_face: str) -> bool:
    """BFS flood-fill: is there a solid-element path between the two faces?"""
    nely, nelx = xPhys.shape
    solid = xPhys > 0.45

    def _face_elements(face: str) -> set:
        s: set = set()
        if "Left" in face:
            for iy in range(nely):
                if solid[iy, 0]: s.add((iy, 0))
        elif "Right" in face:
            for iy in range(nely):
                if solid[iy, nelx - 1]: s.add((iy, nelx - 1))
        elif "Bottom" in face:
            for ix in range(nelx):
                if solid[0, ix]: s.add((0, ix))
        elif "Top" in face:
            for ix in range(nelx):
                if solid[nely - 1, ix]: s.add((nely - 1, ix))
        return s

    seeds = _face_elements(fixed_face)
    targets = _face_elements(load_face)

    if not seeds or not targets:
        return False
    if seeds & targets:
        return True

    visited = set(seeds)
    frontier = list(seeds)
    while frontier:
        iy, ix = frontier.pop()
        if (iy, ix) in targets:
            return True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = iy + dy, ix + dx
            if (0 <= ny < nely and 0 <= nx < nelx
                    and solid[ny, nx] and (ny, nx) not in visited):
                visited.add((ny, nx))
                frontier.append((ny, nx))
    return False


def _check_non_convergent(history: list[float], threshold: float = 8) -> bool:
    """Return True if compliance stopped improving for the last `threshold` iters."""
    if len(history) < threshold + 2:
        return False
    recent = history[-int(threshold):]
    span = max(recent) - min(recent)
    reference = abs(history[0]) if history[0] else 1.0
    return span < reference * 0.001


# ── Fidelity presets ───────────────────────────────────────────────────────────
_FIDELITY = {
    "quick":    {"nelx_base": 14, "max_iter": 40},
    "standard": {"nelx_base": 22, "max_iter": 60},
    "detail":   {"nelx_base": 34, "max_iter": 80},
}


# ── Request / response models ──────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    material:  str   = "PLA"
    w:         float = 100.0   # mm
    h:         float = 60.0
    d:         float = 30.0
    fixedFace: str   = "Left"
    loadFace:  str   = "Right"
    forceDir:  str   = "-Y"
    force:     float = 150.0   # N
    vf:        float = 0.40
    sf:        float = 2.0
    fidelity:  str   = "standard"
    infill:    str   = "Gyroid"


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MorphAI API", version="1.0")


@app.get("/api/materials")
def get_materials():
    """Return material catalogue using frontend short names."""
    result = {}
    for short, full in _MAT_MAP.items():
        m = MATERIALS.get(full)
        if m is None:
            continue
        result[short] = {
            "E":         m["E_gpa"],
            "rho":       m["rho_gcc"],
            "yield":     m["yield_strength_mpa"],
            "color":     m["color"],
            "printable": m["printable"],
            "temp":      m["print_temp"],
            "cost":      round(m["cost_per_kg"] / 1000, 4),  # per gram
            "desc":      m.get("notes", ""),
        }
    return result


@app.post("/api/optimize")
def run_optimize(req: OptimizeRequest):
    # ── Resolve material ───────────────────────────────────────────────────────
    mat_full = _MAT_MAP.get(req.material, "PLA (Bioplastic)")
    mat = MATERIALS.get(mat_full)
    if mat is None:
        raise HTTPException(400, f"Unknown material: {req.material}")

    nu = mat["nu"]

    # ── Mesh resolution from fidelity ─────────────────────────────────────────
    fid = _FIDELITY.get(req.fidelity, _FIDELITY["standard"])
    nelx = fid["nelx_base"]
    nely = max(6, round(nelx * req.h / req.w))
    max_iter = fid["max_iter"]

    # ── Build filter ───────────────────────────────────────────────────────────
    H, Hs = build_filter(nelx, nely, rmin=1.5)

    # ── Build load/support specs ───────────────────────────────────────────────
    fixed_face_full = _FACE_MAP.get(req.fixedFace, "Left (X=0)")
    load_face_full  = _FACE_MAP.get(req.loadFace,  "Right (X=W)")
    fx_unit, fy_unit = _DIR_UNIT.get(req.forceDir, (0.0, -1.0))

    load_specs    = [{"type": "surface", "face": load_face_full,
                      "fx": fx_unit * req.force, "fy": fy_unit * req.force}]
    support_specs = [{"type": "fixed", "face": fixed_face_full}]

    load_cases, fixed_dofs, F_mag = build_load_cases(load_specs, support_specs, nelx, nely)

    # ── Run SIMP ───────────────────────────────────────────────────────────────
    xPhys, history, stress_field = simp_core(
        nelx=nelx, nely=nely,
        volfrac=req.vf,
        nu=nu,
        penal=3.0,
        H=H, Hs=Hs,
        max_iter=max_iter,
        load_cases=load_cases,
        fixed_dofs=fixed_dofs,
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    bimodality = _compute_bimodality(xPhys)
    non_convergent = _check_non_convergent(history)
    has_load_path = _check_load_path(xPhys, fixed_face_full, load_face_full)

    # Physical stress: σ_phys [MPa] = σ_norm × F_mag [N] / (dx_mm × thickness_mm)
    dx_mm = req.w / nelx
    sigma_phys_max = float(stress_field.max()) * F_mag / (dx_mm * req.d)
    yield_exceeded = sigma_phys_max * req.sf > mat["yield_strength_mpa"]

    # Mass: volume × vf × density
    mass_g = round(float(req.w * req.h * req.d * req.vf * mat["rho_gcc"]) / 1000.0, 2)

    # Print time estimate
    print_est = estimate_print(
        mass_g=mass_g,
        rho_gcc=mat["rho_gcc"],
        cost_per_kg=mat.get("cost_per_kg", 20.0),
    )

    # ── STL ────────────────────────────────────────────────────────────────────
    stl_bytes, _n_faces = to_stl_bytes(xPhys, req.w, req.h, req.d, iso=0.45)

    run_id = str(uuid.uuid4())[:8]
    _ensure_dirs()
    stl_available = False
    if stl_bytes:
        (STL_DIR / f"{run_id}.stl").write_bytes(stl_bytes)
        stl_available = True

    # ── Persist thin run record (no large arrays) ─────────────────────────────
    run_meta = {
        "id":            run_id,
        "label":         f"{req.material} · {int(req.w)}×{int(req.h)}×{int(req.d)}",
        "material":      req.material,
        "w": req.w, "h": req.h, "d": req.d,
        "fixedFace":     req.fixedFace,
        "loadFace":      req.loadFace,
        "forceDir":      req.forceDir,
        "force":         req.force,
        "vf":            req.vf,
        "sf":            req.sf,
        "fidelity":      req.fidelity,
        "infill":        req.infill,
        "compliance":    round(float(history[-1]), 3) if history else 0.0,
        "max_stress":    round(sigma_phys_max, 2),
        "iterations":    len(history),
        "mass_g":        mass_g,
        "bimodality":    round(bimodality, 3),
        "stl_available": stl_available,
    }
    hist = _load_history()
    hist["runs"].insert(0, run_meta)
    hist["runs"] = hist["runs"][:20]
    _save_history(hist)

    # ── Full response (includes arrays for charts) ────────────────────────────
    return {
        **run_meta,
        "non_convergent":  non_convergent,
        "no_load_path":    not has_load_path,
        "yield_exceeded":  yield_exceeded,
        "print_time_h":    print_est["time_h"],
        "filament_cost":   print_est["material_cost_usd"],
        "convergence":     [round(v, 3) for v in history],
        "density_grid":    xPhys.tolist(),   # list[list[float]], shape nely × nelx
    }


@app.get("/api/runs")
def get_runs():
    return _load_history()


@app.get("/api/runs/{run_id}/stl")
def download_stl(run_id: str):
    stl_path = STL_DIR / f"{run_id}.stl"
    if not stl_path.exists():
        raise HTTPException(404, "STL not found — run the optimizer first")
    fname = f"morphai_{run_id}.stl"
    return Response(
        content=stl_path.read_bytes(),
        media_type="model/stl",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── Serve React frontend (must be last — catches everything else) ──────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
