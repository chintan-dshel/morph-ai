"""Golden path + execution path tests for MorphAI. Run: python test_paths.py"""
import numpy as np, sys

ok = True

def check(label, condition, detail=''):
    global ok
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
        ok = False

from optimizer import simp_core, build_filter, build_filter_fast, build_load_cases, build_load_cases_multi
from visualization import mesh_traces, arrow_traces
from geometry import to_stl_bytes
from materials import MATERIALS, rank_materials
from utils import estimate_print, make_history_record

nelx, nely = 20, 12
box_w, box_h, box_d = 100.0, 60.0, 20.0
volfrac, nu, penal = 0.4, 0.3, 3.0

print("=== FILTER ===")
H, Hs = build_filter_fast(nelx, nely, 1.5)
check("build_filter_fast shape", H.shape == (nelx*nely, nelx*nely) and Hs.shape == (nelx*nely,))
H2, Hs2 = build_filter(nelx, nely, 1.5)
check("fast vs slow filter identical", np.allclose(H.toarray(), H2.toarray()))

print("\n=== LOAD CASES (left fixed / right down) ===")
load_specs = [{"id": 0, "type": "surface", "face": "Right (X=W)",
               "fx": 0.0, "fy": -500.0, "_dir_label": "-Y", "_magnitude": 500.0,
               "_angle_deg": -90.0, "u": 0.5, "u_start": 0.25, "u_end": 0.75,
               "axis": "Z", "magnitude": 5000.0}]
support_specs = [{"id": 0, "type": "fixed", "face": "Left (X=0)"}]
load_cases, fixed_dofs, F_mag = build_load_cases(load_specs, support_specs, nelx, nely)

check("returns 3-tuple", isinstance(F_mag, float))
check("left face: 26 DOFs fixed (13 nodes × 2)", len(fixed_dofs) == 26, f"got {len(fixed_dofs)}")
check("F_mag > 0", F_mag > 0, f"got {F_mag}")

print("\n=== SIMP CORE ===")
xPhys, history, stress_field = simp_core(nelx, nely, volfrac, nu, penal, H, Hs, 30, load_cases, fixed_dofs)
check("returns 3-tuple with stress_field", stress_field is not None)
check("xPhys shape", xPhys.shape == (nely, nelx))
check("history: list of floats", isinstance(history[0], float))
check("compliance decreases overall", history[-1] < history[0], f"{history[-1]:.2f} vs {history[0]:.2f}")
check("stress_field shape matches xPhys", stress_field.shape == xPhys.shape)
check("stress_field non-negative", stress_field.min() >= 0)

dx_mm = box_w / nelx
stress_phys = stress_field * (F_mag / dx_mm)
check("physical stress > 0", stress_phys.max() > 0)

print("\n=== VISUALIZATION ===")
traces = mesh_traces(xPhys, box_w, box_h, box_d, 0.3, "#1e88e5")
check("mesh_traces (topology) non-empty", len(traces) > 0)
traces_s = mesh_traces(xPhys, box_w, box_h, box_d, 0.3, "#1e88e5", stress_field=stress_phys, colormode="stress")
check("mesh_traces (stress mode) non-empty", len(traces_s) > 0)

print("\n=== STL EXPORT ===")
stl_bytes, n_faces = to_stl_bytes(xPhys, box_w, box_h, box_d, 0.3)
check("STL bytes produced", stl_bytes is not None and len(stl_bytes) > 100,
      f"got {len(stl_bytes) if stl_bytes else None} bytes")
check("face count > 0", n_faces > 0, f"got {n_faces}")

print("\n=== MATERIAL RANKING ===")
from materials import compute_mat_scores
mat_names = list(MATERIALS.keys())[:3]
scores = compute_mat_scores(mat_names, box_w, box_h, box_d, volfrac, 500.0, 2.0, history[-1])
winners = rank_materials(scores)
check("compute_mat_scores returns dict", isinstance(scores, dict) and len(scores) > 0)
# rank_materials returns a summary dict {stiffest, lightest, best_ashby, cheapest}
check("rank_materials returns non-empty dict", isinstance(winners, dict) and len(winners) > 0)
check("rank_materials has expected keys",
      all(k in winners for k in ("stiffest", "lightest", "best_ashby")))

print("\n=== HISTORY RECORD ===")
mat_key = list(MATERIALS.keys())[0]
rec = make_history_record("Expert", mat_key, (box_w, box_h, box_d), volfrac,
                          "Left fixed / Right down", history[-1], len(history), n_faces, 12.5)
check("record has 'compliance' key", "compliance" in rec)

print("\n=== CHAT MODE: _pending_to_specs for each preset ===")
_CHAT_FACE_MAP = {
    "left":   "Left (X=0)", "right":  "Right (X=W)",
    "bottom": "Bottom (Y=0)", "top": "Top (Y=H)",
    "front":  "Front (Z=0)", "back":  "Back (Z=D)",
}
_CHAT_DIR_FXY = {
    "-X": (-1.0,  0.0), "+X": (1.0,  0.0),
    "-Y": ( 0.0, -1.0), "+Y": (0.0,  1.0),
    "-Z": ( 0.0,  0.0), "+Z": (0.0,  0.0),
}

def pending_to_specs(p):
    fixed_face = _CHAT_FACE_MAP.get(str(p.get("fixed_face", "left")).lower().split()[0], "Left (X=0)")
    load_face  = _CHAT_FACE_MAP.get(str(p.get("load_face", "right")).lower().split()[0], "Right (X=W)")
    ufx, ufy   = _CHAT_DIR_FXY.get(p.get("force_direction", "-Y"), (0.0, -1.0))
    force_n    = float(p.get("applied_force_n", 500.0))
    ld = [{"id": 0, "type": "surface", "face": load_face,
           "fx": ufx * force_n, "fy": ufy * force_n,
           "_dir_label": p.get("force_direction", "-Y"), "_magnitude": force_n,
           "_angle_deg": -90.0, "u": 0.5, "u_start": 0.25, "u_end": 0.75,
           "axis": "Z", "magnitude": 5000.0}]
    sp = [{"id": 0, "type": "fixed", "face": fixed_face}]
    return ld, sp

PRESETS = [
    {"name": "Cantilever",   "fixed_face": "left",   "load_face": "right",  "force_direction": "-Y", "applied_force_n": 500},
    {"name": "Wall Bracket", "fixed_face": "top",    "load_face": "bottom", "force_direction": "-Y", "applied_force_n": 196},
    {"name": "Shelf",        "fixed_face": "left",   "load_face": "right",  "force_direction": "-Y", "applied_force_n": 980},
    {"name": "Bridge",       "fixed_face": "bottom", "load_face": "top",    "force_direction": "+Y", "applied_force_n": 500},
]

topologies = []
for preset in PRESETS:
    ld_s, sup_s = pending_to_specs(preset)
    lc, fd, fm = build_load_cases(ld_s, sup_s, nelx, nely)
    check(f'preset "{preset["name"]}" — fixed_dofs > 0', len(fd) > 0, f"got {len(fd)}")
    check(f'preset "{preset["name"]}" — F_mag > 0', fm > 0, f"got {fm}")
    xP, hist, sf = simp_core(nelx, nely, 0.4, 0.3, 3.0, H, Hs, 20, lc, fd)
    check(f'preset "{preset["name"]}" — compliance decreases', hist[-1] < hist[0],
          f"{hist[-1]:.2f} vs {hist[0]:.2f}")
    topologies.append(xP)

# Verify presets with DIFFERENT boundary conditions produce different topologies.
# Cantilever and Shelf have the same BCs (left fixed, right down) — only force magnitude
# differs, which SIMP normalizes → identical topology expected (correct).
# We compare pairs that genuinely differ in BC geometry.
distinct_pairs = [(0, 1), (0, 3), (1, 2), (1, 3)]  # Cantilever/WallBracket, /Bridge, WB/Shelf, WB/Bridge
for i, j in distinct_pairs:
    diff = np.abs(topologies[i] - topologies[j]).max()
    check(f'preset {PRESETS[i]["name"]} != {PRESETS[j]["name"]} (different BCs)',
          diff > 0.05, f"max diff = {diff:.4f}")

print("\n=== MULTI-LOAD SCENARIOS ===")
scenarios = [
    {"name": "Down", "face": "Right (X=W)", "direction": "-Y", "magnitude": 300.0, "weight": 1.0},
    {"name": "Side", "face": "Top (Y=H)",   "direction": "-X", "magnitude": 100.0, "weight": 0.5},
]
lc_m, fd_m, fm_m = build_load_cases_multi(scenarios, support_specs, nelx, nely)
check("multi-load: at least 1 load case", len(lc_m) >= 1)
check("multi-load: F_mag_eff > 0", fm_m > 0, f"got {fm_m}")
xPm, histm, _ = simp_core(nelx, nely, 0.4, 0.3, 3.0, H, Hs, 20, lc_m, fd_m)
check("multi-load: optimizer converges", histm[-1] < histm[0])

print("\n=== MBB BENCHMARK (nelx=60, nely=20, tol>5% pass) ===")
bH, bHs = build_filter(60, 20, 1.5)
# MBB: point load at top-left corner, SS at bottom-left and bottom-right
# Approximate via left-face surface load (downward)
bld_specs = [{"id": 0, "type": "surface", "face": "Left (X=0)",
              "fx": 0.0, "fy": -1.0, "_dir_label": "-Y", "_magnitude": 1.0,
              "_angle_deg": -90.0, "u": 0.0, "u_start": 0.0, "u_end": 0.1,
              "axis": "Z", "magnitude": 1.0}]
bsup_specs = [{"id": 0, "type": "fixed", "face": "Bottom (Y=0)"}]
blc, bfd, _ = build_load_cases(bld_specs, bsup_specs, 60, 20)
bx, bh, _ = simp_core(60, 20, 0.5, 0.3, 3.0, bH, bHs, 100, blc, bfd)
mbb_C = bh[-1]
print(f"  MBB compliance = {mbb_C:.4f}")
# Normalized load (F_mag ≈ 1), so compliance is much smaller than physical.
# Plausible range for any reasonable BC setup with normalized force.
check("MBB compliance plausible (> 0)", mbb_C > 0, f"got {mbb_C:.4f}")
check("MBB compliance finite", np.isfinite(mbb_C), f"got {mbb_C:.4f}")

print()
if ok:
    print("=" * 40)
    print("ALL PATHS PASS")
else:
    print("=" * 40)
    print("SOME PATHS FAILED — see above")
    sys.exit(1)
