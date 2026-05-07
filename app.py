import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from materials import (
    MATERIALS, FACES, INFILL_PATTERNS,
    compute_mat_scores, rank_materials,
)
import math
import gzip, base64, json
from optimizer import (
    simp_core, build_filter, build_filter_fast, make_load_cases,
    build_load_cases, build_load_cases_multi,
    simp_core_3d, build_filter_3d, build_load_cases_3d,
)
from geometry import generate_infill, to_stl_bytes, voxelize_mesh
from visualization import (
    mesh_traces, density_isosurface_traces,
    fem_surface_traces, fem_quality_stats,
    wireframe, design_space_box, solid_box_mesh,
    sculpt_solid_frames, element_mesh_traces,
    face_trace, face_center,
    arrow_traces, fixed_label_trace, before_after_traces,
    scene3d, rgba, norm_v,
)
from utils import estimate_print, make_history_record
try:
    from meshing import generate_fem_mesh, smooth_surface
    _HAS_GMSH = True
except Exception:
    _HAS_GMSH = False
    def generate_fem_mesh(*a, **kw): raise RuntimeError("gmsh not available in this environment")
    def smooth_surface(*a, **kw): raise RuntimeError("gmsh not available in this environment")
import chat as chat_module

# ─────────────────────────────────────────────────────────────
#  Page config & global CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MorphAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════
   MORPHAI  —  Global design tokens
   Background: #0e1117  Surface: #141820  Accent: #1565c0
═══════════════════════════════════════════════════════════ */

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    min-width: 380px !important; max-width: 420px !important;
    background: #0c0f16 !important;
}
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stRadio      { margin-bottom: 0.15rem !important; }
[data-testid="stSidebar"] label         { font-size: 0.79rem !important; margin-bottom: 0 !important; }
[data-testid="stSidebar"] .stExpander summary {
    font-size: 0.85rem; font-weight: 600;
    padding: 6px 0 !important;
}

/* ── Main canvas: zero padding so viewer fills the space ── */
.main .block-container {
    padding-top: 0.4rem !important;
    padding-left: 1.0rem !important;
    padding-right: 0.8rem !important;
    max-width: 100% !important;
}

/* ── Hide st.metric — we use our own status strip ── */
[data-testid="metric-container"] { display: none !important; }

/* ── Status strip (results header bar) ── */
.morph-status {
    display: flex; align-items: center; gap: 0; flex-wrap: nowrap;
    background: #141820;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 7px; padding: 8px 16px;
    margin-bottom: 8px; overflow: hidden;
}
.morph-status .ms-item {
    display: flex; flex-direction: column; flex: 1;
    border-right: 1px solid rgba(255,255,255,0.07);
    padding: 0 14px; min-width: 0;
}
.morph-status .ms-item:first-child { padding-left: 0; }
.morph-status .ms-item:last-child  { border-right: none; padding-right: 0; }
.morph-status .ms-label {
    font-size: 0.65rem; font-weight: 600;
    color: rgba(255,255,255,0.50);
    letter-spacing: 0.06em; text-transform: uppercase;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.morph-status .ms-value {
    font-size: 1.05rem; font-weight: 700;
    color: #e8eaf6; line-height: 1.3; white-space: nowrap;
}
.morph-status .ms-value.good  { color: #4CAF50; }
.morph-status .ms-value.warn  { color: #FFC107; }
.morph-status .ms-value.info  { color: #42a5f5; }

/* ── Viewer controls row (above 3D canvas) ── */
.viewer-ctrl {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 4px;
}
.viewer-ctrl-seg {
    display: flex; gap: 2px;
    background: #141820;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 6px; padding: 3px;
}
.viewer-ctrl-seg button {
    background: transparent; border: none; cursor: pointer;
    color: rgba(255,255,255,0.5); font-size: 0.76rem;
    padding: 3px 9px; border-radius: 4px;
    transition: all 0.15s;
}
.viewer-ctrl-seg button.active,
.viewer-ctrl-seg button:hover {
    background: rgba(21,101,192,0.35); color: #90caf9;
}

/* ── Radio buttons: no label, tight pills ── */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    flex-wrap: nowrap !important; gap: 3px !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label {
    font-size: 0.76rem !important; padding: 3px 8px !important;
    white-space: nowrap !important;
}
/* Hide the radio label text when label_visibility=collapsed */
div[data-testid="stRadio"] > label { display: none !important; }

/* ── Download buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a6b3c, #0f4a2a) !important;
    color: white !important; border: none !important; font-weight: 600 !important;
}

/* ── Chat bubbles ── */
.chat-bubble {
    background: rgba(255,255,255,0.035);
    border-radius: 7px; padding: 7px 11px; margin: 3px 0;
    border-left: 3px solid #4CAF50; font-size: 0.83rem; line-height: 1.4;
}

/* ── Primary run button ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important; font-size: 0.93rem !important;
    font-weight: 700 !important; border: none !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.4);
}

/* ── Stiffness progress bar ── */
.stiff-track { background: #1e2d45; border-radius: 3px; height: 4px; margin-top: 4px; }
.stiff-fill  { height: 4px; border-radius: 3px; }

/* ── Expander: no top-border chrome ── */
[data-testid="stExpander"] > div:first-child { border-top: none !important; }

/* ── Section captions: smaller, dimmer ── */
[data-testid="stSidebar"] .stCaption p { font-size: 0.73rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────────────────────
_defaults = {
    "design_history":  [],
    "expert_mode":     False,
    "chat_state":      "idle",       # idle | extracting | confirming | running
    "chat_messages":   [],
    "pending_params":  None,
    "llm_provider":    "Anthropic Claude",
    "llm_model":       "claude-sonnet-4-6",
    "llm_api_key":     "",
    "design_mode":     "box",        # box | mesh
    "uploaded_mesh":   None,
    "design_mask":     None,         # np.ndarray (nely, nelx) bool | None
    "fem_mesh":        None,         # dict | None — generated FEM mesh
    # Structured load/support specs (expert mode)
    "load_specs": [
        {"id": 0, "type": "surface", "face": "Right (X=W)",
         "fx": 0.0, "fy": -500.0,
         "_dir_label": "↓ Downward", "_magnitude": 500.0, "_angle_deg": -90.0,
         "u": 0.5, "u_start": 0.25, "u_end": 0.75,
         "axis": "Z", "magnitude": 5000.0},
    ],
    "support_specs": [
        {"id": 0, "type": "fixed", "face": "Left (X=0)"},
    ],
    "_load_id_ctr":    1,
    "_support_id_ctr": 1,
    "_safety_factor":  2.0,
    # Multi-load scenarios (expert mode). Empty = use single load_specs above.
    # Each entry: {id, name, face, direction, magnitude, weight}
    "load_scenarios":  [],
    "_scenario_id_ctr": 0,
    "_sel_mode":       None,   # None | "support" | "load"
    "_pending_face":   None,
    "_pending_mode":   None,
    # Optimizer preset knobs (keyed sliders read from here)
    "_preset_vf":      0.40,
    "_preset_penal":   3.0,
    "_preset_rmin":    1.5,
    "_preset_maxiter": 60,
    "_preset_nelx":    20,
    "_preset_nely":    12,
    # 3-D SIMP
    "use_3d_simp":     False,
    "_preset_nelz":    8,
    # Pareto sweep results
    "pareto_results":  None,
    # LLM narration cache
    "topology_narration": None,
    # Active page (replaces st.tabs)
    "_page": "⚙️ Setup",
    # Right-panel view selector
    "_right_view": "🔷 Topology",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Restore shared result from URL query param ─────────────
_qstate = st.query_params.get("state", None)
if _qstate and "xPhys" not in st.session_state:
    try:
        _raw = gzip.decompress(base64.urlsafe_b64decode(_qstate.encode() + b"=="))
        _sd  = json.loads(_raw)
        _xp_bytes = gzip.decompress(base64.b64decode(_sd["xPhys_b64"].encode()))
        _xp = np.frombuffer(_xp_bytes, dtype=np.float32).reshape(_sd["xPhys_shape"])
        _smeta = _sd["meta"]
        _s_mat  = _smeta.get("export_mat", "PLA (Bioplastic)")
        _s_mat  = _s_mat if _s_mat in MATERIALS else "PLA (Bioplastic)"
        _stl_b, _nf = to_stl_bytes(_xp, *_smeta["box"], _smeta.get("iso", 0.45))
        _sscores = compute_mat_scores(
            [_s_mat], *_smeta["box"],
            _smeta.get("vf", 0.4), 500.0, 2.0, _smeta.get("compliance", 1.0)
        )
        st.session_state.update({
            "xPhys": _xp, "history": _smeta.get("history", [_smeta.get("compliance", 1.0)]),
            "history_lc2": [], "stress_field": None,
            "stl_bytes": _stl_b, "n_faces": _nf, "mode": "Single Material",
            "mat_scores": _sscores, "winners": {}, "export_mat": _s_mat,
            "compare_mats": [_s_mat], "opt_meta": _smeta,
        })
        st.toast("Restored shared design from URL", icon="🔗")
    except Exception:
        pass  # silently ignore malformed URLs

# ── Seed API key from environment / secrets (so it's never blank on reload) ──
if not st.session_state["llm_api_key"]:
    import os as _os
    for _p_name, _p_data in chat_module.PROVIDER_PRESETS.items():
        _ev = _p_data.get("key_env", "")
        if not _ev:
            continue
        _found = _os.environ.get(_ev, "")
        if not _found:
            try:
                _found = st.secrets.get(_ev, "")
            except Exception:
                pass
        if _found:
            st.session_state["llm_api_key"]  = _found
            st.session_state["llm_provider"] = _p_name
            _model_list = _p_data.get("models", [])
            if _model_list and st.session_state["llm_model"] not in _model_list:
                st.session_state["llm_model"] = _model_list[0]
            break

# ─────────────────────────────────────────────────────────────
#  Shared UI constants (used in sidebar AND Setup tab)
# ─────────────────────────────────────────────────────────────
_DIR_PRESETS = {
    "↓  Downward":       ( 0.0,    -1.0,  -90.0),
    "↑  Upward":         ( 0.0,     1.0,   90.0),
    "→  Rightward":      ( 1.0,     0.0,    0.0),
    "←  Leftward":       (-1.0,     0.0,  180.0),
    "↘  Down-Right":     ( 0.707,  -0.707, -45.0),
    "↙  Down-Left":      (-0.707,  -0.707,-135.0),
    "↗  Up-Right":       ( 0.707,   0.707,  45.0),
    "↖  Up-Left":        (-0.707,   0.707, 135.0),
    "✏  Custom angle":   None,
}
_DIR_PRESET_KEYS = list(_DIR_PRESETS.keys())

_SUP_TYPES = {
    "fixed":    "🔒 Fixed (all DOF)",
    "roller_x": "↔ Roller (slides in Y)",
    "roller_y": "↕ Roller (slides in X)",
}
_LD_TYPES = {
    "surface": "⬛ Surface — full face",
    "point":   "📍 Point load",
    "line":    "📏 Line load",
    "moment":  "🔄 Moment / Torque",
}

# ─────────────────────────────────────────────────────────────
#  Helper: build boundary-condition params from session or sidebar
# ─────────────────────────────────────────────────────────────
def _get_bc_params():
    """Return boundary condition dict from session state or live sidebar values."""
    if st.session_state.get("pending_params") and not st.session_state["expert_mode"]:
        p = st.session_state["pending_params"]
        return (
            p.get("fixed_face",     "Left (X=0)"),
            p.get("load_face",      "Right (X=W)"),
            p.get("force_direction","-Y"),
            float(p.get("applied_force_n", 500)),
            float(p.get("safety_factor",   2.0)),
        )
    return (
        st.session_state.get("_fixed_face",      "Left (X=0)"),
        st.session_state.get("_load_face",       "Right (X=W)"),
        st.session_state.get("_force_direction", "-Y"),
        float(st.session_state.get("_applied_force_n", 500)),
        float(st.session_state.get("_safety_factor",   2.0)),
    )

# ─────────────────────────────────────────────────────────────
#  SIDEBAR (Expert Mode)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Compact header row: logo + mode toggle ────────────────
    _hcol1, _hcol2 = st.columns([3, 4])
    with _hcol1:
        st.markdown(
            "<div style='padding:4px 0;font-size:1.1rem;font-weight:800;"
            "color:#90caf9;letter-spacing:-0.02em;'>🧬 MorphAI</div>",
            unsafe_allow_html=True,
        )
    with _hcol2:
        expert_mode_current = st.session_state["expert_mode"]
        toggle_label = "💬 Chat" if expert_mode_current else "⚙ Expert"
        if st.button(toggle_label, use_container_width=True,
                     help="Switch between Chat Mode (AI-guided) and Expert Mode (direct controls)"):
            st.session_state["expert_mode"] = not expert_mode_current
            st.rerun()

    st.divider()

    if not expert_mode_current:
        st.markdown(
            "<div style='font-size:0.76rem;color:rgba(144,202,249,0.7);"
            "margin-bottom:8px;'>Settings below are optional — the chat fills them in.</div>",
            unsafe_allow_html=True,
        )

    part_name = st.text_input(
        "Part name:", value="My Part", max_chars=40,
        placeholder="e.g. Mounting Bracket",
        label_visibility="collapsed",
    )

    # ── 1. Material ───────────────────────────────────────────
    _sidebar_expanded = expert_mode_current  # collapse in Chat Mode — user picks via chat
    with st.expander("1. Material", expanded=_sidebar_expanded):
        mode = st.radio("Mode:", ["Single Material", "Material Comparison"], horizontal=True)
        if mode == "Single Material":
            sel_mat = st.selectbox("Material:", options=list(MATERIALS.keys()), index=0)
            mat = MATERIALS[sel_mat]
            if mat["printable"]:
                st.success(f"FDM · {mat['print_temp']} · Bed {mat['bed_temp']}")
            else:
                st.warning("Reference only — not FDM printable")
            st.caption(f"*{mat['notes']}*")
            compare_mats = [sel_mat]
        else:
            compare_mats = [
                name for name in MATERIALS
                if st.checkbox(name, value=MATERIALS[name]["printable"], key=f"chk_{name}")
            ]
            if not compare_mats:
                compare_mats = ["PLA (Bioplastic)"]
            sel_mat = compare_mats[0]
            mat = MATERIALS[sel_mat]

    # ── 2. Design Space ───────────────────────────────────────
    with st.expander("2. Design Space", expanded=_sidebar_expanded):
        design_mode = st.radio(
            "Input type:", ["Rectangular box", "Upload mesh (STL)"],
            key="design_mode_radio", horizontal=True,
        )
        st.session_state["design_mode"] = "mesh" if design_mode == "Upload mesh (STL)" else "box"

        if st.session_state["design_mode"] == "mesh":
            uploaded_file = st.file_uploader(
                "Upload STL file", type=["stl", "obj"],
                help="Your mesh will be voxelized and used as the design space.",
            )
            if uploaded_file is not None:
                st.session_state["uploaded_mesh"] = uploaded_file.read()
                st.success(f"Loaded: {uploaded_file.name} ({len(st.session_state['uploaded_mesh'])//1024} KB)")
            elif st.session_state["uploaded_mesh"] is None:
                st.info("No mesh uploaded — will use rectangular box.")
        else:
            st.session_state["uploaded_mesh"] = None

        box_w = st.slider("Width  (X, mm)", 10, 300, 100, 5)
        box_h = st.slider("Height (Y, mm)", 10, 300,  60, 5)
        box_d = st.slider("Depth  (Z, mm)", 10, 300,  40, 5)

    # ── 3. Advanced Settings (collapsed) ─────────────────────
    with st.expander("3. Advanced Settings", expanded=False):
        # Presets
        st.caption("Presets")
        _pa, _pb, _pc = st.columns(3)
        with _pa:
            if st.button("⚡ Quick", use_container_width=True,
                         help="15×8 mesh · 25 iterations — fast preview"):
                st.session_state.update({
                    "_preset_nelx": 15, "_preset_nely": 8,
                    "_preset_maxiter": 25, "_preset_penal": 3.0,
                    "_preset_vf": 0.4, "_min_feat_mm": 5.0,
                })
                st.rerun()
        with _pb:
            if st.button("⚙ Standard", use_container_width=True,
                         help="20×12 mesh · 60 iterations — balanced"):
                st.session_state.update({
                    "_preset_nelx": 20, "_preset_nely": 12,
                    "_preset_maxiter": 60, "_preset_penal": 3.0,
                    "_preset_vf": 0.4, "_min_feat_mm": 5.0,
                })
                st.rerun()
        with _pc:
            if st.button("🎯 Hi-Res", use_container_width=True,
                         help="40×24 mesh · 100 iterations — detailed, slower"):
                st.session_state.update({
                    "_preset_nelx": 40, "_preset_nely": 24,
                    "_preset_maxiter": 100, "_preset_penal": 3.5,
                    "_preset_vf": 0.35, "_min_feat_mm": 4.0,
                })
                st.rerun()

        st.caption("Optimizer")
        volume_fraction = st.slider(
            "Volume Fraction", 0.1, 0.9,
            float(st.session_state["_preset_vf"]), 0.05,
            key="_preset_vf",
            help="How much material to keep. 0.4 = 40%. Lower = lighter but weaker.",
        )
        penal = st.slider(
            "SIMP Penalty (p)", 1.0, 5.0,
            float(st.session_state["_preset_penal"]), 0.5,
            key="_preset_penal",
            help="Higher = sharper solid/void boundary. 3.0 is standard.",
        )
        _nozzle_d = st.slider(
            "Nozzle diameter (mm)", 0.2, 0.8,
            float(st.session_state.get("_nozzle_d", 0.4)), 0.1,
            help="FDM nozzle diameter. Sets the minimum printable feature threshold.",
        )
        st.session_state["_nozzle_d"] = _nozzle_d
        _min_feat_default = max(3.0, 3.0 * _nozzle_d)
        _min_feat_mm = st.slider(
            "Min feature size (mm)", 1.0, 20.0,
            float(st.session_state.get("_min_feat_mm", _min_feat_default)), 0.5,
            help="Smallest structural member. Back-calculates the filter radius for the optimizer.",
        )
        st.session_state["_min_feat_mm"] = _min_feat_mm
        _printable_thresh = 3.0 * _nozzle_d
        if _min_feat_mm < _printable_thresh:
            st.warning(
                f"Min feature {_min_feat_mm:.1f} mm < 3× nozzle ({_printable_thresh:.1f} mm) "
                f"— features may not print reliably."
            )
        # Derive rmin (element units) for build_filter — updates live as sliders change
        _rmin_nelx = int(st.session_state.get("_preset_nelx", 20))
        rmin = max(1.0, _min_feat_mm / max(0.001, box_w / max(1, _rmin_nelx)))
        st.caption(f"Filter radius: {rmin:.2f} elements  ({_min_feat_mm:.1f} mm / {box_w/_rmin_nelx:.1f} mm·el⁻¹)")
        max_iter = st.slider("Max Iterations", 10, 150,
                             int(st.session_state["_preset_maxiter"]), 10,
                             key="_preset_maxiter")
        nelx = st.select_slider("Elements X", [10, 15, 20, 25, 30, 40, 50],
                                value=int(st.session_state["_preset_nelx"]),
                                key="_preset_nelx")
        nely = st.select_slider("Elements Y", [6, 8, 10, 12, 15, 20, 25],
                                value=int(st.session_state["_preset_nely"]),
                                key="_preset_nely")

        st.caption("3-D Topology")
        use_3d = st.checkbox(
            "True 3-D SIMP optimisation",
            value=st.session_state["use_3d_simp"],
            key="use_3d_simp",
            help=(
                "Uses full 3-D hex FEA — the Z-direction is structurally active. "
                "~4× slower than 2-D mode. Recommended: nelx ≤ 20, nelz ≤ 8."
            ),
        )
        if use_3d:
            nelz = st.select_slider(
                "Elements Z", [4, 6, 8, 10, 12, 15, 20],
                value=int(st.session_state["_preset_nelz"]),
                key="_preset_nelz",
            )
            st.caption("⚠️ 3-D solve: ~4-10× slower. Keep nelz ≤ 10 for interactive use.")
        else:
            nelz = int(st.session_state["_preset_nelz"])  # not used in 2-D mode

        st.caption("Infill Pattern")
        infill_pattern = st.selectbox("Pattern:", list(INFILL_PATTERNS.keys()), index=0)
        st.caption(INFILL_PATTERNS[infill_pattern]["desc"])
        period_mm    = st.slider("Cell period (mm)", 4.0, 30.0, 12.0, 1.0)
        void_thresh  = st.slider("Void threshold",   0.05, 0.40, 0.15, 0.05)
        solid_thresh = st.slider("Shell threshold",  0.50, 0.95, 0.75, 0.05)
        infill_res   = st.select_slider(
            "Infill resolution", options=["Low (fast)", "Medium", "High (slow)"],
            value="Medium",
        )
        res_map = {"Low (fast)": (40,24,16), "Medium": (60,36,20), "High (slow)": (80,48,28)}
        fine_nx, fine_ny, fine_nz = res_map[infill_res]

        st.caption("3D Print Estimator")
        layer_h  = st.slider("Layer height (mm)",  0.10, 0.40, 0.20, 0.05)
        speed_ms = st.slider("Print speed (mm/s)", 20, 150, 50, 10)

        st.caption("Export")
        iso_threshold = st.slider(
            "Topology iso-level", 0.10, 0.90, 0.45, 0.05,
            help="Surface threshold: lower shows more material, higher shows less. 0.45 is standard.",
        )

    st.divider()

    # ── Action buttons ────────────────────────────────────────
    run_btn    = st.button("▶ Run Optimizer", type="primary", use_container_width=True)
    infill_btn = st.button(
        "Generate Infill", use_container_width=True,
        disabled=("xPhys" not in st.session_state),
        help="Run the optimizer first, then generate an infill pattern.",
    )
    if "xPhys" in st.session_state:
        if st.button("Clear Results", use_container_width=True):
            for k in ["xPhys", "history", "history_lc2", "stress_field",
                      "stl_bytes", "n_faces", "opt_meta",
                      "mat_scores", "winners", "export_mat", "compare_mats", "mode",
                      "infill_stl", "infill_faces", "infill_vf", "infill_sf",
                      "infill_pattern", "infill_period", "fem_mesh"]:
                st.session_state.pop(k, None)
            st.rerun()

# ─────────────────────────────────────────────────────────────
#  Derive BC scalars from session state (set by Setup tab)
# ─────────────────────────────────────────────────────────────
_sup0 = st.session_state["support_specs"][0] if st.session_state["support_specs"] else {}
_ld0  = st.session_state["load_specs"][0]    if st.session_state["load_specs"]    else {}
fixed_face = _sup0.get("face", "Left (X=0)")
load_face  = _ld0.get("face",  "Right (X=W)")
_total_fx  = sum(s.get("fx", 0.0) for s in st.session_state["load_specs"]
                 if s.get("type") != "moment")
_total_fy  = sum(s.get("fy", 0.0) for s in st.session_state["load_specs"]
                 if s.get("type") != "moment")
applied_force_n = math.sqrt(_total_fx**2 + _total_fy**2) or 500.0
force_direction = (_total_fx, _total_fy, 0.0)
safety_factor   = float(st.session_state.get("_safety_factor", 2.0))

# ─────────────────────────────────────────────────────────────
#  pending_params ↔ load/support specs conversion helpers
# ─────────────────────────────────────────────────────────────
_CHAT_FACE_MAP = {
    "left":   "Left (X=0)",  "right": "Right (X=W)",
    "bottom": "Bottom (Y=0)","top":   "Top (Y=H)",
    "front":  "Front (Z=0)", "back":  "Back (Z=D)",
}
_CHAT_DIR_FXY = {
    "-X": (-1.0,  0.0), "+X": (1.0,  0.0),
    "-Y": ( 0.0, -1.0), "+Y": (0.0,  1.0),
    "-Z": ( 0.0,  0.0), "+Z": (0.0,  0.0),
}

def _pending_to_specs(p):
    """Convert pending_params dict → (load_specs, support_specs) for the optimizer and viewer."""
    fixed_face = _CHAT_FACE_MAP.get(str(p.get("fixed_face","left")).lower().split()[0], "Left (X=0)")
    load_face  = _CHAT_FACE_MAP.get(str(p.get("load_face","right")).lower().split()[0], "Right (X=W)")
    ufx, ufy   = _CHAT_DIR_FXY.get(p.get("force_direction", "-Y"), (0.0, -1.0))
    force_n    = float(p.get("applied_force_n", 500.0))
    load_specs = [{
        "id": 0, "type": "surface", "face": load_face,
        "fx": ufx * force_n, "fy": ufy * force_n,
        "_dir_label": p.get("force_direction", "-Y"),
        "_magnitude": force_n, "_angle_deg": -90.0,
        "u": 0.5, "u_start": 0.25, "u_end": 0.75, "axis": "Z", "magnitude": 5000.0,
    }]
    support_specs = [{"id": 0, "type": "fixed", "face": fixed_face}]
    return load_specs, support_specs

# ─────────────────────────────────────────────────────────────
#  Page header
# ─────────────────────────────────────────────────────────────
def _load_summary(specs):
    parts = []
    for s in specs:
        t = s.get("type", "surface")
        if t == "moment":
            parts.append(f"Moment {s.get('magnitude', 0):.0f} N·mm")
        else:
            _m = math.sqrt(s.get("fx", 0)**2 + s.get("fy", 0)**2)
            parts.append(f"{t.title()} {_m:.0f} N on {s.get('face','?')}")
    return " + ".join(parts) if parts else "No loads"

_load_sum_str = _load_summary(st.session_state['load_specs'])
st.markdown(
    f"<div style='display:flex;align-items:center;gap:10px;padding:2px 0 6px 0;"
    f"border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:8px;'>"
    f"<span style='font-size:0.78rem;font-weight:700;color:#546e7a;"
    f"letter-spacing:0.05em;text-transform:uppercase;'>{part_name}</span>"
    f"<span style='color:rgba(255,255,255,0.12);font-size:0.75rem;'>·</span>"
    f"<span style='font-size:0.78rem;color:#455a64;'>{sel_mat.split('(')[0].strip()}</span>"
    f"<span style='color:rgba(255,255,255,0.12);font-size:0.75rem;'>·</span>"
    f"<span style='font-size:0.78rem;color:#37474f;'>{box_w}×{box_h}×{box_d} mm</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
#  2-D face diagram helpers  (Phase 1.1 — replaces 3D canvas)
# ─────────────────────────────────────────────────────────────
def _face_status(face, sup_specs, load_specs, pend_face):
    """Return 'pending' | 'support' | 'load' | 'free'."""
    if face == pend_face:
        return "pending"
    if any(s["face"] == face for s in sup_specs):
        return "support"
    if any(l["face"] == face for l in load_specs):
        return "load"
    return "free"

_STATUS_FILL = {
    "support": "#1565c0", "load": "#b71c1c",
    "pending": "#e65100", "free": "#1e2d45",
}
_STATUS_STROKE = {
    "support": "#42a5f5", "load": "#ef5350",
    "pending": "#ffb74d", "free": "#4a6080",
}

def _face_iso_svg(sup_specs, load_specs, pend_face=None):
    """Return an HTML string with a labelled isometric box SVG."""
    def _fill(f):   return _STATUS_FILL[_face_status(f, sup_specs, load_specs, pend_face)]
    def _stroke(f): return _STATUS_STROKE[_face_status(f, sup_specs, load_specs, pend_face)]

    # Fixed isometric geometry for a generic box (W=120, H=75, D=45 in SVG units)
    # Visible faces: Front (Z=0), Top (Y=H), Right (X=W)
    front  = "80,140 200,140 200,65 80,65"
    top    = "80,65  200,65  228,38 108,38"
    right  = "200,140 228,113 228,38 200,65"
    # Hidden faces shown as flat rects
    lft_x, lft_y, lft_w, lft_h  = 6, 65, 70, 75
    bot_x, bot_y, bot_w, bot_h  = 80, 144, 120, 22
    bk_x,  bk_y,  bk_w,  bk_h  = 232, 38, 80, 75

    ff, tf, rf = _fill("Front (Z=0)"), _fill("Top (Y=H)"), _fill("Right (X=W)")
    fs, ts, rs = _stroke("Front (Z=0)"), _stroke("Top (Y=H)"), _stroke("Right (X=W)")
    lf, ls = _fill("Left (X=0)"),   _stroke("Left (X=0)")
    bf, bs = _fill("Bottom (Y=0)"), _stroke("Bottom (Y=0)")
    kf, ks = _fill("Back (Z=D)"),   _stroke("Back (Z=D)")

    return f"""<div style="background:#0e1117;border:1px solid #1e2d45;border-radius:10px;
padding:10px 12px 6px;margin-bottom:10px;">
<svg viewBox="0 0 320 172" width="100%" style="max-height:172px;display:block;">
  <!-- Visible faces -->
  <polygon points="{front}" fill="{ff}" stroke="{fs}" stroke-width="1.5" opacity="0.93"/>
  <polygon points="{top}"   fill="{tf}" stroke="{ts}" stroke-width="1.5" opacity="0.93"/>
  <polygon points="{right}" fill="{rf}" stroke="{rs}" stroke-width="1.5" opacity="0.93"/>
  <!-- Labels on visible faces -->
  <text x="140" y="107" text-anchor="middle" font-size="11" fill="white"
        font-family="system-ui,sans-serif" font-weight="700">FRONT</text>
  <text x="157" y="56"  text-anchor="middle" font-size="10" fill="white"
        font-family="system-ui,sans-serif" font-weight="700">TOP</text>
  <text x="218" y="85"  text-anchor="middle" font-size="9"  fill="white"
        font-family="system-ui,sans-serif" font-weight="700">RIGHT</text>
  <!-- Hidden face rectangles -->
  <rect x="{lft_x}" y="{lft_y}" width="{lft_w}" height="{lft_h}"
        rx="3" fill="{lf}" stroke="{ls}" stroke-width="1" opacity="0.8"/>
  <text x="{lft_x+lft_w//2}" y="{lft_y+lft_h//2+4}" text-anchor="middle"
        font-size="10" fill="white" font-family="system-ui,sans-serif">LEFT</text>
  <rect x="{bot_x}" y="{bot_y}" width="{bot_w}" height="{bot_h}"
        rx="3" fill="{bf}" stroke="{bs}" stroke-width="1" opacity="0.8"/>
  <text x="{bot_x+bot_w//2}" y="{bot_y+bot_h//2+4}" text-anchor="middle"
        font-size="10" fill="white" font-family="system-ui,sans-serif">BOTTOM</text>
  <rect x="{bk_x}" y="{bk_y}" width="{bk_w}" height="{bk_h}"
        rx="3" fill="{kf}" stroke="{ks}" stroke-width="1" opacity="0.8"/>
  <text x="{bk_x+bk_w//2}" y="{bk_y+bk_h//2+4}" text-anchor="middle"
        font-size="10" fill="white" font-family="system-ui,sans-serif">BACK</text>
  <!-- Legend -->
  <circle cx="10" cy="12" r="5" fill="#1565c0"/>
  <text x="18" y="16" font-size="9" fill="#90a4ae" font-family="system-ui,sans-serif">Support</text>
  <circle cx="74" cy="12" r="5" fill="#b71c1c"/>
  <text x="82" y="16" font-size="9" fill="#90a4ae" font-family="system-ui,sans-serif">Load</text>
  <circle cx="130" cy="12" r="5" fill="#e65100"/>
  <text x="138" y="16" font-size="9" fill="#90a4ae" font-family="system-ui,sans-serif">Selected</text>
  <circle cx="196" cy="12" r="5" fill="#1e2d45" stroke="#4a6080" stroke-width="1"/>
  <text x="204" y="16" font-size="9" fill="#90a4ae" font-family="system-ui,sans-serif">Unassigned</text>
</svg></div>"""

# ─────────────────────────────────────────────────────────────
#  Pre-compute derived scalars (used throughout the page)
# ─────────────────────────────────────────────────────────────
allowable_stress = round(MATERIALS[sel_mat]["yield_strength_mpa"] / safety_factor, 1)
load_area_mm2 = {
    "Left (X=0)": box_h*box_d, "Right (X=W)": box_h*box_d,
    "Bottom (Y=0)": box_w*box_d, "Top (Y=H)": box_w*box_d,
    "Front (Z=0)": box_w*box_h, "Back (Z=D)": box_w*box_h,
}.get(load_face, box_w*box_h)
est_mass_g = round(
    (box_w/10) * (box_h/10) * (box_d/10) * volume_fraction * MATERIALS[sel_mat]["rho_gcc"], 1
)
_has_results = "xPhys" in st.session_state
result_mode  = st.session_state.get("mode", "Single Material") if _has_results else "Single Material"

expert_mode = st.session_state["expert_mode"]

# ─────────────────────────────────────────────────────────────
#  LAYOUT — chat-first landing vs 2-panel results view
# ─────────────────────────────────────────────────────────────
_landing = (not expert_mode and not _has_results
            and st.session_state["chat_state"] not in ("running",))
if _landing:
    _, _left_col, _ = st.columns([1, 4, 1])
    _right_col = st.container()
else:
    _left_col, _right_col = st.columns([2, 8] if expert_mode else [3, 7])

# ─────────────────────────────────────────────────────────────
#  LEFT COLUMN — user interaction panel
# ─────────────────────────────────────────────────────────────
with _left_col:
    if not expert_mode:

        if _landing:
            st.markdown(
                "<div style='text-align:center;padding:1.5rem 0 1.2rem 0;'>"
                "<div style='font-size:1.75rem;font-weight:800;color:#e8eaf6;"
                "letter-spacing:-0.03em;line-height:1.25;'>"
                "Describe a part.<br>Get a print-ready structure."
                "</div>"
                "<div style='font-size:0.88rem;color:rgba(255,255,255,0.45);"
                "margin-top:0.6rem;line-height:1.5;'>"
                "MorphAI reads your plain-English description, extracts engineering "
                "constraints, runs a real SIMP topology optimizer, and outputs a "
                "print-ready STL — no form-filling required."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Provider/model/key config ─────────────────────────
        _has_key = bool(st.session_state["llm_api_key"])
        _ai_status = (
            f"● {st.session_state['llm_model']}" if _has_key
            else "○ No AI key — click to set up"
        )
        _ai_color = "#4CAF50" if _has_key else "#FFC107"
        st.markdown(
            f"<div style='font-size:0.76rem;color:{_ai_color};"
            f"margin-bottom:4px;'>{_ai_status}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("AI settings", expanded=False):
            provider_names = list(chat_module.PROVIDER_PRESETS.keys())
            prov_idx = (provider_names.index(st.session_state["llm_provider"])
                        if st.session_state["llm_provider"] in provider_names else 0)
            provider = st.selectbox("Provider:", provider_names, index=prov_idx)
            st.session_state["llm_provider"] = provider

            preset = chat_module.PROVIDER_PRESETS[provider]
            model_opts = preset["models"]
            cur_model = st.session_state.get("llm_model", model_opts[0])
            model_idx = model_opts.index(cur_model) if cur_model in model_opts else 0
            model_custom = st.text_input(
                "Model (or type custom):", value=model_opts[model_idx],
                help="Exact model string accepted by litellm, e.g. 'gpt-4o' or 'ollama/llama3.2'"
            )
            st.session_state["llm_model"] = model_custom

            env_key = chat_module.get_api_key_from_env(provider)
            if env_key:
                st.success(f"API key found in environment ({preset.get('key_env','')}).")
                st.session_state["llm_api_key"] = env_key
            else:
                api_key_input = st.text_input(
                    "API Key:", value=st.session_state["llm_api_key"],
                    type="password",
                    help="Never stored to disk. Cleared when you close the browser tab."
                )
                st.session_state["llm_api_key"] = api_key_input

            test_col, _ = st.columns([1, 2])
            with test_col:
                if st.button("Test connection", use_container_width=True):
                    with st.spinner("Testing…"):
                        ok, msg = chat_module.test_connection(
                            st.session_state["llm_model"],
                            st.session_state["llm_api_key"],
                            st.session_state["llm_provider"],
                        )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        # ── Quick-Start presets (no API key needed) ──────────────
        # Quick-start presets — label, icon, short description, params
        _QUICK_PRESETS = [
            ("Cantilever", "↔", "Fixed left · load right · 500 N · PLA", {
                "fixed_face": "left", "load_face": "right",
                "force_direction": "-Y", "applied_force_n": 500.0,
                "material": "PLA", "safety_factor": 2.0, "volume_fraction": 0.40,
                "confidence_notes": "Classic cantilever: fixed on left, downward load on right.",
            }),
            ("Bracket", "⌐", "Fixed left · load bottom · 20 kg · PLA", {
                "fixed_face": "left", "load_face": "bottom",
                "force_direction": "-Y", "applied_force_n": 196.0,
                "material": "PLA", "safety_factor": 2.0, "volume_fraction": 0.40,
                "confidence_notes": "Wall bracket: wall-mounted on left, 20 kg shelf load downward.",
            }),
            ("Shelf", "⬛", "Fixed bottom · load top · 100 kg · PETG", {
                "fixed_face": "bottom", "load_face": "top",
                "force_direction": "-Y", "applied_force_n": 981.0,
                "material": "PETG", "safety_factor": 3.0, "volume_fraction": 0.50,
                "confidence_notes": "Shelf support: base fixed, 100 kg load from above.",
            }),
        ]

        if st.session_state["chat_state"] == "idle":
            st.markdown(
                "<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.05em;"
                "color:rgba(255,255,255,0.35);text-transform:uppercase;"
                "margin:6px 0 5px 0;'>Try a preset</div>",
                unsafe_allow_html=True,
            )
            for _qi, (_qlabel, _qicon, _qdesc, _qparams) in enumerate(_QUICK_PRESETS):
                _btn_html = (
                    f"<div style='display:flex;align-items:center;gap:8px;"
                    f"background:#141820;border:1px solid rgba(255,255,255,0.08);"
                    f"border-radius:6px;padding:7px 10px;margin-bottom:5px;"
                    f"cursor:pointer;'>"
                    f"<span style='font-size:1.1rem;'>{_qicon}</span>"
                    f"<div><div style='font-size:0.83rem;font-weight:600;color:#e8eaf6;'>{_qlabel}</div>"
                    f"<div style='font-size:0.71rem;color:rgba(255,255,255,0.38);'>{_qdesc}</div></div>"
                    f"</div>"
                )
                # Render the styled card, then an invisible button beneath it
                _bcol1, _bcol2 = st.columns([5, 1])
                with _bcol1:
                    st.markdown(_btn_html, unsafe_allow_html=True)
                with _bcol2:
                    if st.button("▶", key=f"qpreset_{_qi}",
                                 help=_qparams["confidence_notes"],
                                 use_container_width=True):
                        st.session_state["pending_params"] = dict(_qparams)
                        st.session_state["chat_messages"].append({
                            "role": "user", "content": f"[Preset: {_qlabel}]"
                        })
                        st.session_state["chat_messages"].append({
                            "role": "assistant",
                            "content": f"Loaded **{_qlabel}** preset. Review below — edit anything, then run."
                        })
                        st.session_state["chat_state"] = "confirming"
                        st.rerun()

        st.markdown(
            "<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.05em;"
            "color:rgba(255,255,255,0.35);text-transform:uppercase;"
            "margin:10px 0 5px 0;'>Or describe in plain English</div>",
            unsafe_allow_html=True,
        )

        # ── Conversation history display ──────────────────────
        for msg in st.session_state["chat_messages"]:
            role_label = "You" if msg["role"] == "user" else "AI"
            bubble_color = "royalblue" if msg["role"] == "user" else "#4CAF50"
            st.markdown(
                f"<div class='chat-bubble' style='border-left-color:{bubble_color}'>"
                f"<b>{role_label}:</b> {msg['content']}</div>",
                unsafe_allow_html=True,
            )

        # ── Chat state machine (input states) ─────────────────
        chat_state = st.session_state["chat_state"]

        if chat_state == "idle":
            with st.expander(
                "📎 Upload CAD / STL file  (optional — or just describe your shape below)",
                expanded=(st.session_state["design_mode"] == "mesh"),
            ):
                _chat_upfile = st.file_uploader(
                    "Upload STL or OBJ", type=["stl", "obj"],
                    key="chat_file_uploader",
                    label_visibility="collapsed",
                    help="Your geometry will be voxelized and used as the design space.",
                )
                if _chat_upfile is not None:
                    _raw_bytes = _chat_upfile.read()
                    if _raw_bytes != st.session_state.get("uploaded_mesh"):
                        st.session_state["uploaded_mesh"] = _raw_bytes
                        st.session_state["design_mode"]   = "mesh"
                        with st.spinner("Reading mesh…"):
                            try:
                                _prev_mask = voxelize_mesh(_raw_bytes, nelx, nely, 8)
                                st.session_state["design_mask"] = (
                                    _prev_mask.any(axis=2) if _prev_mask.ndim == 3
                                    else _prev_mask
                                )
                                _in_pct = int(st.session_state["design_mask"].sum() /
                                              st.session_state["design_mask"].size * 100)
                                st.success(
                                    f"Loaded **{_chat_upfile.name}** — "
                                    f"{_in_pct}% of grid elements inside design domain."
                                )
                            except Exception as _ue:
                                st.error(f"Could not read mesh: {_ue}")
                elif st.session_state["design_mode"] == "mesh":
                    if st.button("✕ Remove uploaded mesh", key="chat_rm_mesh"):
                        st.session_state["uploaded_mesh"] = None
                        st.session_state["design_mode"]   = "box"
                        st.session_state["design_mask"]   = None
                        st.rerun()
                    else:
                        _nb = len(st.session_state.get("uploaded_mesh") or b"")
                        st.info(f"Mesh loaded ({_nb//1024} KB). Upload a new file to replace it.")

            with st.form("chat_form", clear_on_submit=True):
                user_msg = st.text_area(
                    "Your description:", height=120 if _landing else 80,
                    placeholder="e.g. I need a bracket fixed on the left that holds 50kg from the top, made of Nylon PA12.",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Send ▶", use_container_width=True)

            if submitted and user_msg.strip():
                st.session_state["chat_messages"].append(
                    {"role": "user", "content": user_msg.strip()}
                )
                st.session_state["chat_state"] = "extracting"
                st.rerun()

        elif chat_state == "extracting":
            with st.spinner("Thinking…"):
                try:
                    params = chat_module.extract_params(
                        st.session_state["chat_messages"][-1]["content"],
                        st.session_state["llm_model"],
                        st.session_state["llm_api_key"],
                        st.session_state["llm_provider"],
                    )
                    st.session_state["pending_params"] = params
                    notes = params.get("confidence_notes", "")
                    reply = (
                        f"I understood your request. Please review the parameters below"
                        + (f" — note: *{notes}*" if notes else "") + "."
                    )
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": reply}
                    )
                    st.session_state["chat_state"] = "confirming"
                except Exception as e:
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": f"Sorry, I couldn't extract parameters: {e}"}
                    )
                    st.session_state["chat_state"] = "idle"
            st.rerun()

        elif chat_state == "confirming":
            p = st.session_state["pending_params"]

            # Friendly face labels (no coordinate jargon for non-engineers)
            _CONF_FACES = ["left", "right", "bottom", "top", "front", "back"]
            _CONF_FACE_LABELS = {
                "left": "Left face", "right": "Right face",
                "bottom": "Bottom face", "top": "Top face",
                "front": "Front face", "back": "Back face",
            }
            _CONF_DIR_LABELS = {
                "-Y": "Downward (-Y)", "+Y": "Upward (+Y)",
                "-X": "Leftward (-X)", "+X": "Rightward (+X)",
                "-Z": "Into screen (-Z)", "+Z": "Out of screen (+Z)",
            }
            _conf_face_opts = list(_CONF_FACE_LABELS.keys())

            def _conf_face_idx(val):
                v = str(val).lower().split()[0]  # "Left (X=0)" → "left", "left" → "left"
                return _conf_face_opts.index(v) if v in _conf_face_opts else 0

            def _conf_dir_idx(val):
                dirs = list(_CONF_DIR_LABELS.keys())
                return dirs.index(val) if val in dirs else 2  # default -Y

            c1c, c2c = st.columns(2)
            with c1c:
                _ff_sel = st.selectbox(
                    "Fixed Face",
                    options=_conf_face_opts,
                    format_func=lambda v: _CONF_FACE_LABELS[v],
                    index=_conf_face_idx(p["fixed_face"]),
                    key="conf_fixed",
                )
                p["fixed_face"] = _ff_sel

                _lf_sel = st.selectbox(
                    "Load Face",
                    options=_conf_face_opts,
                    format_func=lambda v: _CONF_FACE_LABELS[v],
                    index=_conf_face_idx(p["load_face"]),
                    key="conf_load",
                )
                p["load_face"] = _lf_sel

                _dir_opts = list(_CONF_DIR_LABELS.keys())
                _dir_sel = st.selectbox(
                    "Force Direction",
                    options=_dir_opts,
                    format_func=lambda v: _CONF_DIR_LABELS[v],
                    index=_conf_dir_idx(p["force_direction"]),
                    key="conf_dir",
                )
                p["force_direction"] = _dir_sel
            with c2c:
                p["applied_force_n"] = st.number_input("Applied Force (N)",
                                                        value=float(p["applied_force_n"]),
                                                        min_value=1.0, step=10.0,
                                                        key="conf_force")
                mat_names = list(MATERIALS.keys())
                mat_idx = mat_names.index(p["material"]) if p["material"] in mat_names else 0
                p["material"] = st.selectbox("Material", mat_names, index=mat_idx, key="conf_mat")
                p["safety_factor"] = st.number_input("Safety Factor",
                                                      value=float(p["safety_factor"]),
                                                      min_value=1.0, max_value=5.0, step=0.5,
                                                      key="conf_sf")

            st.session_state["pending_params"] = p

            # Sync load/support specs → viewer updates live as user edits confirming fields
            _conf_ld_specs, _conf_sup_specs = _pending_to_specs(p)
            st.session_state["load_specs"]    = _conf_ld_specs
            st.session_state["support_specs"] = _conf_sup_specs

            # Live face diagram — shows fixed (blue) and load (red) faces
            st.markdown(
                _face_iso_svg(_conf_sup_specs, _conf_ld_specs),
                unsafe_allow_html=True,
            )

            run_col, cancel_col = st.columns(2)
            with run_col:
                if st.button("▶ Run Optimization", type="primary", use_container_width=True):
                    st.session_state["chat_state"] = "running"
                    st.rerun()
            with cancel_col:
                if st.button("✗ Try again", use_container_width=True):
                    st.session_state["chat_state"] = "idle"
                    st.session_state["pending_params"] = None
                    st.rerun()

        elif chat_state == "running":
            st.info("Running optimization… see results on the right →")

    else:
        # ── Expert mode: BC editor ────────────────────────────

        # Show a "defaults active" banner when no faces have been manually changed.
        _sup_faces = {s["face"] for s in st.session_state["support_specs"]}
        _ld_faces  = {l["face"] for l in st.session_state["load_specs"]}
        _using_defaults = (
            _sup_faces == {"Left (X=0)"}
            and _ld_faces == {"Right (X=W)"}
            and len(st.session_state["support_specs"]) == 1
            and len(st.session_state["load_specs"]) == 1
        )
        if _using_defaults:
            st.info(
                "**Default setup active:** Left face = Fixed support · Right face = 500 N downward load.  \n"
                "Click any face below to change the boundary conditions.",
                icon="ℹ️",
            )

        _c_canvas, _c_props = st.columns([3, 2])
        with _c_canvas:
            _pend_f = st.session_state.get("_pending_face")
            st.markdown(
                _face_iso_svg(st.session_state["support_specs"],
                              st.session_state["load_specs"], _pend_f),
                unsafe_allow_html=True,
            )
            st.caption("Select a face to configure:")
            _FACE_ROWS = [
                [("Left (X=0)", "Left"), ("Right (X=W)", "Right"), ("Bottom (Y=0)", "Bottom")],
                [("Top (Y=H)",  "Top"),  ("Front (Z=0)", "Front"),  ("Back (Z=D)",  "Back")],
            ]
            _STATUS_EMOJI = {"support": "🔵", "load": "🔴", "pending": "🟡", "free": "⬜"}
            for _row_idx, _row in enumerate(_FACE_ROWS):
                _btn_cols = st.columns(3)
                for _col_idx, (_fkey, _flabel) in enumerate(_row):
                    _fst  = _face_status(_fkey, st.session_state["support_specs"],
                                         st.session_state["load_specs"], _pend_f)
                    _femo = _STATUS_EMOJI[_fst]
                    _btn_type = "primary" if _fst == "pending" else "secondary"
                    with _btn_cols[_col_idx]:
                        if st.button(f"{_femo} {_flabel}",
                                     use_container_width=True,
                                     type=_btn_type,
                                     key=f"face_btn_{_row_idx}_{_col_idx}"):
                            st.session_state["_pending_face"] = _fkey
                            st.session_state["_pending_mode"] = None
                            st.rerun()

        with _c_props:
            _pending_face = st.session_state.get("_pending_face")
            _pending_mode = st.session_state.get("_pending_mode")

            if not _pending_face:
                st.info("← Click a face on the canvas to configure it.")
            else:
                st.markdown(f"**Selected: {_pending_face}**")
                _pb1, _pb2, _pb3 = st.columns(3)
                with _pb1:
                    if st.button("🔒 Support", use_container_width=True, key="setup_mode_sup",
                                 type="primary" if _pending_mode == "support" else "secondary"):
                        st.session_state["_pending_mode"] = "support"
                        st.rerun()
                with _pb2:
                    if st.button("⚡ Load", use_container_width=True, key="setup_mode_ld",
                                 type="primary" if _pending_mode == "load" else "secondary"):
                        st.session_state["_pending_mode"] = "load"
                        st.rerun()
                with _pb3:
                    if st.button("✕ Clear", use_container_width=True, key="setup_desel"):
                        st.session_state["_pending_face"] = None
                        st.session_state["_pending_mode"] = None
                        st.rerun()

                if _pending_mode == "support":
                    with st.container(border=True):
                        _sup_type_lbl = st.selectbox(
                            "Constraint type", list(_SUP_TYPES.values()), key="setup_sup_type"
                        )
                        _sup_type_k = {v: k for k, v in _SUP_TYPES.items()}[_sup_type_lbl]
                        if st.button("Add Support ✓", type="primary",
                                     use_container_width=True, key="setup_add_sup"):
                            _nsid = st.session_state["_support_id_ctr"]
                            st.session_state["_support_id_ctr"] += 1
                            st.session_state["support_specs"].append(
                                {"id": _nsid, "type": _sup_type_k, "face": _pending_face}
                            )
                            st.session_state["_pending_face"] = None
                            st.session_state["_pending_mode"] = None
                            st.rerun()

                elif _pending_mode == "load":
                    with st.container(border=True):
                        _ld_type_lbl = st.selectbox(
                            "Load type", list(_LD_TYPES.values()), key="setup_ld_type"
                        )
                        _ld_type_k = {v: k for k, v in _LD_TYPES.items()}[_ld_type_lbl]

                        _s_ufx, _s_ufy, _s_adeg = 0.0, -1.0, -90.0
                        _s_mag, _s_torque = 500, 5000
                        _s_u, _s_us, _s_ue = 0.5, 0.25, 0.75
                        _s_dir_label = _DIR_PRESET_KEYS[0]

                        if _ld_type_k != "moment":
                            _s_dir_label = st.selectbox(
                                "Direction", _DIR_PRESET_KEYS, key="setup_ld_dir"
                            )
                            if _DIR_PRESETS[_s_dir_label] is None:
                                _s_angle = st.slider("Angle (° from +X)", -180, 180, -90, 5,
                                                     key="setup_ld_angle",
                                                     help="0°=→  90°=↑  −90°=↓  180°=←")
                                _s_ufx = math.cos(math.radians(_s_angle))
                                _s_ufy = math.sin(math.radians(_s_angle))
                                _s_adeg = float(_s_angle)
                            else:
                                _s_ufx, _s_ufy, _s_adeg = _DIR_PRESETS[_s_dir_label]
                            _s_mag = st.slider("Magnitude (N)", 1, 5000, 500, 10, key="setup_ld_mag")
                            st.caption(f"Fx {_s_ufx*_s_mag:.0f} N · Fy {_s_ufy*_s_mag:.0f} N")
                        else:
                            _s_torque = st.slider("Torque (N·mm)", -10000, 10000, 5000, 500,
                                                  key="setup_ld_torque",
                                                  help="Positive = CCW  ·  Negative = CW")
                            st.caption("Equal & opposite forces at face extremes.")

                        if _ld_type_k == "point":
                            _s_u = st.slider("Position (%)", 0, 100, 50, 5, key="setup_ld_u",
                                             help="0% = bottom/left  ·  100% = top/right") / 100.0
                        elif _ld_type_k == "line":
                            _s_us = st.slider("From (%)", 0, 100, 25, 5, key="setup_ld_us") / 100.0
                            _s_ue = st.slider("To (%)",   0, 100, 75, 5, key="setup_ld_ue") / 100.0
                            if _s_ue <= _s_us:
                                st.warning("'To' must be greater than 'From'.")

                        if st.button("Add Load ✓", type="primary",
                                     use_container_width=True, key="setup_add_ld"):
                            _nlid = st.session_state["_load_id_ctr"]
                            st.session_state["_load_id_ctr"] += 1
                            _new_ld = {
                                "id": _nlid, "type": _ld_type_k, "face": _pending_face,
                                "u": _s_u, "u_start": _s_us, "u_end": _s_ue, "axis": "Z",
                            }
                            if _ld_type_k != "moment":
                                _new_ld.update({
                                    "fx": _s_ufx * _s_mag, "fy": _s_ufy * _s_mag,
                                    "_dir_label": _s_dir_label, "_magnitude": float(_s_mag),
                                    "_angle_deg": _s_adeg, "magnitude": 5000.0,
                                })
                            else:
                                _new_ld.update({
                                    "fx": 0.0, "fy": 0.0, "magnitude": float(_s_torque),
                                    "_dir_label": "🔄 Moment", "_magnitude": 0.0, "_angle_deg": 0.0,
                                })
                            st.session_state["load_specs"].append(_new_ld)
                            st.session_state["_pending_face"] = None
                            st.session_state["_pending_mode"] = None
                            st.rerun()

            st.divider()
            st.slider(
                "Safety Factor", 1.0, 5.0,
                float(st.session_state.get("_safety_factor", 2.0)), 0.5,
                key="_safety_factor",
                help="Allowable stress = Yield strength ÷ Safety Factor. 2.0 is standard.",
            )
            st.divider()
            _bl, _br = st.columns(2)
            with _bl:
                st.markdown("**🔒 Supports**")
                for _sii, _sup in enumerate(st.session_state["support_specs"]):
                    _sha, _sdb = st.columns([4, 1])
                    with _sha:
                        st.caption(f"{_sup['face']}\n*{_SUP_TYPES.get(_sup['type'],'?')}*")
                    with _sdb:
                        if (len(st.session_state["support_specs"]) > 1
                                and st.button("✕", key=f"bc_del_sup_{_sup['id']}")):
                            st.session_state["support_specs"].pop(_sii)
                            st.rerun()
            with _br:
                st.markdown("**⚡ Loads**")
                for _lii, _ldit in enumerate(st.session_state["load_specs"]):
                    _lha, _ldb = st.columns([4, 1])
                    with _lha:
                        if _ldit.get("type") == "moment":
                            st.caption(f"{_ldit['face']}\n*{_ldit.get('magnitude',0):.0f} N·mm*")
                        else:
                            _lm = math.sqrt(_ldit.get("fx",0)**2 + _ldit.get("fy",0)**2)
                            st.caption(f"{_ldit['face']}\n*{_lm:.0f} N*")
                    with _ldb:
                        if (len(st.session_state["load_specs"]) > 1
                                and st.button("✕", key=f"bc_del_ld_{_ldit['id']}")):
                            st.session_state["load_specs"].pop(_lii)
                            st.rerun()

        # ── Multi-Load Scenarios ──────────────────────────────
        _has_scenarios = bool(st.session_state["load_scenarios"])
        with st.expander(
            f"Multi-Load Scenarios {'(' + str(len(st.session_state['load_scenarios'])) + ' active)' if _has_scenarios else '(optional)'}",
            expanded=_has_scenarios,
        ):
            st.caption(
                "Define multiple simultaneous load cases with individual weights. "
                "The optimizer produces a topology stiff under all of them at once. "
                "Leave empty to use the single load above."
            )

            _DIRECTIONS = ["-Y (Down)", "+Y (Up)", "-X (Left)", "+X (Right)", "-Z (Front)", "+Z (Back)"]
            _DIR_STR_MAP = {d.split()[0]: d for d in _DIRECTIONS}
            _DIR_RMAP = {v: k for k, v in _DIR_STR_MAP.items()}

            # Header row
            if st.session_state["load_scenarios"]:
                _hc = st.columns([3, 2, 2, 2, 2, 1])
                for _hi, _hl in enumerate(["Name", "Face", "Direction", "Force (N)", "Weight", ""]):
                    _hc[_hi].markdown(f"<div style='font-size:0.7rem;color:#546e7a;text-transform:uppercase;'>{_hl}</div>", unsafe_allow_html=True)

            _scenarios = st.session_state["load_scenarios"]
            for _sci, _sc in enumerate(_scenarios):
                _scc = st.columns([3, 2, 2, 2, 2, 1])
                _sc["name"]      = _scc[0].text_input("n", _sc.get("name", f"Scenario {_sci+1}"),
                                                       key=f"sc_name_{_sc['id']}", label_visibility="collapsed")
                _sc["face"]      = _scc[1].selectbox("f", FACES, index=FACES.index(_sc.get("face", FACES[1])),
                                                      key=f"sc_face_{_sc['id']}", label_visibility="collapsed")
                _cur_dir_str     = _DIR_STR_MAP.get(_sc.get("direction", "-Y"), _DIRECTIONS[0])
                _sel_dir_str     = _scc[2].selectbox("d", _DIRECTIONS, index=_DIRECTIONS.index(_cur_dir_str),
                                                      key=f"sc_dir_{_sc['id']}", label_visibility="collapsed")
                _sc["direction"] = _DIR_RMAP[_sel_dir_str]
                _sc["magnitude"] = _scc[3].number_input("m", 1, 100000, int(_sc.get("magnitude", 500)),
                                                          key=f"sc_mag_{_sc['id']}", label_visibility="collapsed")
                _sc["weight"]    = _scc[4].slider("w", 0.1, 1.0, float(_sc.get("weight", 1.0)), 0.1,
                                                   key=f"sc_w_{_sc['id']}", label_visibility="collapsed")
                if _scc[5].button("✕", key=f"sc_del_{_sc['id']}"):
                    _scenarios.pop(_sci)
                    st.rerun()

            if st.button("+ Add Load Scenario", key="sc_add_btn", use_container_width=True):
                _sid = st.session_state["_scenario_id_ctr"]
                st.session_state["_scenario_id_ctr"] += 1
                _scenarios.append({
                    "id": _sid, "name": f"Scenario {len(_scenarios)+1}",
                    "face": "Right (X=W)", "direction": "-Y", "magnitude": 500, "weight": 1.0,
                })
                if not _has_scenarios:
                    # Auto-seed first scenario from current primary load
                    _l0 = st.session_state["load_specs"][0] if st.session_state["load_specs"] else {}
                    _lm = math.sqrt(_l0.get("fx",0)**2 + _l0.get("fy",0)**2) or 500.0
                    _scenarios[0].update({
                        "name": "Primary", "face": _l0.get("face","Right (X=W)"),
                        "direction": "-Y", "magnitude": int(_lm),
                    })
                st.rerun()

            if _has_scenarios:
                _tw = sum(s.get("weight", 1.0) for s in _scenarios)
                st.caption(f"Total weight: {_tw:.1f} (auto-normalised to 1.0 at run time)")

# ─────────────────────────────────────────────────────────────
#  RIGHT COLUMN — viewer + execution
# ─────────────────────────────────────────────────────────────
with _right_col:

    # ── Chat mode: Run optimizer (running state) ─────────────
    if st.session_state["chat_state"] == "running":
        p = st.session_state["pending_params"]
        _c_applied_force_n = float(p["applied_force_n"])
        _c_safety_factor   = float(p["safety_factor"])
        _c_sel_mat         = p.get("material", sel_mat)
        _c_vf              = float(p.get("volume_fraction", volume_fraction))
        _c_mean_nu         = float(MATERIALS[_c_sel_mat]["nu"])

        # Build load/support specs from pending_params (face + direction + magnitude)
        # This is the definitive fix: no longer uses the legacy named-scenario string.
        _c_ld_specs, _c_sup_specs = _pending_to_specs(p)
        _c_load_cases, _c_opt_fixed, _c_F_mag = build_load_cases(
            _c_ld_specs, _c_sup_specs, nelx, nely
        )

        _c_design_mask = None
        if st.session_state["design_mode"] == "mesh" and st.session_state["uploaded_mesh"]:
            _c_raw = voxelize_mesh(st.session_state["uploaded_mesh"], nelx, nely, 8)
            _c_design_mask = _c_raw.any(axis=2) if _c_raw.ndim == 3 else _c_raw
        st.session_state["design_mask"] = _c_design_mask

        _c_mat_color = MATERIALS[_c_sel_mat]["color"]
        _c_prog = st.progress(0, text=f"Building filter — {nelx}×{nely} mesh…")
        _c_live_cols = st.columns([2, 1])
        _c_dens_ph = _c_live_cols[0].empty()
        _c_met_ph  = _c_live_cols[1].empty()

        def _c_live_cb(iteration, _max, compliance, _xPhys, change):
            _c_prog.progress(min(int(iteration / _max * 100), 99),
                text=f"Iter {iteration}/{_max} · C = {compliance:.4f}")
            if iteration % max(1, _max // 15) == 0 or iteration <= 3:
                _fl = go.Figure(go.Heatmap(
                    z=_xPhys,
                    colorscale=[[0,"#050508"],[0.35,"#1a1a2e"],[1,_c_mat_color]],
                    showscale=False, zmin=0, zmax=1,
                ))
                _fl.update_layout(
                    xaxis=dict(scaleanchor="y",showticklabels=False,showgrid=False),
                    yaxis=dict(autorange="reversed",showticklabels=False,showgrid=False),
                    height=180, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0e1117",
                    title=dict(text=f"Iter {iteration} · density field",
                               font=dict(color="#546e7a",size=10), x=0.5, y=0.98),
                )
                _c_dens_ph.plotly_chart(_fl, use_container_width=True, key=f"clive_{iteration}")
            with _c_met_ph.container():
                st.metric("Iteration",  f"{iteration}/{_max}")
                st.metric("Compliance", f"{compliance:.4f}")
                st.metric("Change",     f"{change:.4f}")

        with st.spinner("Running SIMP optimizer…"):
            _c_H, _c_Hs = build_filter_fast(nelx, nely, rmin)
            _c_xPhys, _c_hist, _c_sf = simp_core(
                nelx, nely, _c_vf, _c_mean_nu, penal, _c_H, _c_Hs, max_iter,
                _c_load_cases, _c_opt_fixed, yield_callback=_c_live_cb,
                design_mask=_c_design_mask,
            )
        _c_prog.progress(100, text=f"Done — {len(_c_hist)} iterations")
        _c_dens_ph.empty(); _c_met_ph.empty()

        # Scale chat-mode stress to physical MPa using the preserved F_mag from build_load_cases
        if _c_sf is not None and _c_F_mag > 0:
            _c_dx_mm = box_w / nelx
            _c_sf = _c_sf * (_c_F_mag / _c_dx_mm)

        _c_comp       = _c_hist[-1]
        _c_cdrop      = round((_c_hist[0]-_c_hist[-1])/_c_hist[0]*100,1) if len(_c_hist)>=2 else 0
        _c_avf        = float(_c_xPhys.mean())
        _c_scores     = compute_mat_scores([_c_sel_mat], box_w, box_h, box_d,
                                           _c_vf, _c_applied_force_n, _c_safety_factor, _c_comp)
        _c_stl, _c_nf = to_stl_bytes(_c_xPhys, box_w, box_h, box_d, iso_threshold)
        _c_meta = {
            "nelx": nelx, "nely": nely, "vf": _c_vf, "achieved_vf": _c_avf,
            "penal": penal, "rmin": rmin, "iters": len(_c_hist),
            "compliance": round(_c_comp,6), "compliance_drop": _c_cdrop,
            "stiffness_index": 0, "stiffness_label": "—",
            "mean_nu": _c_mean_nu, "load_scenario": _load_summary(_c_ld_specs), "n_lc": len(_c_load_cases),
            "iso": iso_threshold, "n_faces": _c_nf, "export_mat": _c_sel_mat,
            "box": (box_w, box_h, box_d),
        }
        st.session_state.update({
            "xPhys": _c_xPhys, "history": _c_hist, "history_lc2": [],
            "stress_field": _c_sf, "stl_bytes": _c_stl, "n_faces": _c_nf,
            "mode": "Single Material", "mat_scores": _c_scores,
            "winners": {}, "export_mat": _c_sel_mat,
            "compare_mats": [_c_sel_mat], "opt_meta": _c_meta,
        })
        _c_massg = round((box_w/10)*(box_h/10)*(box_d/10)*_c_avf*MATERIALS[_c_sel_mat]["rho_gcc"],1)
        st.session_state["design_history"].insert(0, make_history_record(
            part_name, _c_sel_mat, f"{box_w}x{box_h}x{box_d}",
            _c_vf, _load_summary(_c_ld_specs), _c_comp, len(_c_hist),
            _c_nf if _c_stl else 0, _c_massg
        ))
        st.session_state["chat_messages"].append({"role": "assistant", "content":
            f"Done! {len(_c_hist)} iterations · Compliance {_c_comp:.3f} · "
            f"{_c_cdrop}% improvement · {_c_nf:,} triangles."})
        st.session_state["chat_state"]   = "idle"
        st.session_state["pending_params"] = None
        st.session_state["_right_view"]  = "🔷 Topology"
        st.rerun()

    # ── Expert mode: Run optimizer ────────────────────────────
    elif run_btn:
        mean_nu   = float(np.mean([MATERIALS[n]["nu"] for n in compare_mats]))
        _use_3d   = st.session_state.get("use_3d_simp", False)
        _nelz_run = int(st.session_state.get("_preset_nelz", 8))

        _active_scenarios = st.session_state.get("load_scenarios", [])

        if _use_3d:
            load_cases, opt_fixed_dofs = build_load_cases_3d(
                st.session_state["load_specs"], st.session_state["support_specs"],
                nelx, nely, _nelz_run,
            )
            _F_mag = float(applied_force_n)  # 3-D path: no normalization yet
            n_lc = len(load_cases)
            H, Hs = build_filter_3d(nelx, nely, _nelz_run, rmin)
        elif _active_scenarios:
            load_cases, opt_fixed_dofs, _F_mag = build_load_cases_multi(
                _active_scenarios, st.session_state["support_specs"], nelx, nely,
            )
            n_lc = len(load_cases)
            H, Hs = build_filter_fast(nelx, nely, rmin)
        else:
            load_cases, opt_fixed_dofs, _F_mag = build_load_cases(
                st.session_state["load_specs"], st.session_state["support_specs"],
                nelx, nely,
            )
            n_lc = len(load_cases)
            H, Hs = build_filter_fast(nelx, nely, rmin)

        _run_design_mask = None
        if st.session_state["design_mode"] == "mesh" and st.session_state["uploaded_mesh"]:
            with st.spinner("Voxelizing uploaded mesh…"):
                _raw_mask = voxelize_mesh(
                    st.session_state["uploaded_mesh"], nelx, nely, _nelz_run if _use_3d else 8
                )
            _run_design_mask = _raw_mask.any(axis=2) if (not _use_3d and _raw_mask.ndim==3) else _raw_mask
            _in_pct = int(_run_design_mask.sum() / _run_design_mask.size * 100)
            st.info(f"Mesh uploaded: {_in_pct}% of grid elements inside design domain.")
        st.session_state["design_mask"] = _run_design_mask

        _prog_bar   = st.progress(0, text=f"Building filter — {nelx}×{nely} mesh…")
        _live_cols  = st.columns([2, 1])
        _density_ph = _live_cols[0].empty()
        _metrics_ph = _live_cols[1].empty()

        def _live_cb(iteration, _max, compliance, _xPhys, change):
            pct = min(int(iteration / _max * 100), 99)
            _prog_bar.progress(pct, text=f"Iter {iteration}/{_max} · C = {compliance:.4f} · Δ = {change:.4f}")
            if iteration % max(1, _max // 15) == 0 or iteration <= 3:
                _mat_color = MATERIALS[sel_mat]["color"]
                _fig_live = go.Figure(go.Heatmap(
                    z=_xPhys,
                    colorscale=[[0,"#050508"],[0.35,"#1a1a2e"],[1,_mat_color]],
                    showscale=False, zmin=0, zmax=1,
                ))
                _fig_live.update_layout(
                    xaxis=dict(scaleanchor="y",showticklabels=False,showgrid=False),
                    yaxis=dict(autorange="reversed",showticklabels=False,showgrid=False),
                    height=180, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0e1117",
                    title=dict(text=f"Iter {iteration} · density field",
                               font=dict(color="#546e7a",size=10), x=0.5, y=0.98),
                )
                _density_ph.plotly_chart(_fig_live, use_container_width=True, key=f"live_{iteration}")
            with _metrics_ph.container():
                st.metric("Iteration",  f"{iteration}/{_max}")
                st.metric("Compliance", f"{compliance:.4f}")
                st.metric("Change",     f"{change:.4f}")

        if _use_3d:
            xPhys, history, stress_field = simp_core_3d(
                nelx, nely, _nelz_run, volume_fraction, mean_nu, penal, H, Hs, max_iter,
                load_cases, opt_fixed_dofs, yield_callback=_live_cb,
            )
        else:
            xPhys, history, stress_field = simp_core(
                nelx, nely, volume_fraction, mean_nu, penal, H, Hs, max_iter,
                load_cases, opt_fixed_dofs, yield_callback=_live_cb,
                design_mask=_run_design_mask,
            )
        _prog_bar.progress(100, text=f"Done — {len(history)} iterations")
        _density_ph.empty(); _metrics_ph.empty()
        history_lc2 = []
        norm_comp   = history[-1]

        # Scale stress to physical MPa: σ_phys = σ_norm × F_mag / (dx_mm × thickness_mm)
        # dx_mm = box_w / nelx (element width). Thickness = 1 mm (plane-stress assumption).
        # For 3-D, stress is already in relative units — skip scaling.
        if not _use_3d and stress_field is not None:
            # σ_phys [MPa] = σ_norm × F_mag [N] / (dx_mm [mm] × thickness [mm])
            # Derivation: displacement scales as F_mag/(E×t), B scales as 1/dx,
            # so σ = E × B × u cancels E → σ ∝ F_mag/(dx × t).
            _dx_mm = box_w / nelx
            _stress_scale = _F_mag / (_dx_mm * 1.0)  # thickness = 1 mm (plane-stress)
            stress_field = stress_field * _stress_scale

        if mode == "Material Comparison":
            mat_scores = compute_mat_scores(compare_mats, box_w, box_h, box_d,
                                            volume_fraction, applied_force_n, safety_factor, norm_comp)
            winners    = rank_materials(mat_scores)
            export_mat = winners.get("best_ashby", compare_mats[0])
        else:
            mat_scores = compute_mat_scores([sel_mat], box_w, box_h, box_d,
                                            volume_fraction, applied_force_n, safety_factor, norm_comp)
            winners    = {}
            export_mat = sel_mat

        stl_bytes, n_faces = to_stl_bytes(xPhys, box_w, box_h, box_d, iso_threshold)
        achieved_vf        = float(xPhys.mean())
        compliance_drop    = round((history[0]-history[-1])/history[0]*100,1) if len(history)>=2 else 0
        _sp_r = float((xPhys>0.5).sum())/xPhys.size*100
        _vp_r = float((xPhys<0.2).sum())/xPhys.size*100
        _bim_r = _sp_r + _vp_r
        _stiff_idx   = min(100, round(compliance_drop*0.65 + _bim_r*0.35))
        _stiff_label = ("Optimal" if _stiff_idx>=95 else "Excellent" if _stiff_idx>=80 else
                        "Good" if _stiff_idx>=65 else "Moderate" if _stiff_idx>=40 else "Mild")
        meta = {
            "nelx": nelx, "nely": nely, "vf": volume_fraction, "achieved_vf": achieved_vf,
            "penal": penal, "rmin": rmin, "iters": len(history),
            "compliance": round(norm_comp,6), "compliance_drop": compliance_drop,
            "stiffness_index": _stiff_idx, "stiffness_label": _stiff_label,
            "mean_nu": mean_nu, "load_scenario": (
                " + ".join(f"{s['name']} ({s.get('weight',1.0):.1f}×)" for s in _active_scenarios)
                if _active_scenarios else _load_summary(st.session_state["load_specs"])
            ),
            "n_lc": n_lc, "iso": iso_threshold, "n_faces": n_faces, "export_mat": export_mat,
            "box": (box_w, box_h, box_d), "use_3d": _use_3d, "nelz": _nelz_run if _use_3d else None,
        }
        st.session_state.update({
            "xPhys": xPhys, "history": history, "history_lc2": history_lc2,
            "stress_field": stress_field, "stl_bytes": stl_bytes, "n_faces": n_faces,
            "mode": mode, "mat_scores": mat_scores, "winners": winners,
            "export_mat": export_mat, "compare_mats": compare_mats, "opt_meta": meta,
        })
        _hist_scenario = (
            " + ".join(s["name"] for s in _active_scenarios)
            if _active_scenarios else _load_summary(st.session_state["load_specs"])
        )
        st.session_state["design_history"].insert(0, make_history_record(
            part_name, export_mat, f"{box_w}x{box_h}x{box_d}",
            volume_fraction, _hist_scenario,
            norm_comp, len(history), n_faces if stl_bytes else 0, est_mass_g
        ))
        if len(st.session_state["design_history"]) > 20:
            st.session_state["design_history"] = st.session_state["design_history"][:20]

        _run_msg = f"{len(history)} iters · Compliance {norm_comp:.3f} · {compliance_drop}% stiffer"
        _run_msg += f" · {n_faces:,} triangles" if stl_bytes else " · No STL — lower iso-level"
        st.success(_run_msg)
        _bim2 = float((xPhys>0.5).sum())/xPhys.size*100 + float((xPhys<0.2).sum())/xPhys.size*100
        if _bim2 < 65:
            st.info(f"Grey zone {100-_bim2:.0f}% — try SIMP Penalty 4.0+ for sharper result.")
        if not stl_bytes:
            st.info(f"No STL: lower iso-level to ~{max(0.1, round(xPhys.max()-0.1, 2))}.")
        if _bim2 >= 85:
            st.info(f"Sharp result — {_bim2:.0f}% cleanly solid/void. Ready for infill.")

        st.session_state["_right_view"] = "🔷 Topology"
        st.rerun()

    # ── Infill generation ─────────────────────────────────────
    elif infill_btn:
        if not _has_results:
            st.warning("Run the optimizer first.")
        else:
            xPhys_i = st.session_state["xPhys"]
            with st.spinner(f"Generating {infill_pattern} infill…"):
                i_stl, i_faces, i_vf, i_sf = generate_infill(
                    xPhys_i, box_w, box_h, box_d, infill_pattern,
                    period_mm=period_mm, void_threshold=void_thresh,
                    solid_threshold=solid_thresh,
                    fine_nx=fine_nx, fine_ny=fine_ny, fine_nz=fine_nz,
                )
            st.session_state.update({
                "infill_stl": i_stl, "infill_faces": i_faces,
                "infill_vf": i_vf, "infill_sf": i_sf,
                "infill_pattern": infill_pattern, "infill_period": period_mm,
            })
            if i_stl:
                _em_i = st.session_state.get("export_mat", sel_mat)
                _im_mass = round((box_w/10)*(box_h/10)*(box_d/10)*i_vf*MATERIALS[_em_i]["rho_gcc"],1)
                if st.session_state["design_history"]:
                    st.session_state["design_history"][0]["infill"]    = infill_pattern
                    st.session_state["design_history"][0]["infill_vf"] = round(i_vf*100, 0)
                st.success(f"Infill done: {i_faces:,} tri · {round(len(i_stl)/1024,1)} KB · "
                           f"{i_vf*100:.0f}% solid · {_im_mass}g")
                st.session_state["_right_view"] = "🧩 Infill"
                st.rerun()
            else:
                st.error("Infill failed — try lowering void threshold or changing period.")

    # ── VIEW SELECTOR + CONTENT ───────────────────────────────
    else:
        if not _has_results and not _landing:
            # ── Design preview (no results yet) ───────────────
            st.caption("Design space · boundary conditions preview")
            _fig_pre = go.Figure()
            for _t in design_space_box(box_w, box_h, box_d):
                _fig_pre.add_trace(_t)
            _fig_pre.add_trace(solid_box_mesh(box_w, box_h, box_d, color="#37474f", opacity=0.06))
            _fig_pre.add_trace(face_trace(fixed_face, box_w, box_h, box_d, "royalblue", 0.4, "Fixed"))
            _fig_pre.add_trace(face_trace(load_face,  box_w, box_h, box_d, "crimson",   0.3, "Load"))
            _ptx, _pty, _ptz = face_center(load_face, box_w, box_h, box_d)
            _pal, _pac, _pan = arrow_traces(_ptx, _pty, _ptz, force_direction, force_n=applied_force_n)
            _fig_pre.add_trace(_pal); _fig_pre.add_trace(_pac); _fig_pre.add_trace(_pan)
            _ftx, _fty, _ftz = face_center(fixed_face, box_w, box_h, box_d)
            _fig_pre.add_trace(fixed_label_trace(_ftx, _fty, _ftz))
            # Uploaded mesh ghost
            _dm_pre = st.session_state.get("design_mask")
            if _dm_pre is not None:
                for _t in mesh_traces(_dm_pre.astype(np.float32), box_w, box_h, box_d,
                                      iso=0.5, color="#29b6f6", name="Design domain"):
                    _t.opacity = 0.15
                    _fig_pre.add_trace(_t)
            _fig_pre.update_layout(**scene3d(box_w, box_h, box_d, 480))
            st.plotly_chart(_fig_pre, use_container_width=True)
            if not expert_mode:
                st.markdown("""
<div style='background:linear-gradient(135deg,#0d1929,#0e1117);border:1px solid rgba(21,101,192,0.25);
border-radius:10px;padding:22px 28px;margin-top:8px;'>
<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
color:#1565c0;margin-bottom:10px;'>How it works</div>
<div style='display:flex;gap:24px;flex-wrap:wrap;'>
  <div style='flex:1;min-width:140px;'>
    <div style='font-size:1.3rem;margin-bottom:4px;color:#42a5f5;font-weight:700;'>1</div>
    <div style='font-size:0.85rem;font-weight:600;color:#cfd8dc;margin-bottom:2px;'>Describe your part</div>
    <div style='font-size:0.78rem;color:#546e7a;line-height:1.5;'>Tell the AI what it is, what loads it carries, what face is fixed.</div>
  </div>
  <div style='flex:1;min-width:140px;'>
    <div style='font-size:1.3rem;margin-bottom:4px;color:#42a5f5;font-weight:700;'>2</div>
    <div style='font-size:0.85rem;font-weight:600;color:#cfd8dc;margin-bottom:2px;'>Optimizer runs</div>
    <div style='font-size:0.78rem;color:#546e7a;line-height:1.5;'>SIMP algorithm removes material that isn't carrying load, keeping exactly what's needed.</div>
  </div>
  <div style='flex:1;min-width:140px;'>
    <div style='font-size:1.3rem;margin-bottom:4px;color:#4CAF50;font-weight:700;'>3</div>
    <div style='font-size:0.85rem;font-weight:600;color:#cfd8dc;margin-bottom:2px;'>Export & print</div>
    <div style='font-size:0.78rem;color:#546e7a;line-height:1.5;'>Download the STL. Load-bearing paths are shown. Stress is in real MPa.</div>
  </div>
</div>
<div style='margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.06);
font-size:0.76rem;color:#37474f;'>
Try a <strong style='color:#546e7a;'>Quick Start preset</strong> on the left to see results in under 60 seconds — no API key needed.
</div>
</div>""", unsafe_allow_html=True)
        else:
            # ── View selector: 3 primary tabs + More drawer ────
            _rv_primary = ["🔷 Topology", "📈 Convergence", "💾 Export"]
            _rv_secondary = ["🧩 Infill", "🗺️ Density Map", "📊 Pareto",
                             "🖨️ Print Est.", "📋 History", "❓ Help"]
            if result_mode == "Material Comparison":
                _rv_secondary.insert(0, "📊 Compare")

            _rv_opts = _rv_primary + _rv_secondary  # full list for validation

            _rv_stored = st.session_state.get("_right_view", "🔷 Topology")
            if _rv_stored not in _rv_opts:
                _rv_stored = "🔷 Topology"

            _tab_col, _more_col = st.columns([6, 2])
            with _tab_col:
                # Primary tabs as pill-style radio
                _primary_val = _rv_stored if _rv_stored in _rv_primary else _rv_primary[0]
                _right_view_radio = st.radio(
                    "view_sel", _rv_primary,
                    index=_rv_primary.index(_primary_val) if _rv_stored in _rv_primary else 0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="_right_view_primary",
                )
            with _more_col:
                # Secondary views in a compact selectbox
                _more_label = "More views…" if _rv_stored in _rv_primary else _rv_stored
                _more_choice = st.selectbox(
                    "More:", ["More views…"] + _rv_secondary,
                    index=(["More views…"] + _rv_secondary).index(_rv_stored)
                          if _rv_stored in _rv_secondary else 0,
                    label_visibility="collapsed",
                    key="_right_view_more",
                )

            # Resolve which view is active (secondary takes precedence when changed)
            if _more_choice != "More views…" and _more_choice != _rv_stored:
                _right_view = _more_choice
                st.session_state["_right_view"] = _right_view
            elif _rv_stored in _rv_secondary:
                _right_view = _rv_stored
            else:
                _right_view = _right_view_radio
                st.session_state["_right_view"] = _right_view

            # ── Load result session state ──────────────────────
            xPhys        = st.session_state["xPhys"]
            history      = st.session_state["history"]
            history_lc2  = st.session_state.get("history_lc2", [])
            stress_field = st.session_state.get("stress_field")
            stl_bytes    = st.session_state["stl_bytes"]
            n_faces      = st.session_state["n_faces"]
            meta         = st.session_state["opt_meta"]
            mat_scores   = st.session_state["mat_scores"]
            winners      = st.session_state["winners"]
            export_mat   = st.session_state["export_mat"]
            compare_mats_saved = st.session_state.get("compare_mats", [sel_mat])
            improvement  = meta["compliance_drop"]
            infill_stl   = st.session_state.get("infill_stl")
            infill_faces = st.session_state.get("infill_faces", 0)
            infill_vf    = st.session_state.get("infill_vf", 0.0)
            infill_sf    = st.session_state.get("infill_sf")
            i_pattern    = st.session_state.get("infill_pattern", infill_pattern)
            export_color = MATERIALS[export_mat]["color"]
            em_props     = MATERIALS[export_mat]

            topo_mass   = round((box_w/10)*(box_h/10)*(box_d/10)*meta["achieved_vf"]*em_props["rho_gcc"],1)
            infill_mass = round((box_w/10)*(box_h/10)*(box_d/10)*infill_vf*em_props["rho_gcc"],1) if infill_vf else topo_mass
            active_mass = infill_mass if infill_stl else topo_mass
            print_est   = estimate_print(active_mass, em_props["rho_gcc"], layer_h, speed_ms,
                                         cost_per_kg=em_props["cost_per_kg"])

            # ── 3D Topology view ───────────────────────────────
            if _right_view == "🔷 Topology":
                # ── Status strip ──────────────────────────────
                _si       = meta.get("stiffness_index", 0)
                _sl       = meta.get("stiffness_label", "—")
                _si_color = ("#4CAF50" if _si>=80 else "#8BC34A" if _si>=65 else
                             "#FFC107" if _si>=40 else "#FF5722")
                _imp_str  = f"+{improvement}% stiffer"
                _mass_str = f"{topo_mass} g"
                _fill_str = f"{meta['achieved_vf']*100:.0f}% material"
                _mesh_str = f"{n_faces:,} triangles" if stl_bytes else "export pending"
                # Stress pass/fail vs allowable
                _sf_field = st.session_state.get("stress_field")
                if _sf_field is not None and _sf_field.max() > 0:
                    _vm_max = float(_sf_field.max())
                    _stress_ok = _vm_max <= allowable_stress
                    _stress_str = f"{'✓' if _stress_ok else '✗'} {_vm_max:.0f} / {allowable_stress:.0f} MPa"
                    _stress_cls = "good" if _stress_ok else "warn"
                else:
                    _stress_str, _stress_cls = f"≤ {allowable_stress} MPa", "info"
                st.markdown(
                    f"""<div class="morph-status">
  <div class="ms-item">
    <div class="ms-label">Stiffness gain</div>
    <div class="ms-value good">{_imp_str}</div>
  </div>
  <div class="ms-item">
    <div class="ms-label">Mass</div>
    <div class="ms-value">{_mass_str}</div>
  </div>
  <div class="ms-item">
    <div class="ms-label">Fill density</div>
    <div class="ms-value">{_fill_str}</div>
  </div>
  <div class="ms-item">
    <div class="ms-label">Max stress (SF {safety_factor}×)</div>
    <div class="ms-value {_stress_cls}">{_stress_str}</div>
  </div>
  <div class="ms-item">
    <div class="ms-label">Quality</div>
    <div class="ms-value" style="color:{_si_color};">{_sl} · {_si}/100</div>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

                # ── Color mode selector ───────────────────────
                _cm_col, _cm_adv_col = st.columns([4, 4])
                with _cm_col:
                    colormode_radio = st.radio(
                        "View", ["Topology", "Stress (MPa)"],
                        horizontal=True, key="viz_colormode",
                        help="Topology = optimized shape colored by material. Stress = von Mises heat map.",
                    )
                with _cm_adv_col:
                    _adv_view = st.selectbox(
                        "Advanced",
                        ["—", "Density field", "Element mesh"],
                        key="viz_adv_mode",
                        label_visibility="collapsed",
                        help="Advanced visualization modes",
                    )

                # ── Full-width 3D viewer ──────────────────────
                fig3d = go.Figure()
                _ba_trace = solid_box_mesh(box_w, box_h, box_d, color="#607d8b", opacity=0.18)
                _ba_trace.visible = False
                fig3d.add_trace(_ba_trace)
                before_traces = [_ba_trace]

                # Advanced modes override the primary selector
                _adv = st.session_state.get("viz_adv_mode", "—")
                if _adv == "Density field":
                    for t in density_isosurface_traces(xPhys, box_w, box_h, box_d):
                        fig3d.add_trace(t)
                elif _adv == "Element mesh":
                    for t in element_mesh_traces(xPhys, box_w, box_h, box_d, threshold=meta["iso"]):
                        fig3d.add_trace(t)
                else:
                    colormode = "stress" if colormode_radio == "Stress (MPa)" else "density"
                    for t in mesh_traces(
                        xPhys, box_w, box_h, box_d, meta["iso"], export_color,
                        "Topology", stress_field=stress_field, colormode=colormode
                    ):
                        fig3d.add_trace(t)

                _dm_ss = st.session_state.get("design_mask")
                if _dm_ss is not None:
                    for t in mesh_traces(
                        _dm_ss.astype(np.float32), box_w, box_h, box_d,
                        iso=0.5, color="#29b6f6", name="Design domain",
                    ):
                        t.opacity = 0.12
                        t.name = "Uploaded design space"
                        fig3d.add_trace(t)

                for t in design_space_box(box_w, box_h, box_d):
                    fig3d.add_trace(t)
                fig3d.add_trace(face_trace(fixed_face, box_w, box_h, box_d, "royalblue", 0.3, "Fixed"))
                fig3d.add_trace(face_trace(load_face,  box_w, box_h, box_d, "crimson",   0.2, "Load"))
                tx, ty, tz = face_center(load_face, box_w, box_h, box_d)
                al, ac, annot = arrow_traces(tx, ty, tz, force_direction, force_n=applied_force_n)
                fig3d.add_trace(al); fig3d.add_trace(ac); fig3d.add_trace(annot)
                ftx, fty, ftz = face_center(fixed_face, box_w, box_h, box_d)
                fig3d.add_trace(fixed_label_trace(ftx, fty, ftz))

                n_before  = len(before_traces)
                n_total   = len(fig3d.data)
                vis_opt   = [False]*n_before + [True]*(n_total-n_before)
                vis_both  = [True]*n_total
                _is_true3d = meta.get("use_3d", False)
                _simp_badge = "3D·SIMP" if _is_true3d else "2D·SIMP extruded"
                fig3d.update_layout(
                    **scene3d(box_w, box_h, box_d, 640),
                    updatemenus=[dict(
                        buttons=[
                            dict(label="Optimized",      method="update", args=[{"visible": vis_opt}]),
                            dict(label="Before / After", method="update", args=[{"visible": vis_both}]),
                        ],
                        direction="left", showactive=True,
                        x=0.0, y=1.08, xanchor="left",
                        bgcolor="rgba(20,28,45,0.85)", font=dict(color="white", size=12),
                    )],
                    annotations=[dict(
                        text=_simp_badge,
                        xref="paper", yref="paper", x=1.0, y=0.0,
                        xanchor="right", yanchor="bottom",
                        showarrow=False,
                        font=dict(size=10, color="rgba(100,120,150,0.8)"),
                        bgcolor="rgba(10,14,24,0.6)",
                        borderpad=3,
                    )],
                )
                st.plotly_chart(fig3d, use_container_width=True)

                # ── Sculpt animation ──────────────────────────
                with st.expander("Sculpt animation — watch SIMP carve the solid block", expanded=False):
                    _sculpt_key = (id(st.session_state["xPhys"]), box_w, box_h, box_d, export_color)
                    if st.session_state.get("_sculpt_cache_key") != _sculpt_key:
                        with st.spinner("Building sculpt frames… (one-time, cached)"):
                            _sculpt_frames, _sculpt_thrs = sculpt_solid_frames(
                                xPhys, box_w, box_h, box_d, export_color, n_frames=18
                            )
                        st.session_state["_sculpt_frames"]    = _sculpt_frames
                        st.session_state["_sculpt_thrs"]      = _sculpt_thrs
                        st.session_state["_sculpt_cache_key"] = _sculpt_key
                    _sculpt_frames = st.session_state["_sculpt_frames"]
                    _sculpt_thrs   = st.session_state["_sculpt_thrs"]
                    _sculpt_steps = [
                        dict(method="animate",
                             args=[[f.name], {"mode":"immediate","frame":{"duration":0,"redraw":True},
                                              "transition":{"duration":0}}],
                             label=f"{_sculpt_thrs[i]:.2f}")
                        for i, f in enumerate(_sculpt_frames)
                    ]
                    _fig_sc = go.Figure(data=_sculpt_frames[0].data, frames=_sculpt_frames)
                    _sc_layout = scene3d(box_w, box_h, box_d, 450)
                    _sc_layout.update(
                        updatemenus=[dict(
                            type="buttons", showactive=False,
                            x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                            bgcolor="rgba(14,20,35,0.9)", font=dict(color="white",size=12),
                            bordercolor="rgba(255,255,255,0.15)",
                            buttons=[
                                dict(label="▶ Play", method="animate",
                                     args=[None, {"frame":{"duration":130,"redraw":True},
                                                   "fromcurrent":True,"transition":{"duration":0}}]),
                                dict(label="⏸", method="animate",
                                     args=[[None], {"frame":{"duration":0,"redraw":False},
                                                    "mode":"immediate","transition":{"duration":0}}]),
                            ],
                        )],
                        sliders=[dict(
                            active=0,
                            currentvalue=dict(prefix="Density cutoff: ",
                                              font=dict(color="#90caf9",size=13), visible=True),
                            pad=dict(t=55,b=5,l=60,r=20),
                            font=dict(color="#90a4ae",size=10),
                            bgcolor="rgba(14,20,35,0.8)",
                            bordercolor="rgba(255,255,255,0.1)",
                            tickcolor="rgba(255,255,255,0.25)",
                            steps=_sculpt_steps,
                        )],
                    )
                    _fig_sc.update_layout(**_sc_layout)
                    st.plotly_chart(_fig_sc, use_container_width=True, key="sculpt_3d")
                    st.caption("Slider: left=solid block · right=structural skeleton · ▶ Play animates carving.")

                # ── Summary panel (collapsible) ───────────────
                with st.expander("Result summary & AI analysis", expanded=False):
                    _info1, _info2 = st.columns([1, 1])
                    with _info1:
                        # Stiffness index bar
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);"
                            f"border-radius:7px;padding:10px 14px;margin-bottom:8px;'>"
                            f"<div style='font-size:0.72rem;color:#78909c;'>Stiffness Index</div>"
                            f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:2px;'>"
                            f"<span style='font-size:1.8rem;font-weight:800;color:{_si_color};'>{_si}</span>"
                            f"<span style='font-size:0.9rem;color:#546e7a;'>/100</span>"
                            f"<span style='font-size:0.82rem;color:{_si_color};font-weight:600;margin-left:4px;'>{_sl}</span>"
                            f"</div>"
                            f"<div style='background:#1e2d45;border-radius:3px;height:4px;margin-top:5px;'>"
                            f"<div style='background:{_si_color};width:{_si}%;height:4px;border-radius:3px;'></div>"
                            f"</div>"
                            f"<div style='font-size:0.70rem;color:#455a64;margin-top:3px;'>"
                            f"{meta.get('compliance_drop',0):.0f}% compliance drop · topology sharpness</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        # Full params table
                        with st.expander("Full parameters", expanded=False):
                            rep = pd.DataFrame({
                                "Parameter": ["Part name","Material","Load","Iterations","Improvement",
                                              "Compliance","Target VF","Achieved VF","Mesh",
                                              "Triangles","Mass","Allowable","SF"],
                                "Value": [
                                    part_name, export_mat.split("(")[0].strip(),
                                    meta["load_scenario"], str(meta["iters"]),
                                    f"{improvement}% stiffer", f"{meta['compliance']:.5f}",
                                    f"{meta['vf']*100:.0f}%", f"{meta['achieved_vf']*100:.1f}%",
                                    f"{meta['nelx']}×{meta['nely']}",
                                    f"{n_faces:,}" if stl_bytes else "—",
                                    f"{topo_mass}g", f"{allowable_stress} MPa", f"{safety_factor}×",
                                ],
                            })
                            st.dataframe(rep, use_container_width=True, hide_index=True)
                        # Share
                        if st.button("Share result", use_container_width=True, key="share_btn",
                                     help="Encodes design into a shareable URL"):
                            try:
                                _xp_gz  = gzip.compress(xPhys.astype(np.float32).tobytes(), compresslevel=6)
                                _s_meta = {k:v for k,v in meta.items()
                                           if isinstance(v,(int,float,str,bool,type(None)))}
                                _s_meta["box"]     = list(meta["box"])
                                _s_meta["history"] = history[-50:]
                                _payload = json.dumps({"xPhys_b64": base64.b64encode(_xp_gz).decode(),
                                                       "xPhys_shape": list(xPhys.shape),
                                                       "meta": _s_meta}).encode()
                                _enc = base64.urlsafe_b64encode(
                                    gzip.compress(_payload, compresslevel=9)
                                ).decode().rstrip("=")
                                st.text_input("Copy this URL:", value=f"?state={_enc}",
                                              key="share_url_display")
                            except Exception as _se:
                                st.error(f"Share failed: {_se}")

                    with _info2:
                        st.markdown("**AI Analysis**")
                        _narration = st.session_state.get("topology_narration")
                        if _narration:
                            st.markdown(
                                f'<div class="chat-bubble" style="font-size:0.83rem;line-height:1.6;">'
                                f'{_narration}</div>', unsafe_allow_html=True,
                            )
                            if st.button("Re-generate", key="regen_narration", use_container_width=True):
                                st.session_state["topology_narration"] = None
                                st.rerun()
                        else:
                            _llm_ready = bool(st.session_state.get("llm_api_key","").strip())
                            if st.button(
                                "Generate Analysis" if _llm_ready else "Configure AI first",
                                key="gen_narration", use_container_width=True,
                                disabled=not _llm_ready,
                            ):
                                _sp = (xPhys>0.5).sum()/xPhys.size*100
                                _vp = (xPhys<0.2).sum()/xPhys.size*100
                                _prompt = (
                                    f"Analyze this topology optimization result for a structural engineer:\n"
                                    f"- Design space: {box_w}×{box_h}×{box_d} mm\n"
                                    f"- Material: {export_mat}\n"
                                    f"- Load scenario: {meta['load_scenario']}\n"
                                    f"- Volume fraction: {meta['achieved_vf']*100:.1f}% (target {meta['vf']*100:.0f}%)\n"
                                    f"- SIMP compliance: {meta['compliance']:.5f} → "
                                    f"{meta['compliance_drop']:.1f}% improvement\n"
                                    f"- Stiffness Index: {meta.get('stiffness_index',0)}/100\n"
                                    f"- Topology: {_sp:.0f}% solid, {_vp:.0f}% void\n\n"
                                    f"Give 3–4 sentences: where material concentrates, load paths, "
                                    f"printability, one suggestion. Plain text."
                                )
                                with st.spinner("Generating analysis…"):
                                    try:
                                        _narr = chat_module.call_llm(
                                            st.session_state["llm_model"],
                                            st.session_state["llm_api_key"],
                                            _prompt,
                                        )
                                        st.session_state["topology_narration"] = _narr
                                        st.rerun()
                                    except Exception as _ne:
                                        st.error(f"LLM error: {_ne}")

            # ── Infill view ────────────────────────────────────
            elif _right_view == "🧩 Infill":
                pi1, pi2 = st.columns([3, 1])
                with pi1:
                    if infill_sf is not None:
                        st.caption(f"{i_pattern} · {st.session_state.get('infill_period',12):.0f}mm period")
                        pcolor = INFILL_PATTERNS[i_pattern]["color"]
                        fig_inf = go.Figure()
                        for t in mesh_traces(infill_sf, box_w, box_h, box_d, 0.0, pcolor, "Infill",
                                             is_xphys=False):
                            fig_inf.add_trace(t)
                        for t in wireframe(box_w, box_h, box_d, 0.2):
                            fig_inf.add_trace(t)
                        fig_inf.update_layout(**scene3d(box_w, box_h, box_d, 480))
                        st.plotly_chart(fig_inf, use_container_width=True)
                    else:
                        st.info("Click **Generate Infill** in the sidebar after running the optimizer.")
                        fig_ph = go.Figure()
                        for t in mesh_traces(xPhys, box_w, box_h, box_d, meta["iso"],
                                             export_color, "Topology"):
                            fig_ph.add_trace(t)
                        for t in wireframe(box_w, box_h, box_d):
                            fig_ph.add_trace(t)
                        fig_ph.update_layout(**scene3d(box_w, box_h, box_d, 480))
                        st.plotly_chart(fig_ph, use_container_width=True)
                with pi2:
                    st.subheader("Pattern stats")
                    if infill_stl:
                        solid_mass_ref = round(
                            (box_w/10)*(box_h/10)*(box_d/10)*meta["vf"]*em_props["rho_gcc"], 1
                        )
                        saving = round((1-infill_mass/solid_mass_ref)*100,1) if solid_mass_ref>0 else 0
                        mi1, mi2 = st.columns(2)
                        with mi1: st.metric("Triangles", f"{infill_faces:,}")
                        with mi2: st.metric("File size",  f"{round(len(infill_stl)/1024,1)} KB")
                        ma1, ma2 = st.columns(2)
                        with ma1: st.metric("Infill mass", f"{infill_mass} g")
                        with ma2: st.metric("Mass saving", f"{saving}%", delta="lighter")
                        st.divider()
                        fname_inf = (f"infill_{i_pattern.lower().replace('-','_')}_"
                                     f"{part_name.replace(' ','_')}_{box_w}x{box_h}x{box_d}.stl")
                        st.download_button("Download Infill STL", data=infill_stl,
                                           file_name=fname_inf, mime="application/octet-stream",
                                           use_container_width=True, type="primary")
                        st.caption(f"`{fname_inf}`")
                        mid_z = infill_sf.shape[2]//2
                        fig_sl = go.Figure(go.Heatmap(
                            z=infill_sf[:,:,mid_z].T,
                            colorscale=[[0,INFILL_PATTERNS[i_pattern]["color"]],
                                        [0.5,"#1a1a2e"],[1,"#050508"]],
                            showscale=False, zmin=-2, zmax=2,
                        ))
                        fig_sl.update_layout(
                            xaxis=dict(title="X",scaleanchor="y"), yaxis=dict(title="Y"),
                            height=200, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=30,r=10,t=10,b=30), font=dict(color="white"),
                        )
                        st.caption("Cross-section slice (Z midpoint)")
                        st.plotly_chart(fig_sl, use_container_width=True)

            # ── Compare view ───────────────────────────────────
            elif _right_view == "📊 Compare" and result_mode == "Material Comparison":
                st.subheader("Material Comparison")
                if winners:
                    wc1, wc2, wc3, wc4 = st.columns(4)
                    for col, (label, key, desc) in zip(
                        [wc1,wc2,wc3,wc4],
                        [("Stiffest","stiffest","Lowest compliance"),
                         ("Lightest","lightest","Lowest mass"),
                         ("Best Ashby","best_ashby","E^1/3 / rho"),
                         ("Cheapest","cheapest","Lowest cost")],
                    ):
                        wn = winners.get(key,"—")
                        short  = wn.split("(")[0].strip() if wn!="—" else "—"
                        wcolor = MATERIALS[wn]["color"] if wn!="—" else "#888"
                        with col:
                            st.markdown(f"**{label}**")
                            st.markdown(f"<div style='color:{wcolor};font-size:18px;font-weight:700'>{short}</div>",
                                        unsafe_allow_html=True)
                            st.caption(desc)
                    st.divider()
                col_tbl, col_rad = st.columns([1.3, 1])
                with col_tbl:
                    rows = [{
                        "Material": n.split("(")[0].strip(),
                        "E (GPa)": MATERIALS[n]["E_gpa"],
                        "rho": MATERIALS[n]["rho_gcc"],
                        "Mass (g)": mat_scores[n]["mass_g"],
                        "Compliance": mat_scores[n]["comp"],
                        "Ashby": mat_scores[n]["ashby"],
                        "Allow. MPa": mat_scores[n]["allowable"],
                        "OK": "Yes" if mat_scores[n]["stress_ok"] else "No",
                    } for n in compare_mats_saved]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True, height=300)
                with col_rad:
                    all_E2=[MATERIALS[n]["E_gpa"] for n in compare_mats_saved]
                    all_rho2=[MATERIALS[n]["rho_gcc"] for n in compare_mats_saved]
                    all_nu2=[MATERIALS[n]["nu"] for n in compare_mats_saved]
                    all_ys2=[MATERIALS[n]["yield_strength_mpa"] for n in compare_mats_saved]
                    fig_r = go.Figure()
                    for name in compare_mats_saved:
                        m2=MATERIALS[name]
                        rv=[norm_v(m2["E_gpa"],all_E2),1-norm_v(m2["rho_gcc"],all_rho2),
                            norm_v(m2["nu"],all_nu2),norm_v(m2["yield_strength_mpa"],all_ys2),
                            1.0 if m2["printable"] else 0.0]
                        rv2=rv+[rv[0]]
                        rc2=["Stiffness","Lightness","Compressibility","Strength","Printable","Stiffness"]
                        fig_r.add_trace(go.Scatterpolar(
                            r=rv2,theta=rc2,fill="toself",
                            fillcolor=rgba(m2["color"],0.18),
                            line=dict(color=m2["color"],width=2),
                            name=name.split("(")[0].strip(),
                        ))
                    fig_r.update_layout(
                        polar=dict(radialaxis=dict(visible=True,range=[0,1],showticklabels=False)),
                        showlegend=True, height=300,
                        margin=dict(l=40,r=40,t=30,b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(bgcolor="rgba(0,0,0,0.4)",font=dict(color="white",size=11)),
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

            # ── Density Map view ───────────────────────────────
            elif _right_view == "🗺️ Density Map":
                st.caption("Bright = solid · dark = void · increase SIMP Penalty for sharper result")
                fig_h = go.Figure(go.Heatmap(
                    z=1-xPhys,
                    colorscale=[[0,export_color],[0.5,"#1a1a2e"],[1,"#050508"]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Solid/Void",font=dict(color="white")),
                        tickvals=[0,0.5,1],ticktext=["Solid","0.5","Void"],
                        tickfont=dict(color="white"),
                    ),
                    hovertemplate="X=%{x} Y=%{y}<br>Density=%{customdata:.3f}<extra></extra>",
                    customdata=xPhys,
                ))
                fig_h.update_layout(
                    xaxis=dict(title="Element X",scaleanchor="y"),
                    yaxis=dict(title="Element Y",autorange="reversed"),
                    height=max(280,xPhys.shape[0]*18),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40,r=40,t=10,b=40),font=dict(color="white"),
                )
                st.plotly_chart(fig_h, use_container_width=True)
                hv,hb=np.histogram(xPhys.flatten(),bins=25,range=(0,1))
                fig_hist=go.Figure(go.Bar(
                    x=[(hb[i]+hb[i+1])/2 for i in range(len(hb)-1)],y=hv,
                    marker_color=export_color,
                    marker_line_color="rgba(255,255,255,0.2)",marker_line_width=1,
                ))
                fig_hist.update_layout(
                    xaxis=dict(title="Density",range=[0,1]),yaxis=dict(title="Elements"),
                    height=200,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0e1117",
                    xaxis_gridcolor="#222",yaxis_gridcolor="#222",
                    margin=dict(l=50,r=20,t=10,b=45),font=dict(color="white"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                sp=(xPhys>0.5).sum()/xPhys.size*100
                vp=(xPhys<0.2).sum()/xPhys.size*100
                bc1,bc2,bc3=st.columns(3)
                with bc1: st.metric("Solid >0.5",   f"{sp:.0f}%")
                with bc2: st.metric("Grey 0.2–0.5", f"{100-sp-vp:.0f}%")
                with bc3: st.metric("Void <0.2",    f"{vp:.0f}%")

            # ── Convergence view ───────────────────────────────
            elif _right_view == "📈 Convergence":
                fig_c=go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=list(range(1,len(history)+1)),y=history,
                    mode="lines",line=dict(color=export_color,width=2),
                    name=meta["load_scenario"],
                    hovertemplate="Iter %{x}<br>%{y:.5f}<extra></extra>",
                ))
                if history_lc2:
                    fig_c.add_trace(go.Scatter(
                        x=list(range(1,len(history_lc2)+1)),y=history_lc2,
                        mode="lines",line=dict(color="rgba(180,180,180,0.5)",width=1.5,dash="dash"),
                        name="Single-load ref",
                    ))
                if len(history)>5:
                    fig_c.add_hline(y=np.mean(history[-5:]),
                                    line=dict(color="orange",width=1,dash="dot"),
                                    annotation_text="Avg (last 5)",
                                    annotation_font_color="orange")
                fig_c.update_layout(
                    xaxis=dict(title="Iteration"),yaxis=dict(title="Compliance"),
                    height=360,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0e1117",
                    xaxis_gridcolor="#222",yaxis_gridcolor="#222",
                    margin=dict(l=60,r=20,t=20,b=50),font=dict(color="white"),
                    legend=dict(bgcolor="rgba(0,0,0,0.4)",font=dict(color="white")),
                )
                st.plotly_chart(fig_c, use_container_width=True)
                if len(history)>=2:
                    ca,cb,cc,cd,ce=st.columns(5)
                    with ca: st.metric("Initial",      f"{history[0]:.3f}")
                    with cb: st.metric("Final",         f"{history[-1]:.3f}")
                    with cc: st.metric("Improvement",   f"{improvement}%", delta="stiffer")
                    with cd: st.metric("Iterations",    str(meta["iters"]))
                    _ce_si = meta.get('stiffness_index', 0)
                    with ce: st.metric("Stiffness Index",
                                       f"{_ce_si}/100" if _ce_si > 0 else "—",
                                       delta=meta.get("stiffness_label","—") if _ce_si > 0 else None)

                # ── Convergence quality panel ──────────────────
                st.markdown("**Convergence Quality**")
                _xp_cur = st.session_state.get("xPhys", xPhys)
                _solid_pct = float((_xp_cur > 0.9).sum()) / _xp_cur.size * 100
                _void_pct  = float((_xp_cur < 0.1).sum()) / _xp_cur.size * 100
                _bim_idx   = _solid_pct + _void_pct
                _grey_pct  = 100.0 - _bim_idx
                _delta_x   = abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-12) * 100 if len(history) >= 2 else None
                _conv_ok   = _bim_idx >= 75
                _bim_color = "#4CAF50" if _conv_ok else "#FF9800" if _bim_idx >= 55 else "#f44336"
                _bim_label = "Well converged" if _conv_ok else "Partially converged" if _bim_idx >= 55 else "Poor convergence"
                st.markdown(f"""
<div style="background:#141820;border-radius:6px;padding:12px 16px;margin:8px 0;display:flex;gap:24px;flex-wrap:wrap;">
  <div><span style="font-size:0.7rem;color:#546e7a;text-transform:uppercase;">Bimodality Index</span><br>
       <span style="font-size:1.3rem;font-weight:700;color:{_bim_color};">{_bim_idx:.1f}%</span>
       <span style="font-size:0.75rem;color:#546e7a;"> ({_bim_label})</span></div>
  <div><span style="font-size:0.7rem;color:#546e7a;text-transform:uppercase;">Solid (&gt;90%)</span><br>
       <span style="font-size:1.1rem;font-weight:600;color:#cfd8dc;">{_solid_pct:.1f}%</span></div>
  <div><span style="font-size:0.7rem;color:#546e7a;text-transform:uppercase;">Void (&lt;10%)</span><br>
       <span style="font-size:1.1rem;font-weight:600;color:#cfd8dc;">{_void_pct:.1f}%</span></div>
  <div><span style="font-size:0.7rem;color:#546e7a;text-transform:uppercase;">Grey zone</span><br>
       <span style="font-size:1.1rem;font-weight:600;color:#cfd8dc;">{_grey_pct:.1f}%</span></div>
  {f'<div><span style="font-size:0.7rem;color:#546e7a;text-transform:uppercase;">Last Δ compliance</span><br><span style="font-size:1.1rem;font-weight:600;color:#cfd8dc;">{_delta_x:.3f}%</span></div>' if _delta_x is not None else ''}
</div>
""", unsafe_allow_html=True)
                if _bim_idx < 55:
                    st.info("Low bimodality — try increasing SIMP penalty (p) or running more iterations.")

                # ── MBB Benchmark ──────────────────────────────
                with st.expander("MBB Benchmark (Sigmund 2001)", expanded=False):
                    st.caption(
                        "Classic test: simply-supported beam loaded at midspan. "
                        "Expected compliance ≈ 187.96 for nelx=60, nely=20, volfrac=0.5, p=3, rmin=1.5."
                    )
                    if st.button("Run MBB Benchmark", key="mbb_run_btn"):
                        with st.spinner("Running MBB benchmark (60×20, 100 iters)…"):
                            _mbb_nelx, _mbb_nely = 60, 20
                            _mbb_ndof = 2 * (_mbb_nelx + 1) * (_mbb_nely + 1)
                            _mbb_F = np.zeros(_mbb_ndof)
                            # Midspan top-left node, downward
                            _mbb_mid_node = (_mbb_nely + 1) * (_mbb_nelx // 2) + _mbb_nely
                            _mbb_F[2 * _mbb_mid_node + 1] = -1.0
                            # Supports: left column (X + Y fixed), right-bottom roller (Y fixed)
                            _mbb_fixed = set()
                            for _iy in range(_mbb_nely + 1):
                                _mbb_fixed.add(2 * _iy)
                                _mbb_fixed.add(2 * _iy + 1)
                            _mbb_fixed.add(2 * ((_mbb_nely + 1) * _mbb_nelx + _mbb_nely) + 1)
                            _mbb_H, _mbb_Hs = build_filter_fast(_mbb_nelx, _mbb_nely, 1.5)
                            _mbb_lc = [(_mbb_F, 1.0)]
                            _mbb_fd = np.array(sorted(_mbb_fixed))
                            _mbb_x, _mbb_hist, _ = simp_core(
                                _mbb_nelx, _mbb_nely, 0.5, 0.3, 3.0,
                                _mbb_H, _mbb_Hs, 100, _mbb_lc, _mbb_fd,
                            )
                            _mbb_c = _mbb_hist[-1]
                            _mbb_err = abs(_mbb_c - 187.96) / 187.96 * 100
                        _mbb_pass = _mbb_err < 5.0
                        _mbb_col = "#4CAF50" if _mbb_pass else "#FF9800"
                        st.markdown(f"""
<div style="background:#141820;border-radius:6px;padding:12px 16px;margin:6px 0;">
  <strong>MBB Result:</strong> C = <span style="color:{_mbb_col};font-weight:700;">{_mbb_c:.2f}</span>
  &nbsp;|&nbsp; Expected: 187.96
  &nbsp;|&nbsp; Error: <span style="color:{_mbb_col};">{_mbb_err:.2f}%</span>
  &nbsp;|&nbsp; {'✓ PASS' if _mbb_pass else '✗ FAIL — check solver or filter settings'}
</div>""", unsafe_allow_html=True)

            # ── Pareto Sweep view ──────────────────────────────
            elif _right_view == "📊 Pareto":
                st.caption("Runs the optimizer over a range of volume fractions to map the mass–stiffness Pareto frontier.")
                _vf_range = [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65]
                _pareto_iters = st.slider("Iterations per point", 10, 60, 30, 5, key="pareto_iters")
                if st.button("Run Pareto Sweep", type="primary", use_container_width=True, key="pareto_run_btn"):
                    _pareto_ph   = st.empty()
                    _pareto_prog = st.progress(0, text="Pareto sweep…")
                    _p_mean_nu   = float(np.mean([MATERIALS[n]["nu"] for n in compare_mats]))
                    _p_use_3d    = st.session_state.get("use_3d_simp", False)
                    _p_nelz      = int(st.session_state.get("_preset_nelz", 8))
                    _pareto_pts  = []
                    for _pi, _vf in enumerate(_vf_range):
                        _pareto_prog.progress(int(_pi/len(_vf_range)*100),
                                              text=f"VF={_vf:.0%} ({_pi+1}/{len(_vf_range)})…")
                        try:
                            if _p_use_3d:
                                _pH,_pHs = build_filter_3d(nelx,nely,_p_nelz,rmin)
                                _p_lc,_p_fd = build_load_cases_3d(
                                    st.session_state["load_specs"],st.session_state["support_specs"],
                                    nelx,nely,_p_nelz)
                                _px,_ph,_psf=simp_core_3d(nelx,nely,_p_nelz,_vf,_p_mean_nu,penal,
                                                           _pH,_pHs,_pareto_iters,_p_lc,_p_fd)
                            else:
                                _pH,_pHs = build_filter_fast(nelx,nely,rmin)
                                _p_lc,_p_fd,_ = build_load_cases(
                                    st.session_state["load_specs"],st.session_state["support_specs"],
                                    nelx,nely)
                                _px,_ph,_psf=simp_core(nelx,nely,_vf,_p_mean_nu,penal,
                                                        _pH,_pHs,_pareto_iters,_p_lc,_p_fd)
                            _em=st.session_state.get("export_mat",sel_mat)
                            _mass=round((box_w/10)*(box_h/10)*(box_d/10)*float(_px.mean())*MATERIALS[_em]["rho_gcc"],1)
                            _pareto_pts.append({"vf":_vf,"compliance":round(_ph[-1],5),
                                                "mass_g":_mass,"iters":len(_ph)})
                        except Exception as _pe:
                            st.warning(f"VF={_vf:.0%} failed: {_pe}")
                    _pareto_prog.progress(100, text="Pareto sweep complete")
                    st.session_state["pareto_results"] = _pareto_pts
                    st.rerun()

                _par = st.session_state.get("pareto_results")
                if _par:
                    _df_par = pd.DataFrame(_par).sort_values("mass_g")
                    fig_par = go.Figure()
                    fig_par.add_trace(go.Scatter(
                        x=_df_par["mass_g"], y=_df_par["compliance"],
                        mode="lines+markers",
                        line=dict(color="#1565c0",width=2),
                        marker=dict(size=10, color=list(_df_par["vf"]),
                                    colorscale="Viridis", showscale=True,
                                    colorbar=dict(title=dict(text="Volume Fraction",
                                                             font=dict(color="white")),
                                                  tickfont=dict(color="white"))),
                        text=[f"VF={r['vf']:.0%}<br>C={r['compliance']:.4f}<br>{r['mass_g']}g"
                              for _,r in _df_par.iterrows()],
                        hoverinfo="text", name="Pareto front",
                    ))
                    _cur_meta = st.session_state.get("opt_meta",{})
                    _cur_mass = round((box_w/10)*(box_h/10)*(box_d/10)*_cur_meta.get("achieved_vf",0)*em_props["rho_gcc"],1)
                    fig_par.add_trace(go.Scatter(
                        x=[_cur_mass],y=[_cur_meta.get("compliance",None)],
                        mode="markers",
                        marker=dict(size=16,color="#4CAF50",symbol="star"),
                        name="Current design",
                    ))
                    fig_par.add_annotation(x=_cur_mass,y=_cur_meta.get("compliance",0),
                                           text="Current",showarrow=True,arrowhead=2,
                                           font=dict(color="#4CAF50"),arrowcolor="#4CAF50",
                                           ax=30,ay=-30)
                    fig_par.update_layout(
                        xaxis=dict(title="Mass (g)",gridcolor="#222"),
                        yaxis=dict(title="Compliance (lower=stiffer)",gridcolor="#222"),
                        height=420,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0e1117",
                        margin=dict(l=60,r=20,t=20,b=50),font=dict(color="white"),
                        legend=dict(bgcolor="rgba(0,0,0,0.4)",font=dict(color="white")),
                    )
                    st.plotly_chart(fig_par, use_container_width=True)
                    st.caption("Lower-left = lighter AND stiffer. Green star = current design.")
                    st.dataframe(_df_par[["vf","mass_g","compliance","iters"]].rename(columns={
                        "vf":"Volume Fraction","mass_g":"Mass (g)","compliance":"Compliance","iters":"Iterations"}),
                        use_container_width=True, hide_index=True)
                    if st.button("Clear Pareto results", key="pareto_clear"):
                        st.session_state["pareto_results"] = None
                        st.rerun()
                else:
                    st.info("Click **Run Pareto Sweep** to generate the mass–stiffness tradeoff curve.")

            # ── Export view ────────────────────────────────────
            elif _right_view == "💾 Export":
                col_topo, col_inf = st.columns(2)
                with col_topo:
                    st.markdown("**Topology STL** — solid shell from SIMP")
                    if stl_bytes:
                        fname_t = (f"topo_{part_name.replace(' ','_')}_"
                                   f"{export_mat.split()[0].lower()}_{box_w}x{box_h}x{box_d}.stl")
                        st.download_button(
                            label=f"Download Topology STL ({round(len(stl_bytes)/1024,1)} KB · {n_faces:,} tri)",
                            data=stl_bytes, file_name=fname_t, mime="application/octet-stream",
                            use_container_width=True, type="primary",
                        )
                        st.caption(f"`{fname_t}`")
                        st.caption("Use in slicer with 15% gyroid infill + 3-4 walls.")
                    else:
                        st.error("No topology STL — lower iso-level and re-run.")
                with col_inf:
                    st.markdown("**Infill STL** — TPMS patterned interior")
                    if infill_stl:
                        fname_i = (f"infill_{i_pattern.lower().replace('-','_')}_"
                                   f"{part_name.replace(' ','_')}_{box_w}x{box_h}x{box_d}.stl")
                        st.download_button(
                            label=f"Download Infill STL ({round(len(infill_stl)/1024,1)} KB · {infill_faces:,} tri)",
                            data=infill_stl, file_name=fname_i, mime="application/octet-stream",
                            use_container_width=True, type="primary",
                        )
                        st.caption(f"`{fname_i}`")
                        st.caption("Print at 100% infill in slicer.")
                    else:
                        st.info("Generate infill first.")
                st.divider()
                st.markdown(f"**Slicer settings · {export_mat.split('(')[0].strip()}**")
                m_g = MATERIALS[export_mat]
                st.markdown(f"""
| Setting | Topology STL | Infill STL |
|---|---|---|
| Infill % | 15% gyroid | **100%** |
| Walls | 3–4 | 2–3 |
| Layer height | 0.2 mm | {layer_h} mm |
| Print temp | {m_g['print_temp']} | {m_g['print_temp']} |
| Bed temp | {m_g['bed_temp']} | {m_g['bed_temp']} |
""")
                st.divider()
                st.markdown("### Raw Data Export")
                _npy_col1, _npy_col2 = st.columns(2)
                with _npy_col1:
                    st.markdown("**Density field (xPhys)** — NumPy array `(nely, nelx)`")
                    _xp_arr = st.session_state.get("xPhys")
                    if _xp_arr is not None:
                        import io as _io
                        _npy_buf = _io.BytesIO()
                        np.save(_npy_buf, _xp_arr)
                        _npy_bytes = _npy_buf.getvalue()
                        _npy_fname = (f"xphys_{part_name.replace(' ','_')}_"
                                      f"{_xp_arr.shape[1]}x{_xp_arr.shape[0]}.npy")
                        st.download_button(
                            label=f"Download xPhys .npy ({_xp_arr.shape[1]}×{_xp_arr.shape[0]})",
                            data=_npy_bytes, file_name=_npy_fname,
                            mime="application/octet-stream",
                            use_container_width=True, key="dl_xphys_npy",
                        )
                        st.caption("Load with `x = np.load('file.npy')` — values in [0, 1], 1 = solid.")
                    else:
                        st.info("Run optimizer to generate density field.")
                with _npy_col2:
                    st.markdown("**Von Mises stress field** — NumPy array `(nely, nelx)` (normalized)")
                    _sf_arr = st.session_state.get("stress_field")
                    if _sf_arr is not None:
                        import io as _io2
                        _sf_buf = _io2.BytesIO()
                        np.save(_sf_buf, _sf_arr)
                        _sf_bytes = _sf_buf.getvalue()
                        _sf_fname = (f"stress_vm_{part_name.replace(' ','_')}_"
                                     f"{_sf_arr.shape[1]}x{_sf_arr.shape[0]}.npy")
                        st.download_button(
                            label=f"Download stress .npy ({_sf_arr.shape[1]}×{_sf_arr.shape[0]})",
                            data=_sf_bytes, file_name=_sf_fname,
                            mime="application/octet-stream",
                            use_container_width=True, key="dl_stress_npy",
                        )
                        st.caption("Normalized (E=1 reference). Multiply by E₀ for physical units.")
                    else:
                        st.info("Run optimizer to generate stress field.")
                st.divider()
                st.markdown("### FEM Mesh Export")
                st.caption(
                    "Generate a quality tet mesh ready for ICEM CFD, HyperMesh, Fluent, Nastran, or Abaqus. "
                    "Named boundary groups FIXED / LOAD / FREE are pre-tagged."
                )
                if not _HAS_GMSH:
                    st.warning("gmsh not installed — FEM mesh export unavailable. Run `pip install gmsh` locally.")
                with st.expander("Mesh settings", expanded=False):
                    msc1, msc2, msc3 = st.columns(3)
                    with msc1:
                        m_global = st.number_input("Target size (mm)", value=5.0,
                                                   min_value=0.5, max_value=50.0, step=0.5, key="fem_global_size")
                    with msc2:
                        m_min = st.number_input("Min size (mm)", value=1.0,
                                                min_value=0.1, max_value=10.0, step=0.1, key="fem_min_size")
                    with msc3:
                        m_max = st.number_input("Max size (mm)", value=15.0,
                                                min_value=1.0, max_value=100.0, step=1.0, key="fem_max_size")
                    do_smooth = st.checkbox("Apply Laplacian smoothing (3 iterations)", value=True, key="fem_smooth")
                    do_refine = st.checkbox("Stress-guided refinement", value=True, key="fem_refine_stress")

                if stl_bytes:
                    _xp = st.session_state.get("xPhys", np.zeros((1,1)))
                    _solid_pct = float((_xp>0.5).sum())/_xp.size*100
                    _void_pct  = float((_xp<0.2).sum())/_xp.size*100
                    _bimod     = _solid_pct + _void_pct
                    _mesh_ok   = _bimod >= 75.0
                    if not _mesh_ok:
                        st.warning(
                            f"**Topology too fuzzy for meshing** — {100-_bimod:.0f}% grey-zone. "
                            f"Re-run with SIMP Penalty ≥ 4.0 and Max Iterations ≥ 80 "
                            f"(current bimodality {_bimod:.0f}%, need ≥ 75%)."
                        )
                    else:
                        st.success(f"Topology quality OK — {_bimod:.0f}% cleanly solid/void.")
                    if st.button("Generate FEM Mesh", type="primary", use_container_width=True,
                                 key="fem_generate_btn", disabled=(not _mesh_ok or not _HAS_GMSH)):
                        with st.spinner("Running Gmsh…"):
                            try:
                                from concurrent.futures import ProcessPoolExecutor
                                with ProcessPoolExecutor(max_workers=1) as _pool:
                                    fem_result = _pool.submit(
                                        generate_fem_mesh, stl_bytes, box_w, box_h, box_d,
                                        fixed_face, load_face, stress_field,
                                        m_global, m_min, m_max, do_smooth, do_refine,
                                    ).result()
                                st.session_state["fem_mesh"] = fem_result
                                st.rerun()
                            except ImportError as ie: st.error(str(ie))
                            except ValueError  as ve: st.error(f"Mesh failed: {ve}")
                            except Exception  as exc: st.error(f"Mesh failed: {exc}")
                else:
                    st.warning("Run the optimizer first to generate the topology surface.")

                fem_mesh = st.session_state.get("fem_mesh")
                if fem_mesh:
                    ne=fem_mesh["n_elements"]; nn=fem_mesh["n_nodes"]
                    ft=fem_mesh["fixed_tris"]; lt=fem_mesh["load_tris"]; frt=fem_mesh["free_tris"]
                    st.success(f"✓ Mesh ready — **{nn:,} nodes · {ne:,} tet elements**")
                    gc1,gc2,gc3=st.columns(3)
                    with gc1: st.metric("FIXED group", f"{ft:,} tri" if ft else "—")
                    with gc2: st.metric("LOAD group",  f"{lt:,} tri" if lt else "—")
                    with gc3: st.metric("FREE group",  f"{frt:,} tri")
                    with st.expander("🔍 Mesh Preview", expanded=True):
                        _prev_l, _prev_r = st.columns([3, 1])
                        with _prev_r:
                            st.markdown("**Render mode**")
                            _render_mode = st.radio(
                                "render_mode",
                                options=["solid_edges","solid","wireframe","quality"],
                                format_func={"solid_edges":"⬛ Solid + Edges","solid":"🟦 Solid only",
                                             "wireframe":"🔲 Wireframe","quality":"🌈 Element Quality"}.get,
                                index=0, key="fem_render_mode", label_visibility="collapsed",
                            )
                            st.divider()
                            st.markdown("**Legend**")
                            st.markdown(
                                "<div style='font-size:0.82rem;line-height:1.8;'>"
                                "<span style='color:#4d90ff;'>■</span> Fixed (support)<br>"
                                "<span style='color:#ff5252;'>■</span> Load (force)<br>"
                                "<span style='color:#607d8b;'>■</span> Free surface</div>",
                                unsafe_allow_html=True,
                            )
                            if fem_mesh.get("msh_bytes"):
                                try:
                                    _qs=fem_quality_stats(fem_mesh["msh_bytes"])
                                    _all_tris=sum(v["n_tris"] for v in _qs.values())
                                    _all_good=sum(v["n_tris"]*v["pct_good"]/100 for v in _qs.values())
                                    _pct_good=(_all_good/_all_tris*100 if _all_tris else 0)
                                    _grade=("🟢 Excellent" if _pct_good>=90 else "🟡 Good" if _pct_good>=75
                                            else "🟠 Fair" if _pct_good>=55 else "🔴 Poor")
                                    st.markdown("**Mesh quality**")
                                    st.markdown(f"<div style='font-size:0.82rem;line-height:1.9;'>"
                                                f"<b>{_grade}</b><br>AR &lt; 2: {_pct_good:.0f}% of tris<br>"
                                                f"Triangles: {_all_tris:,}</div>", unsafe_allow_html=True)
                                except Exception: pass
                        with _prev_l:
                            _msh_traces = fem_surface_traces(fem_mesh["msh_bytes"], render_mode=_render_mode)
                            if _msh_traces:
                                _fig_mesh = go.Figure()
                                for _t in _msh_traces: _fig_mesh.add_trace(_t)
                                for _t in wireframe(box_w, box_h, box_d, alpha=0.12): _fig_mesh.add_trace(_t)
                                _mesh_layout = scene3d(box_w, box_h, box_d, 500)
                                for _ax in ["xaxis","yaxis","zaxis"]:
                                    _mesh_layout["scene"][_ax]["backgroundcolor"] = "rgba(8,12,20,0)"
                                _mesh_layout["scene"]["bgcolor"]  = "rgba(8,12,20,1)"
                                _mesh_layout["paper_bgcolor"]     = "rgba(8,12,20,1)"
                                _mesh_layout["plot_bgcolor"]      = "rgba(8,12,20,1)"
                                _fig_mesh.update_layout(**_mesh_layout)
                                st.plotly_chart(_fig_mesh, use_container_width=True, key="fem_preview_chart")
                            else:
                                st.info("Could not parse mesh for preview.")
                    base=part_name.replace(" ","_"); tag=f"{box_w}x{box_h}x{box_d}"
                    dc1,dc2,dc3=st.columns(3)
                    with dc1:
                        st.download_button("⬇ .msh  (Fluent / ICEM)", data=fem_mesh["msh_bytes"],
                                           file_name=f"{base}_{tag}.msh", mime="application/octet-stream",
                                           use_container_width=True, type="primary", key="fem_dl_msh")
                        st.caption("Fluent · ICEM CFD · OpenFOAM")
                    with dc2:
                        st.download_button("⬇ .bdf  (Nastran)", data=fem_mesh["bdf_bytes"],
                                           file_name=f"{base}_{tag}.bdf", mime="application/octet-stream",
                                           use_container_width=True, key="fem_dl_bdf")
                        st.caption("Nastran · HyperMesh · Patran")
                    with dc3:
                        st.download_button("⬇ .inp  (Abaqus)", data=fem_mesh["inp_bytes"],
                                           file_name=f"{base}_{tag}.inp", mime="application/octet-stream",
                                           use_container_width=True, key="fem_dl_inp")
                        st.caption("Abaqus · CalculiX")

            # ── Print Estimate view ────────────────────────────
            elif _right_view == "🖨️ Print Est.":
                st.caption(f"Layer height {layer_h}mm · speed {speed_ms}mm/s · "
                           f"{export_mat.split('(')[0].strip()} @ ${em_props['cost_per_kg']:.0f}/kg")
                pe1,pe2,pe3,pe4=st.columns(4)
                with pe1: st.metric("Print time",    f"{print_est['time_min']:.0f} min",
                                    delta=f"{print_est['time_h']:.1f} h")
                with pe2: st.metric("Filament",      f"{print_est['filament_m']:.1f} m")
                with pe3: st.metric("Mass",          f"{active_mass} g")
                with pe4: st.metric("Material cost", f"${print_est['material_cost_usd']:.2f}")
                st.divider()
                st.markdown("**Scenario Comparison**")
                scenarios=[("Topology STL (standard)",topo_mass,em_props["rho_gcc"])]
                if infill_stl: scenarios.append(("With infill (patterned)",infill_mass,em_props["rho_gcc"]))
                vf_rows=[]
                for sc_name,sc_mass,sc_rho in scenarios:
                    est=estimate_print(sc_mass,sc_rho,layer_h,speed_ms,cost_per_kg=em_props["cost_per_kg"])
                    vf_rows.append({"Scenario":sc_name,"Mass (g)":sc_mass,
                                    "Time (min)":est["time_min"],"Filament (m)":est["filament_m"],
                                    "Cost ($)":est["material_cost_usd"]})
                st.dataframe(pd.DataFrame(vf_rows), use_container_width=True, hide_index=True)
                st.divider()
                st.markdown("**Speed vs Quality**")
                speeds=[20,30,40,50,60,80,100,120,150]
                rows=[]
                for sp in speeds:
                    est=estimate_print(active_mass,em_props["rho_gcc"],layer_h,sp,
                                       cost_per_kg=em_props["cost_per_kg"])
                    rows.append({"Speed (mm/s)":sp,"Time (min)":est["time_min"]})
                fig_spd=go.Figure(go.Scatter(
                    x=speeds,y=[r["Time (min)"] for r in rows],
                    mode="lines+markers",line=dict(color=export_color,width=2),
                    marker=dict(size=6,color=export_color),
                    hovertemplate="Speed %{x}mm/s<br>%{y:.0f}min<extra></extra>",
                ))
                fig_spd.add_vline(x=speed_ms,line=dict(color="orange",width=2,dash="dot"),
                                  annotation_text=f"Current: {speed_ms}mm/s",annotation_font_color="orange")
                fig_spd.update_layout(xaxis=dict(title="Print speed (mm/s)"),yaxis=dict(title="Time (min)"),
                                      height=260,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0e1117",
                                      xaxis_gridcolor="#222",yaxis_gridcolor="#222",
                                      margin=dict(l=60,r=20,t=20,b=50),font=dict(color="white"))
                st.plotly_chart(fig_spd, use_container_width=True)
                st.divider()
                st.markdown("**All Material Print Costs**")
                mat_rows=[]
                for mname,mprops in MATERIALS.items():
                    if not mprops["printable"]: continue
                    m_mass=round((box_w/10)*(box_h/10)*(box_d/10)*meta["achieved_vf"]*mprops["rho_gcc"],1)
                    est2=estimate_print(m_mass,mprops["rho_gcc"],layer_h,speed_ms,cost_per_kg=mprops["cost_per_kg"])
                    mat_rows.append({"Material":mname.split("(")[0].strip(),"Mass (g)":m_mass,
                                     "Time (min)":est2["time_min"],"Cost ($)":est2["material_cost_usd"],
                                     "$/kg":mprops["cost_per_kg"]})
                st.dataframe(pd.DataFrame(mat_rows).sort_values("Cost ($)"),
                             use_container_width=True, hide_index=True)

            # ── History view ───────────────────────────────────
            elif _right_view == "📋 History":
                st.caption("Automatically saved each run. Cleared when browser tab closes.")
                design_history=st.session_state.get("design_history",[])
                if not design_history:
                    st.info("No designs yet. Run the optimizer to save a record here.")
                else:
                    h_rows=[{"ID":r["id"],"Name":r["name"],
                              "Material":r["material"].split("(")[0].strip(),
                              "Box":r["box"],"VF%":f"{r['vf']*100:.0f}%",
                              "Load":r["load"],"Compliance":r["compliance"],
                              "Iters":r["iters"],"Mass(g)":r["mass_g"],
                              "Triangles":r["faces"],"Infill":r["infill"],
                              "Time":r["timestamp"]} for r in design_history]
                    st.dataframe(pd.DataFrame(h_rows), use_container_width=True,
                                 hide_index=True, height=min(400,len(h_rows)*50+60))
                    st.divider()
                    if len(design_history)>=2:
                        names_h=[r["name"] for r in reversed(design_history)]
                        comps_h=[r["compliance"] for r in reversed(design_history)]
                        bar_colors=[next((v["color"] for k,v in MATERIALS.items()
                                          if k==r["material"] or k.startswith(r["material"])),"#888")
                                    for r in reversed(design_history)]
                        fig_evo=go.Figure(go.Bar(x=names_h,y=comps_h,
                                                  marker_color=bar_colors,
                                                  marker_line_color="rgba(255,255,255,0.2)",
                                                  marker_line_width=1))
                        fig_evo.update_layout(
                            xaxis=dict(title="Design"),yaxis=dict(title="Compliance (lower=stiffer)"),
                            height=280,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0e1117",
                            xaxis_gridcolor="#222",yaxis_gridcolor="#222",
                            margin=dict(l=60,r=20,t=20,b=80),font=dict(color="white"))
                        st.plotly_chart(fig_evo, use_container_width=True)
                        best=min(design_history,key=lambda r:r["compliance"])
                        st.success(f"Best design: **{best['name']}** · compliance {best['compliance']:.3f} · "
                                   f"{best['material'].split('(')[0].strip()}")
                    else:
                        st.info("Run more designs to see a comparison chart.")
                    if st.button("Clear history"):
                        st.session_state["design_history"]=[]
                        st.rerun()

            # ── Help view ──────────────────────────────────────
            elif _right_view == "❓ Help":
                st.markdown("""
**Quick start (5 minutes)**

1. Describe your part in the left panel (or switch to Expert Mode for manual controls)
2. Review the extracted parameters and click **Run Optimization**
3. See results in the right panel — drag to rotate the 3D mesh
4. Switch to **Stress** color mode to see load distribution
5. Click **Generate Infill** in the sidebar, then download from **Export**

---

**Chat mode vs Expert Mode**

| Mode | Best for |
|---|---|
| Chat (default) | Non-coders, quick exploration, first-time users |
| Expert Mode | Fine-tuning optimizer parameters, material comparison |

Click **⚙ Switch to Expert Mode** in the sidebar to toggle.

---

**Reading the stress visualization**

Switch Color Mode to **Stress (strain energy)**:
- **Red** = high strain energy = material is doing most work here
- **Blue** = low strain energy = could potentially be removed

---

**Element mesh view**

Switch Color Mode to **Element mesh** to see the actual FEA grid. Each element is colored by density — solid elements in white, void elements dark. This shows the SIMP discretization directly.

---

**Common fixes**

| Problem | Fix |
|---|---|
| 3D mesh looks like a fuzzy blob | Raise SIMP Penalty to 4.0+ |
| "No STL surface" warning | Lower iso-level slider to 0.35 or less |
| Grey zone warning | Raise penalty or max iterations |
| Print time too long | Raise print speed; reduce volume fraction |
| Chat mode not working | Check AI Provider settings, test connection |

---

**Deploy this app**

```bash
docker compose up
# Access at http://your-server-ip:8501
```
""")

# End of right column

# ─────────────────────────────────────────────────────────────
#  Material Reference  (full width below 2-panel layout)
# ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("Material Reference", expanded=False):
    cr, ca2 = st.columns(2)
    all_E_r   = [m["E_gpa"]              for m in MATERIALS.values()]
    all_rho_r = [m["rho_gcc"]            for m in MATERIALS.values()]
    all_nu_r  = [m["nu"]                 for m in MATERIALS.values()]
    all_ys_r  = [m["yield_strength_mpa"] for m in MATERIALS.values()]
    with cr:
        m2 = MATERIALS[sel_mat]
        rc = ["Stiffness","Lightness","Compressibility","Strength","Printable"]
        rv = [norm_v(m2["E_gpa"],all_E_r), 1-norm_v(m2["rho_gcc"],all_rho_r),
              norm_v(m2["nu"],all_nu_r), norm_v(m2["yield_strength_mpa"],all_ys_r),
              1.0 if m2["printable"] else 0.0]
        rv2 = rv + [rv[0]]
        rc2 = rc + [rc[0]]
        fr = go.Figure()
        fr.add_trace(go.Scatterpolar(
            r=rv2, theta=rc2, fill="toself",
            fillcolor=rgba(m2["color"],0.28),
            line=dict(color=m2["color"],width=2),
        ))
        fr.update_layout(
            polar=dict(radialaxis=dict(visible=True,range=[0,1],showticklabels=False)),
            showlegend=False, height=250,
            margin=dict(l=40,r=40,t=30,b=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fr, use_container_width=True)
    with ca2:
        fa = go.Figure()
        for name, props in MATERIALS.items():
            is_sel = (name == sel_mat)
            fa.add_trace(go.Scatter(
                x=[props["rho_gcc"]], y=[props["E_gpa"]],
                mode="markers+text",
                marker=dict(size=20 if is_sel else 12, color=props["color"],
                            line=dict(width=3 if is_sel else 1,color="white"),
                            symbol="star" if is_sel else "circle"),
                text=[name.split()[0]], textposition="top center",
                hovertemplate=f"<b>{name}</b><br>E={props['E_gpa']} GPa<extra></extra>",
            ))
        fa.update_layout(
            xaxis=dict(title="Density (g/cm³)",type="log"),
            yaxis=dict(title="E (GPa)",type="log"),
            height=250, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0e1117",
            xaxis_gridcolor="#222",yaxis_gridcolor="#222",
            margin=dict(l=50,r=10,t=10,b=50),font=dict(color="white"),
        )
        st.plotly_chart(fa, use_container_width=True)

st.divider()
st.caption(
    "v11 · Von Mises stress · Full-width viewer · Horizontal toolbar · "
    "SIMP topology optimization · "
    "Powered by NumPy · SciPy · scikit-image · Plotly · litellm"
)
