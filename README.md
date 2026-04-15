# MorphAI — AI-Driven Topology Optimizer for FDM 3D Printing

Describe a mechanical part in plain English. MorphAI extracts the boundary conditions, runs the SIMP topology optimization algorithm, and gives you a print-ready STL with calibrated von Mises stress output — in under 60 seconds.

![MorphAI screenshot](docs/screenshot.png)

---

## What it does

Topology optimization answers: *given a design space, material, and loads — what is the minimum-material structure that won't fail?* Professional tools (Altair OptiStruct, ANSYS) cost tens of thousands of dollars and require engineering expertise to set up. MorphAI makes the same physics accessible through a chat interface.

**Core capabilities:**
- **Natural language → boundary conditions**: Tell the AI "I need a wall bracket that holds 20 kg, fixed on the left, made of PLA." It extracts face constraints, force direction, magnitude, and material — no form-filling.
- **SIMP optimizer**: The same algorithm used in aerospace and automotive design. Drives material to zero where it isn't structurally needed.
- **Physical stress output**: Von Mises stress in real MPa (not normalized). Compares against material yield strength with your chosen safety factor.
- **Print-ready STL**: Export the optimized topology directly to your slicer. Optional infill pattern generation (Gyroid, Schwartz-P, Honeycomb, Diamond).
- **Multi-provider AI**: Works with OpenAI, Anthropic Claude, Google Gemini, Ollama (local), and any litellm-supported model. Presets work with no API key at all.

---

## Quick start (local)

```bash
git clone https://github.com/your-username/morphai
cd morphai
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. Click a **Quick Start preset** — results in ~30 seconds, no API key needed.

To enable AI chat extraction, add your key in the AI Settings panel (never written to disk).

---

## Tech stack

| Layer | What |
|---|---|
| UI | Streamlit |
| Optimizer | SIMP (Solid Isotropic Material with Penalization) — 2D Q4 plane-stress FEA + 3D hex |
| FEA solver | SciPy sparse LU (`spsolve`) |
| Surface extraction | scikit-image marching cubes |
| Mesh voxelization | trimesh |
| AI extraction | litellm (multi-provider gateway) |
| Visualization | Plotly 3D |
| FEM mesh export | gmsh (optional, local only) |

---

## How the physics works

The optimizer minimizes structural compliance (= maximizes stiffness) subject to a volume constraint:

```
minimize  C = F^T U
subject to  K(x) U = F,   sum(x_e) ≤ V* · n
```

Each element has a density `x_e ∈ [0,1]`. The SIMP penalty `x^p` (p=3 by default) makes intermediate densities structurally expensive, driving the result toward a crisp solid/void topology.

Stress output is physically calibrated:
```
σ_phys [MPa] = σ_norm × F_mag [N] / (dx_mm × thickness_mm)
```

This is exact for plane-stress Q4 elements — the material stiffness cancels in the derivation.

---

## Deployment (Streamlit Cloud)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your fork → `app.py`
3. Add secrets in the Streamlit Cloud dashboard (optional):
   ```toml
   # .streamlit/secrets.toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY    = "sk-..."
   ```
4. Deploy — gmsh is disabled automatically on Cloud (all other features work)

---

## Validation

The MBB beam benchmark (Sigmund 2001) is built in. Under Convergence → MBB Benchmark:
- nelx=60, nely=20, volfrac=0.5, p=3, rmin=1.5
- Expected compliance: **C ≈ 187.96**
- Pass threshold: < 5% error

---

## Structure

```
app.py          — Streamlit UI (orchestrator)
optimizer.py    — SIMP core, filter, load case builders
visualization.py — Plotly 3D traces, marching cubes surface
geometry.py     — Infill generation, STL export
materials.py    — Material database, Ashby analysis
chat.py         — LLM extraction, multi-provider config
meshing.py      — FEM mesh generation (gmsh, optional)
utils.py        — Print time estimator, history records
```

---

## License

MIT
