---
title: MorphAI
emoji: 🔧
sdk: docker
pinned: false
---

# MorphAI — AI-Driven Topology Optimizer for FDM 3D Printing

Describe a mechanical part in plain English. MorphAI extracts the boundary conditions, runs the SIMP topology optimization algorithm, and gives you a print-ready STL with calibrated von Mises stress output — in under 60 seconds.

[Live demo on HF Spaces](https://huggingface.co/spaces/cdshelat/MorphAI) · [Watch the demo](TODO)

---

## What's interesting about this

- **Natural language → engineering constraints.** A Claude/GPT extraction layer reads a plain-English description ("wall bracket, holds 20 kg, fixed left, PLA") and extracts face constraints, force direction, magnitude, and material — no form-filling required. This is one of the few AI generalist projects applying LLMs to physics-based optimization rather than text or image tasks.

- **Hand-implemented SIMP solver.** The topology optimizer is not a wrapper around an existing solver library. The SIMP (Solid Isotropic Material with Penalization) algorithm, sparse FEA stiffness assembly, density filter, and optimality criteria update are written from scratch using NumPy and SciPy sparse. The same algorithm is used in aerospace and automotive design tools costing tens of thousands of dollars.

- **2D + 3D solver pipeline.** The 2D Q4 plane-stress solver handles most bracket and plate problems. A true 3D hexahedral solver (`simp_core_3d`) is available for volumetric parts — same SIMP loop, 24 DOF per hex element, full 3D stress and displacement fields.

- **Physically calibrated stress output.** Von Mises stress is returned in real MPa (not normalized), scaled by actual force magnitude and element size, compared against material yield strength with a user-chosen safety factor. The result is actionable for real print decisions.

---

## Tech stack

Streamlit · NumPy · SciPy sparse · scikit-image (marching cubes) · Plotly 3D · litellm (multi-provider LLM gateway) · trimesh

---

## Running locally

```bash
git clone https://github.com/chintan-dshel/morph-ai
cd morph-ai
pip install -r requirements.txt
streamlit run app.py
```

Click a **Quick Start preset** — results in ~30 seconds, no API key needed. To enable AI chat extraction, add your key in the AI Settings panel (never written to disk). With `ANTHROPIC_API_KEY` set as an environment variable, the key is loaded automatically.

---

## Secrets (for Space operators)

Set `ANTHROPIC_API_KEY` as a Space Secret to enable the chat extraction feature for all visitors without requiring them to supply their own key.

---

## Planned v2

A React + FastAPI prototype with no Streamlit dependency lives in `frontend-experimental/`. Planned work: wire the LLM extraction layer from `chat.py` into the FastAPI backend.

---

## Validation

The MBB beam benchmark (Sigmund 2001) is built in. Under Convergence → MBB Benchmark:

- nelx=60, nely=20, volfrac=0.5, p=3, rmin=1.5
- Expected compliance: **C ≈ 187.96**
- Pass threshold: < 5% error

---

## License

MIT
