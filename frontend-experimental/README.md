# frontend-experimental — React + FastAPI prototype (planned v2)

This directory contains a React + FastAPI implementation of MorphAI that is **not currently deployed**.

## What it is

- `api.py` — FastAPI backend that wraps the SIMP optimizer pipeline and serves the React frontend as static files. No LLM integration. Exposes `POST /api/optimize`, `GET /api/materials`, `GET /api/runs`, and STL download.
- `frontend/index.html` — Single-file React app (no build step, loads React via CDN). Provides a parameter form, calls `api.py`, and renders the density grid result inline.
- `run_server.bat` — Windows convenience script: `uvicorn api:app --host 0.0.0.0 --port 8000`

## Why it exists

The deployed Streamlit app (`app.py`) has a richer feature set but ships Streamlit's full overhead. This prototype explores a lighter architecture: FastAPI backend + React frontend, no Streamlit dependency, faster cold start, cleaner API surface.

## Planned v2 work

Integrating the LLM extraction layer from `chat.py` (the deployed Streamlit app) into this frontend is the primary remaining gap. Once the NL → boundary condition extraction is wired to `api.py`, this becomes a drop-in replacement for the Streamlit deployment with a significantly smaller Docker image and no Streamlit session state overhead.

## Running locally

```
pip install fastapi uvicorn[standard]
cd frontend-experimental
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.
