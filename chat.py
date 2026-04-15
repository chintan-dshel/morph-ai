"""
chat.py — Multi-provider AI chat interface for PhysicsSim.

Uses litellm as a universal gateway so the user can choose any LLM provider
(OpenAI, Anthropic Claude, Gemini, Ollama, Azure, etc.) by supplying their own
API key and model string in the UI.  No key is ever written to disk.
"""

import json
import os
import re

# Provider presets shown in the UI dropdown
PROVIDER_PRESETS = {
    "Anthropic Claude": {
        "models": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
        "key_env": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
        "litellm_prefix": "",          # litellm uses bare model name for Claude
    },
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o3-mini"],
        "key_env": "OPENAI_API_KEY",
        "key_prefix": "sk-",
        "litellm_prefix": "",
    },
    "Google Gemini": {
        "models": ["gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash"],
        "key_env": "GEMINI_API_KEY",
        "key_prefix": "",
        "litellm_prefix": "gemini/",
    },
    "Ollama (local)": {
        "models": ["ollama/llama3.2", "ollama/llama3.1", "ollama/mistral", "ollama/phi3"],
        "key_env": "",
        "key_prefix": "",
        "litellm_prefix": "ollama/",
    },
}

# System prompt that seeds any LLM with the task context
SYSTEM_PROMPT = """You are a structural engineering assistant for a topology optimization tool called PhysicsSim.
Your job is to extract simulation parameters from a user's natural language description of a mechanical part.

The design space is a rectangular bounding box with exactly 6 named faces:
  left, right, top, bottom, front, back

Rules:
- fixed_face = the face that is bolted/welded/attached to a rigid surface
- load_face  = the face where the external force is applied
- force_direction must be one of: -X, +X, -Y, +Y, -Z, +Z
  (-Y = downward/gravity, +Y = upward, -X/+X = horizontal left/right, -Z/+Z = depth)
- Mass → Force: multiply kg × 9.81 to get Newtons (e.g. "20 kg" → 196.2 N)
- If no material is mentioned, default to "PLA (Bioplastic)"
- If no safety factor is mentioned, default to 2.0
- If no volume fraction / fill percentage is mentioned, default to 0.4

Available materials (use EXACT name from this list):
  PLA (Bioplastic), PETG (Engineering Plastic), ABS (Acrylonitrile Butadiene Styrene),
  Nylon PA12, TPU (Flexible), Carbon Fiber PETG (Composite),
  Titanium Ti-6Al-4V (Reference), Aluminum 6061 (Reference)

Respond ONLY with a valid JSON object — no markdown, no prose, no code fences. Example:
{
  "fixed_face": "left",
  "load_face": "right",
  "force_direction": "-Y",
  "applied_force_n": 196.2,
  "material": "PLA (Bioplastic)",
  "safety_factor": 2.0,
  "volume_fraction": 0.4,
  "load_scenario": "Downward only",
  "confidence_notes": "Assumed downward gravity for 20kg load."
}

load_scenario must be one of:
  Downward only, Lateral only, Down + Lateral, Down + Upward (top),
  Down + Downward (bot), Symmetric (top+bot), Torsion (top+bot opp)
"""

REQUIRED_KEYS = [
    "fixed_face", "load_face", "force_direction",
    "applied_force_n", "material",
]

VALID_FACES = {"left", "right", "top", "bottom", "front", "back"}
VALID_DIRECTIONS = {"-X", "+X", "-Y", "+Y", "-Z", "+Z"}
VALID_SCENARIOS = {
    "Downward only", "Lateral only", "Down + Lateral",
    "Down + Upward (top)", "Down + Downward (bot)",
    "Symmetric (top+bot)", "Torsion (top+bot opp)",
}

# Map from chat-extracted face names (no suffix) → app FACES list values
FACE_MAP = {
    "left":   "Left (X=0)",
    "right":  "Right (X=W)",
    "bottom": "Bottom (Y=0)",
    "top":    "Top (Y=H)",
    "front":  "Front (Z=0)",
    "back":   "Back (Z=D)",
}


def get_api_key_from_env(provider_name: str) -> str:
    """Check environment / Streamlit secrets for a pre-configured API key."""
    preset = PROVIDER_PRESETS.get(provider_name, {})
    env_var = preset.get("key_env", "")
    if not env_var:
        return ""
    # Try st.secrets first (Streamlit Cloud / local secrets.toml)
    try:
        import streamlit as st
        return st.secrets.get(env_var, "")
    except Exception:
        pass
    return os.environ.get(env_var, "")


def extract_params(user_message: str, model: str, api_key: str,
                   provider_name: str = "") -> dict:
    """Call the chosen LLM and extract simulation parameters as a dict.

    Returns a validated dict with keys matching REQUIRED_KEYS plus optional keys.
    Raises ValueError with a human-readable message on failure.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is not installed. Run: pip install litellm"
        )

    # Build litellm kwargs
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    # Set API key via environment variable so litellm picks it up
    preset = PROVIDER_PRESETS.get(provider_name, {})
    env_var = preset.get("key_env", "")
    original_val = None
    if env_var and api_key:
        original_val = os.environ.get(env_var)
        os.environ[env_var] = api_key

    # Ollama doesn't need a key
    if provider_name == "Ollama (local)":
        kwargs.pop("temperature", None)

    try:
        response = litellm.completion(**kwargs)
        raw = response.choices[0].message.content.strip()
    finally:
        # Restore env var
        if env_var and original_val is not None:
            os.environ[env_var] = original_val
        elif env_var and api_key:
            os.environ.pop(env_var, None)

    # Parse JSON — strip any accidental markdown fences
    raw_clean = re.sub(r"^```[a-z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        params = json.loads(raw_clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model returned invalid JSON: {e}\n\nRaw response:\n{raw[:400]}"
        )

    # Validate required keys
    missing = [k for k in REQUIRED_KEYS if k not in params]
    if missing:
        raise ValueError(
            f"Model response is missing required fields: {missing}\n"
            f"Try rephrasing your description to include: "
            f"which face is fixed, which face receives the load, "
            f"force direction, force magnitude, and material."
        )

    # Validate values
    if params.get("fixed_face") not in VALID_FACES:
        raise ValueError(
            f"fixed_face '{params.get('fixed_face')}' is not valid. "
            f"Must be one of: {sorted(VALID_FACES)}"
        )
    if params.get("load_face") not in VALID_FACES:
        raise ValueError(
            f"load_face '{params.get('load_face')}' is not valid. "
            f"Must be one of: {sorted(VALID_FACES)}"
        )
    if params.get("force_direction") not in VALID_DIRECTIONS:
        raise ValueError(
            f"force_direction '{params.get('force_direction')}' is not valid. "
            f"Must be one of: {sorted(VALID_DIRECTIONS)}"
        )

    # Apply defaults for optional keys
    params.setdefault("safety_factor", 2.0)
    params.setdefault("volume_fraction", 0.4)
    params.setdefault("load_scenario", "Downward only")
    params.setdefault("confidence_notes", "")

    # Clamp numbers to safe ranges
    params["applied_force_n"] = max(1.0, float(params["applied_force_n"]))
    params["safety_factor"]   = max(1.0, min(5.0, float(params["safety_factor"])))
    params["volume_fraction"] = max(0.1, min(0.9, float(params["volume_fraction"])))

    # Map bare face name → app-style label with suffix
    params["fixed_face"] = FACE_MAP.get(params["fixed_face"], params["fixed_face"])
    params["load_face"]  = FACE_MAP.get(params["load_face"],  params["load_face"])

    # Ensure load_scenario is valid
    if params["load_scenario"] not in VALID_SCENARIOS:
        params["load_scenario"] = "Downward only"

    return params


def call_llm(model: str, api_key: str, prompt: str,
             provider_name: str = "") -> str:
    """Call the LLM with a plain-text prompt and return the response as a string.

    Unlike extract_params, this does NOT enforce JSON — it is used for free-form
    commentary (e.g. topology narration).
    """
    try:
        import litellm
    except ImportError:
        raise ImportError("litellm is not installed. Run: pip install litellm")

    preset = PROVIDER_PRESETS.get(provider_name, {})
    env_var = preset.get("key_env", "")
    # Auto-detect provider from model string when provider_name is empty
    if not env_var:
        if model.startswith("claude"):
            env_var = "ANTHROPIC_API_KEY"
        elif model.startswith("gpt") or model.startswith("o3") or model.startswith("o1"):
            env_var = "OPENAI_API_KEY"
        elif model.startswith("gemini"):
            env_var = "GEMINI_API_KEY"

    original_val = None
    if env_var and api_key:
        original_val = os.environ.get(env_var)
        os.environ[env_var] = api_key

    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    finally:
        if env_var and original_val is not None:
            os.environ[env_var] = original_val
        elif env_var and api_key:
            os.environ.pop(env_var, None)


def test_connection(model: str, api_key: str, provider_name: str) -> tuple[bool, str]:
    """Quick connectivity test — sends a tiny message to verify the key works.

    Returns (success: bool, message: str).
    """
    try:
        import litellm
    except ImportError:
        return False, "litellm not installed. Run: pip install litellm"

    preset = PROVIDER_PRESETS.get(provider_name, {})
    env_var = preset.get("key_env", "")
    original_val = None
    if env_var and api_key:
        original_val = os.environ.get(env_var)
        os.environ[env_var] = api_key

    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        return True, f"Connected — model replied: \"{text}\""
    except Exception as e:
        return False, f"Connection failed: {str(e)[:200]}"
    finally:
        if env_var and original_val is not None:
            os.environ[env_var] = original_val
        elif env_var and api_key:
            os.environ.pop(env_var, None)
