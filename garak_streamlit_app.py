#!/usr/bin/env python3
"""
SecureLLM Streamlit App (Garak-powered)
======================================

A Streamlit UI to run Garak scans in a Streamlit-Cloud-friendly way.

Security / ops goals:
- No arbitrary shell execution (only `sys.executable -m garak` via GarakScanner wrapper)
- Constrained scan parameters (timeouts, generations, parallelism)
- Outputs written to an app-owned directory under /tmp (ephemeral on Streamlit Cloud)
- Optional password gate via Streamlit Secrets (APP_PASSWORD)
- No secrets printed to logs/UI (masked diagnostics only)

Repo expectations:
- model_registry.py provides ModelRegistry + ModelInfo
- garak_scanner_toolkit.py provides GarakScanner, GarakProbeRegistry, ScanCategory, ScanConfig

License: Apache 2.0
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from model_registry import ModelRegistry, ModelInfo
from garak_scanner_toolkit import GarakScanner, GarakProbeRegistry, ScanCategory, ScanConfig

APP_NAME = "SecureLLM"

# Streamlit Cloud: /tmp is writable, but ephemeral.
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("SECURELLM_OUTPUT_ROOT", "/tmp/securellm_runs"))
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Providers we allow through the UI (defense-in-depth)
_ALLOWED_MODEL_TYPES = {"openai", "anthropic", "google", "cohere", "huggingface", "replicate", "ollama", "test"}

# Conservative allowlist for model names (avoid weird injection / path behavior)
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/\-]{0,127}$")


# -----------------------------
# Small utilities
# -----------------------------

def _safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s[:60] or "run")


def _clamp_int(val: int, lo: int, hi: int) -> int:
    try:
        v = int(val)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _validate_model_type(model_type: str) -> bool:
    return model_type in _ALLOWED_MODEL_TYPES


def _validate_model_name(model_name: str) -> bool:
    return bool(_MODEL_NAME_RE.match(model_name or ""))


def _mask(s: str, keep: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= keep:
        return "*" * len(s)
    return "*" * (len(s) - keep) + s[-keep:]


def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Streamlit Cloud: secrets are in st.secrets (not automatically exported to env).
    Local dev: can fallback to environment variables.
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def _export_api_keys_to_env(api_keys: Dict[str, str]) -> None:
    """
    Ensure keys are available to subprocesses (garak runs in a subprocess).
    """
    for k, v in (api_keys or {}).items():
        if v:
            os.environ[k] = v


def _auth_gate() -> None:
    """
    Optional password gate. Set APP_PASSWORD in Streamlit Secrets or env.
    If unset, the app is open.
    """
    app_password = _get_secret("APP_PASSWORD")
    if not app_password:
        return

    if st.session_state.get("authed") is True:
        return

    st.warning("Password required.")
    pw = st.text_input("Enter password", type="password")
    if st.button("Unlock"):
        if secrets.compare_digest(pw or "", app_password):
            st.session_state["authed"] = True
            st.success("Unlocked.")
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def _rate_limit(min_seconds: float = 10.0) -> None:
    """
    Very small per-session rate limiter to avoid accidental API spam.
    """
    last = float(st.session_state.get("last_run_ts", 0.0))
    now = time.time()
    if now - last < min_seconds:
        st.error(f"Rate limit: wait {min_seconds - (now - last):.1f}s before running again.")
        st.stop()
    st.session_state["last_run_ts"] = now


def _resolve_api_keys(provider: str) -> Dict[str, str]:
    """
    Pull provider keys from Streamlit Secrets or env.
    No UI entry for keys (avoid browser history / accidental leaks).
    """
    keys: Dict[str, str] = {}

    if provider == "openai":
        v = _get_secret("OPENAI_API_KEY")
        if v:
            keys["OPENAI_API_KEY"] = v

    elif provider == "anthropic":
        v = _get_secret("ANTHROPIC_API_KEY")
        if v:
            keys["ANTHROPIC_API_KEY"] = v

    elif provider == "google":
        v = _get_secret("GOOGLE_API_KEY") or _get_secret("GEMINI_API_KEY")
        if v:
            keys["GOOGLE_API_KEY"] = v

    elif provider == "cohere":
        v = _get_secret("COHERE_API_KEY")
        if v:
            keys["COHERE_API_KEY"] = v

    elif provider == "replicate":
        v = _get_secret("REPLICATE_API_TOKEN")
        if v:
            keys["REPLICATE_API_TOKEN"] = v

    # Optional key used by some toxicity-related detectors
    v = _get_secret("PERSPECTIVE_API_KEY")
    if v:
        keys["PERSPECTIVE_API_KEY"] = v

    return keys


def _provider_hint(provider: str) -> str:
    hints = {
        "openai": "Needs OPENAI_API_KEY",
        "anthropic": "Needs ANTHROPIC_API_KEY",
        "google": "Needs GOOGLE_API_KEY (or GEMINI_API_KEY)",
        "cohere": "Needs COHERE_API_KEY",
        "replicate": "Needs REPLICATE_API_TOKEN",
        "huggingface": "May need HF_TOKEN depending on your setup",
        "ollama": "Local only (Streamlit Cloud won't run Ollama locally)",
        "test": "No keys needed",
    }
    return hints.get(provider, "")


# -----------------------------
# UI helpers
# -----------------------------

def _sidebar_branding() -> None:
    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.caption("Red Team Your AI Before Adversaries Do")
    st.sidebar.divider()


def _select_model(registry: ModelRegistry) -> Tuple[str, str, Optional[ModelInfo]]:
    providers = [p for p in registry.get_all_providers() if p in _ALLOWED_MODEL_TYPES]
    if not providers:
        providers = ["test"]

    default_provider_idx = providers.index("openai") if "openai" in providers else 0
    provider = st.sidebar.selectbox("Provider", providers, index=default_provider_idx)
    st.sidebar.caption(_provider_hint(provider))

    models = registry.get_models_by_provider(provider)
    if not models:
        # fallback
        return provider, "test", None

    display = [f"{m.display_name} ({m.name})" for m in models]
    choice = st.sidebar.selectbox("Model", display, index=0)
    picked = models[display.index(choice)]
    return provider, picked.name, registry.get_model_info(provider, picked.name)


def _select_scan_categories() -> List[ScanCategory]:
    opts = [
        ("Comprehensive", ScanCategory.COMPREHENSIVE),
        ("Jailbreaks", ScanCategory.JAILBREAKS),
        ("Prompt injection", ScanCategory.PROMPT_INJECTION),
        ("Data leakage", ScanCategory.DATA_LEAKAGE),
        ("Toxicity", ScanCategory.TOXICITY),
        ("Malware", ScanCategory.MALWARE),
        ("Hallucination", ScanCategory.HALLUCINATION),
        ("Encoding", ScanCategory.ENCODING),
    ]
    labels = [o[0] for o in opts]
    default = ["Jailbreaks", "Prompt injection", "Data leakage", "Toxicity"]
    picked = st.multiselect("Scan categories", labels, default=default)
    selected = [cat for lbl, cat in opts if lbl in picked]
    return selected or [ScanCategory.JAILBREAKS]


def _scan_safety_controls() -> Tuple[int, int, int]:
    st.subheader("Safety limits")
    col1, col2, col3 = st.columns(3)
    with col1:
        generations = st.number_input("Generations / probe", min_value=1, max_value=20, value=3, step=1)
    with col2:
        timeout = st.number_input("Timeout (sec) / batch", min_value=60, max_value=7200, value=1200, step=60)
    with col3:
        parallel = st.number_input("Parallel batches", min_value=1, max_value=4, value=1, step=1)
    return int(generations), int(timeout), int(parallel)


def _run_dir(provider: str, model_name: str) -> Path:
    run_id = f"{_safe_slug(provider)}-{_safe_slug(model_name)}-{int(time.time())}"
    out = DEFAULT_OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_reports(run_dir: Path) -> Dict[str, List[Path]]:
    jsonl = sorted(run_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    html = sorted(run_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    md = sorted(run_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    j = sorted(run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"jsonl": jsonl, "html": html, "md": md, "json": j}


def _render_results(run_dir: Path) -> None:
    st.subheader("Artifacts")
    reports = _read_reports(run_dir)

    cols = st.columns(2)
    with cols[0]:
        st.write("Files in run directory:")
        try:
            st.code("\n".join([p.name for p in sorted(run_dir.iterdir())]) or "(none)")
        except Exception:
            st.code("(unable to list files)")

    with cols[1]:
        for kind in ("md", "json", "html"):
            if reports[kind]:
                p = reports[kind][0]
                st.download_button(
                    label=f"Download latest {kind.upper()}",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="text/plain" if kind in ("md", "json") else "text/html",
                )

    # quick summary if our wrapper JSON exists
    if reports["json"]:
        for p in reports["json"]:
            # Prefer scan_results_*.json (contains summary), fall back to run_metadata.json
            if p.name.startswith("scan_results_") or p.name == "run_metadata.json":
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    st.subheader("Run summary")
                    st.json(data.get("summary", data))
                except Exception:
                    pass
                break


# -----------------------------
# Main app
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="SecureLLM", layout="wide")
    _auth_gate()

    registry = ModelRegistry()
    probe_registry = GarakProbeRegistry()

    _sidebar_branding()
    provider, model_name, model_info = _select_model(registry)

    st.title("SecureLLM")
    st.caption("A Streamlit front-end for Garak-based LLM red teaming. Be nice to your API bill.")
    st.divider()

    # Model details
    if model_info:
        with st.expander("Selected model details", expanded=False):
            st.write(f"**Name:** {model_info.display_name}")
            st.write(f"**Provider:** {model_info.provider}")
            st.write(f"**Context window:** {model_info.context_window or 'N/A'}")
            st.write(f"**Cost tier:** {model_info.cost_tier or 'N/A'}")
            if model_info.description:
                st.write(model_info.description)

    colA, colB = st.columns([1, 1])
    with colA:
        categories = _select_scan_categories()
    with colB:
        generations, timeout, parallel = _scan_safety_controls()

    # Advanced exclusions
    with st.expander("Advanced probe selection (optional)", expanded=False):
        st.caption("Exclude noisy probes. (Custom probes are disabled in hosted mode.)")
        if ScanCategory.COMPREHENSIVE in categories:
            all_probes = sorted(set(probe_registry.get_probes_by_category([ScanCategory.COMPREHENSIVE])))
        else:
            all_probes = sorted(set(probe_registry.get_probes_by_category(categories)))

        default_exclude = ["test.Test"] if "test.Test" in all_probes else []
        exclude = st.multiselect("Exclude probes", all_probes, default=default_exclude)

    st.divider()

    # Validate inputs early
    if not _validate_model_type(provider):
        st.error("Unsupported provider.")
        st.stop()
    if not _validate_model_name(model_name):
        st.error("Model name failed validation.")
        st.stop()

    api_keys = _resolve_api_keys(provider)
    needs_key = provider in {"openai", "anthropic", "google", "cohere", "replicate"}
    missing_key = needs_key and not api_keys

    if missing_key:
        st.warning(
            "No API key detected for this provider. Add it in Streamlit Secrets "
            "(recommended) or as an environment variable."
        )
        with st.expander("Secret names to set"):
            st.code(
                "\n".join(
                    [
                        "OPENAI_API_KEY",
                        "ANTHROPIC_API_KEY",
                        "GOOGLE_API_KEY (or GEMINI_API_KEY)",
                        "COHERE_API_KEY",
                        "REPLICATE_API_TOKEN",
                        "PERSPECTIVE_API_KEY (optional)",
                        "APP_PASSWORD (optional)",
                    ]
                )
            )

    with st.expander("Key diagnostics (masked)", expanded=False):
        if api_keys:
            st.write({k: _mask(v) for k, v in api_keys.items()})
        else:
            st.write("(no keys detected)")

    run_btn = st.button(
        "Run scan",
        type="primary",
        use_container_width=True,
        disabled=missing_key,
    )

    if run_btn:
        _rate_limit(10.0)

        # Clamp safety limits (defense-in-depth)
        generations = _clamp_int(generations, 1, 20)
        timeout = _clamp_int(timeout, 60, 7200)
        parallel = _clamp_int(parallel, 1, 4)

        run_dir = _run_dir(provider, model_name)
        report_prefix = "garak_scan"

        # Export keys so subprocesses definitely see them
        _export_api_keys_to_env(api_keys)

        config = ScanConfig(
            target_model=model_name,
            model_type=provider,
            scan_categories=categories,
            output_dir=str(run_dir),
            report_prefix=report_prefix,
            max_generations=generations,
            timeout=timeout,
            parallel_probes=parallel,
            custom_probes=None,  # disabled in hosted mode
            exclude_probes=exclude or None,
            api_keys=api_keys or None,  # wrapper also exports, but keep for completeness
        )

        st.info(f"Run directory: {run_dir}")
        st.write("Starting scan…")

        scanner = GarakScanner(config)

        with st.spinner("Running garak… this can take a while depending on probe set."):
            try:
                results = scanner.run_comprehensive_scan()
                st.success("Scan completed.")

                # Persist small run metadata for the “Recent runs” list
                meta = {
                    "provider": provider,
                    "model": model_name,
                    "run_dir": str(run_dir),
                    "categories": [c.value for c in categories],
                    "summary": results.get("summary", {}),
                }
                (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

                _render_results(run_dir)

            except Exception as e:
                st.error(f"Scan failed: {e}")
                st.caption("Tip: confirm provider key is set in Streamlit Secrets and garak is installed.")

    # Recent runs (best-effort; ephemeral on Streamlit Cloud)
    st.divider()
    st.subheader("Recent runs on this server")

    try:
        runs = sorted(
            [p for p in DEFAULT_OUTPUT_ROOT.glob("*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:10]
    except Exception:
        runs = []

    if not runs:
        st.caption("No runs yet.")
    else:
        for p in runs:
            with st.expander(p.name, expanded=False):
                meta = p / "run_metadata.json"
                if meta.exists():
                    try:
                        st.json(json.loads(meta.read_text(encoding="utf-8")))
                    except Exception:
                        pass
                st.write(f"Path: {p}")
                if st.button(f"Show artifacts for {p.name}", key=f"show_{p.name}"):
                    _render_results(p)


if __name__ == "__main__":
    main()
