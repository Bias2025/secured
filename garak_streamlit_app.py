#!/usr/bin/env python3
"""
SecureLLM Streamlit App (Garak-powered)
======================================

This module contains the Streamlit UI and orchestration logic for running
Garak scans in a safer, Streamlit-Cloud-friendly way.

Security goals (pragmatic, not perfect):
- No arbitrary shell execution (we only invoke `python -m garak` via a controlled wrapper)
- Constrained parameters (timeouts, generations, parallelism)
- Output written to an app-owned directory under /tmp
- Optional password gate via Streamlit Secrets
- No secrets printed to logs/UI

Author: Isi Idemudia (original project) + deployment hardening
License: Apache 2.0
"""

from __future__ import annotations

import os
import re
import time
import json
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from model_registry import ModelRegistry, ModelInfo
from garak_scanner_toolkit import (
    GarakScanner,
    GarakProbeRegistry,
    ScanCategory,
    ScanConfig,
)

APP_NAME = "SecureLLM"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("SECURELLM_OUTPUT_ROOT", "/tmp/securellm_runs"))
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Security / validation helpers
# -----------------------------

_ALLOWED_MODEL_TYPES = {"openai", "anthropic", "google", "cohere", "huggingface", "replicate", "ollama", "test"}

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/\-]{0,127}$")  # conservative allowlist


def _safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:60] or "run"


def _clamp_int(val: int, lo: int, hi: int) -> int:
    try:
        val = int(val)
    except Exception:
        return lo
    return max(lo, min(hi, val))


def _validate_model_type(model_type: str) -> bool:
    return model_type in _ALLOWED_MODEL_TYPES


def _validate_model_name(model_name: str) -> bool:
    return bool(_MODEL_NAME_RE.match(model_name))


def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit Cloud uses st.secrets; locally you can use env vars as fallback.
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def _auth_gate() -> None:
    """
    Optional password gate. Set APP_PASSWORD in Streamlit Secrets or env.
    If unset, the app is open (use Streamlit Cloud's access controls if needed).
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
            st.error("Nope.")
    st.stop()


def _rate_limit(min_seconds: float = 10.0) -> None:
    """
    Very small per-session rate limiter to avoid accidental DoS of paid APIs.
    """
    last = st.session_state.get("last_run_ts", 0.0)
    now = time.time()
    if now - last < min_seconds:
        st.error(f"Rate limit: wait {min_seconds - (now - last):.1f}s before running again.")
        st.stop()
    st.session_state["last_run_ts"] = now


def _resolve_api_keys(provider: str) -> Dict[str, str]:
    """
    Pull provider keys from Streamlit Secrets or env.
    We DO NOT ask users to paste keys into the UI (keys could land in browser history).
    """
    keys: Dict[str, str] = {}

    # common keys used by garak providers
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
            # google-generativeai uses GOOGLE_API_KEY commonly
            keys["GOOGLE_API_KEY"] = v
    elif provider == "cohere":
        v = _get_secret("COHERE_API_KEY")
        if v:
            keys["COHERE_API_KEY"] = v
    elif provider == "replicate":
        v = _get_secret("REPLICATE_API_TOKEN")
        if v:
            keys["REPLICATE_API_TOKEN"] = v

    # Optional Perspective key used by some toxicity detectors
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
        "huggingface": "Usually local/inference endpoint; may need HF_TOKEN depending on your setup",
        "ollama": "Runs locally; not supported on Streamlit Cloud unless you host Ollama elsewhere",
        "test": "No keys needed",
    }
    return hints.get(provider, "")


# -----------------------------
# UI components
# -----------------------------

def _sidebar_branding() -> None:
    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.caption("Red Team Your AI Before Adversaries Do")
    st.sidebar.divider()


def _select_model(registry: ModelRegistry) -> Tuple[str, str, Optional[ModelInfo]]:
    providers = registry.get_all_providers()
    # Only show providers we can support in this Streamlit context
    providers = [p for p in providers if p in _ALLOWED_MODEL_TYPES]

    provider = st.sidebar.selectbox("Provider", providers, index=providers.index("openai") if "openai" in providers else 0)
    st.sidebar.caption(_provider_hint(provider))

    models = registry.get_models_by_provider(provider)
    names = [m.name for m in models]
    display = [f"{m.display_name} ({m.name})" for m in models]

    model_choice = st.sidebar.selectbox("Model", display, index=0 if display else None)
    model_name = names[display.index(model_choice)] if display else "test"
    model_info = registry.get_model_info(provider, model_name)
    return provider, model_name, model_info


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
    default = ["Jailbreaks", "Prompt injection", "Data leakage"]
    picked = st.multiselect("Scan categories", labels, default=default)
    selected = [cat for lbl, cat in opts if lbl in picked]
    if not selected:
        selected = [ScanCategory.JAILBREAKS]
    return selected


def _scan_safety_controls() -> Tuple[int, int, int]:
    st.subheader("Safety limits")
    col1, col2, col3 = st.columns(3)
    with col1:
        generations = st.number_input("Generations / probe", min_value=1, max_value=20, value=5, step=1)
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


def _read_latest_reports(run_dir: Path) -> Dict[str, List[Path]]:
    # Garak writes .jsonl and html reports with the prefix
    jsonl = sorted(run_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    html = sorted(run_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    md = sorted(run_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    j = sorted(run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"jsonl": jsonl, "html": html, "md": md, "json": j}


def _render_results(run_dir: Path) -> None:
    st.subheader("Artifacts")
    reports = _read_latest_reports(run_dir)

    cols = st.columns(2)
    with cols[0]:
        st.write("Files in run directory:")
        st.code("\n".join([p.name for p in sorted(run_dir.iterdir())]) or "(none)")

    with cols[1]:
        # Provide download buttons for top artifacts
        for kind in ("md", "json", "html"):
            if reports[kind]:
                p = reports[kind][0]
                st.download_button(
                    label=f"Download latest {kind.upper()}",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="text/plain" if kind in ("md", "json") else "text/html",
                )

    # Show a quick JSON summary if available
    if reports["json"]:
        try:
            summary = json.loads(reports["json"][0].read_text())
            st.subheader("Run summary")
            st.json(summary.get("summary", summary))
        except Exception:
            pass


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

    # Show model info
    if model_info:
        with st.expander("Selected model details", expanded=False):
            st.write(f"**Name:** {model_info.display_name}")
            st.write(f"**Provider:** {model_info.provider}")
            st.write(f"**Context window:** {model_info.context_window or 'N/A'}")
            st.write(f"**Cost tier:** {model_info.cost_tier or 'N/A'}")
            st.write(model_info.description)

    st.divider()

    colA, colB = st.columns([1, 1])
    with colA:
        categories = _select_scan_categories()
    with colB:
        generations, timeout, parallel = _scan_safety_controls()

    # Advanced: probe overrides (still constrained)
    with st.expander("Advanced probe selection (optional)", expanded=False):
        st.caption("You can exclude noisy probes. Custom probes are not enabled in the hosted app for safety.")
        all_probes = sorted({p for cat in [c.value for c in categories] for p in probe_registry.get_probes_by_category([ScanCategory(cat)])} if categories else set())
        # Fallback: if category list includes COMPREHENSIVE, this can be big.
        if ScanCategory.COMPREHENSIVE in categories:
            all_probes = sorted(set(probe_registry.get_probes_by_category([ScanCategory.COMPREHENSIVE])))
        exclude = st.multiselect("Exclude probes", all_probes, default=["test.Test"] if "test.Test" in all_probes else [])

    st.divider()

    # Validate inputs
    if not _validate_model_type(provider):
        st.error("Unsupported provider.")
        st.stop()
    if not _validate_model_name(model_name):
        st.error("Model name failed validation.")
        st.stop()

    # Ensure we have keys where necessary
    api_keys = _resolve_api_keys(provider)
    needs_key = provider in {"openai", "anthropic", "google", "cohere", "replicate"}
    if needs_key and not api_keys:
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

    run_btn = st.button("Run scan", type="primary", use_container_width=True)

    if run_btn:
        _rate_limit(10.0)

        run_dir = _run_dir(provider, model_name)
        report_prefix = "garak_scan"

        # Clamp safety parameters (defense-in-depth)
        generations = _clamp_int(generations, 1, 20)
        timeout = _clamp_int(timeout, 60, 7200)
        parallel = _clamp_int(parallel, 1, 4)

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
            api_keys=api_keys or None,
        )

        st.info(f"Run directory: {run_dir}")
        st.write("Starting scan…")

        scanner = GarakScanner(config)

        with st.spinner("Running garak… this can take a while depending on probe set."):
            try:
                results = scanner.run_comprehensive_scan()
                st.success("Scan completed.")
                # Persist a small run metadata file for convenience
                meta = {"provider": provider, "model": model_name, "run_dir": str(run_dir), "summary": results.get("summary", {})}
                (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
                _render_results(run_dir)
            except Exception as e:
                st.error(f"Scan failed: {e}")
                # Show a redacted snippet for debugging without leaking secrets
                st.caption("Tip: verify garak is installed and the provider key is set in Streamlit Secrets.")

    # Show previous runs on this instance (best-effort; ephemeral on Streamlit Cloud)
    st.divider()
    st.subheader("Recent runs on this server")
    runs = sorted([p for p in DEFAULT_OUTPUT_ROOT.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    if not runs:
        st.caption("No runs yet.")
    else:
        for p in runs:
            with st.expander(p.name, expanded=False):
                meta = p / "run_metadata.json"
                if meta.exists():
                    try:
                        st.json(json.loads(meta.read_text()))
                    except Exception:
                        pass
                st.write(f"Path: {p}")
                if st.button(f"Show artifacts for {p.name}", key=f"show_{p.name}"):
                    _render_results(p)
