"""Run metadata collection for benchmark context (issue #6).

Builds a RunContext with three tiers of metadata:
  1. Local environment (always available)
  2. CLI parameters (passed in by caller)
  3. Inference engine probe (best-effort, HTTP calls with tight timeouts)
"""

from __future__ import annotations

import logging
import os
import platform
import re
import socket
import subprocess
from typing import Any

import httpx

from tool_eval_bench.domain.models import BenchmarkConfig, RunContext
from tool_eval_bench.utils.urls import models_url as _models_url

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT = 5  # seconds — tight timeout for engine probes


# ---------------------------------------------------------------------------
# Tier 1: local environment
# ---------------------------------------------------------------------------

def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _tool_version() -> str:
    from tool_eval_bench import __version__
    return __version__


# ---------------------------------------------------------------------------
# Tier 3: inference engine probing (best-effort)
# ---------------------------------------------------------------------------

async def _probe_models(
    base_url: str, api_key: str | None,
) -> dict[str, Any]:
    """Probe /v1/models for model metadata."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    probe: dict[str, Any] = {}
    url = _models_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                data = body.get("data") if isinstance(body, dict) else None
                if isinstance(data, list) and data:
                    first = data[0] if isinstance(data[0], dict) else {}
                    probe["server_model_id"] = first.get("id")
                    probe["server_model_root"] = first.get("root")
                    # vLLM exposes max_model_len in model metadata
                    if "max_model_len" in first:
                        probe["max_model_len"] = first["max_model_len"]
    except Exception as exc:
        logger.debug("models probe failed: %s", exc)
    return probe


async def _probe_vllm_version(base_url: str, api_key: str | None) -> dict[str, Any]:
    """Probe /version (vLLM-specific endpoint)."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/version"
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict) and "version" in body:
                    return {
                        "engine_name": "vLLM",
                        "engine_version": body["version"],
                    }
    except Exception as exc:
        logger.debug("vLLM /version probe failed: %s", exc)
    return {}


async def _probe_llamacpp(base_url: str) -> dict[str, Any]:
    """Probe /health or /props (llama.cpp endpoints)."""
    for path in ["/props", "/health"]:
        url = f"{base_url.rstrip('/')}{path}"
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    body = resp.json()
                    if isinstance(body, dict):
                        result: dict[str, Any] = {"engine_name": "llama.cpp"}
                        if "build_info" in body:
                            result["engine_version"] = str(body["build_info"])
                        elif "build_number" in body:
                            result["engine_version"] = f"b{body['build_number']}"
                        if "total_slots" in body:
                            result["gpu_count"] = body.get("total_slots")
                        return result
        except Exception as exc:
            logger.debug("llama.cpp %s probe failed: %s", path, exc)
    return {}


async def _probe_litellm(base_url: str, api_key: str | None) -> dict[str, Any]:
    """Detect LiteLLM from response headers or /health."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
            # LiteLLM sets x-litellm-version header
            version = resp.headers.get("x-litellm-version")
            if version:
                return {"engine_name": "LiteLLM", "engine_version": version}
            # Fallback: check response body
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict) and "litellm_version" in body:
                    return {
                        "engine_name": "LiteLLM",
                        "engine_version": body["litellm_version"],
                    }
    except Exception as exc:
        logger.debug("LiteLLM /health probe failed: %s", exc)
    return {}


def _guess_quantization(model_name: str | None) -> str | None:
    """Infer quantization from model name heuristics."""
    if not model_name:
        return None
    upper = model_name.upper()
    # AutoRound pattern (check before generic INT4/INT8)
    if "AUTOROUND" in upper:
        int_match = re.search(r"INT(\d+)", upper)
        if int_match:
            return f"INT{int_match.group(1)}-AutoRound"
        return "AutoRound"
    # GGUF quantization levels like Q4_K_M, Q5_K_S (check before generic GGUF)
    gguf_match = re.search(r"(Q\d+_K_?\w?)", upper)
    if gguf_match:
        return gguf_match.group(1)
    # Simple keyword match
    for q in ["AWQ", "GPTQ", "GGUF", "EXL2", "BNBQ4", "BNB4", "INT8", "INT4", "FP8", "FP16", "BF16"]:
        if q in upper:
            return q
    return None


async def _probe_engine(
    base_url: str, api_key: str | None, backend: str,
) -> dict[str, Any]:
    """Run all engine probes and merge results. Best-effort."""
    result: dict[str, Any] = {}

    # Always probe /v1/models (works for all backends)
    models_info = await _probe_models(base_url, api_key)
    result.update(models_info)

    # Backend-specific probes
    backend_l = backend.lower()
    if backend_l == "vllm":
        vllm_info = await _probe_vllm_version(base_url, api_key)
        result.update(vllm_info)
    elif backend_l in ("llamacpp", "llama.cpp", "llama_cpp"):
        llama_info = await _probe_llamacpp(base_url)
        result.update(llama_info)
    elif backend_l == "litellm":
        litellm_info = await _probe_litellm(base_url, api_key)
        result.update(litellm_info)
    else:
        # Try all in order (vLLM first as most common)
        for prober in [
            lambda: _probe_vllm_version(base_url, api_key),
            lambda: _probe_litellm(base_url, api_key),
            lambda: _probe_llamacpp(base_url),
        ]:
            info = await prober()
            if info:
                result.update(info)
                break

    # Infer quantization from model name
    if "quantization" not in result:
        model_root = result.get("server_model_root") or result.get("server_model_id")
        quant = _guess_quantization(model_root)
        if quant:
            result["quantization"] = quant

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def collect_run_context(
    *,
    model: str,
    backend: str,
    base_url: str,
    api_key: str | None = None,
    # Tier 2 — CLI parameters (caller fills these)
    temperature: float = 0.0,
    max_turns: int = 8,
    timeout_seconds: float = 60.0,
    seed: int | None = None,
    scenario_selector: str = "all",
    trials: int = 1,
    parallel: int = 1,
    error_rate: float = 0.0,
    thinking_enabled: bool = True,
    extra_params: dict[str, Any] | None = None,
    context_pressure: float | None = None,
    redact_url: bool = True,
    probe_engine: bool = True,
) -> RunContext:
    """Build a RunContext by combining local env, CLI params, and engine probes."""
    # Redact base_url for report storage (default: on for privacy)
    display_url = base_url
    if redact_url:
        from tool_eval_bench.utils.urls import redact_url as _redact
        display_url = _redact(base_url)

    # Tier 3: probe engine (best-effort, can be disabled)
    engine_info: dict[str, Any] = {}
    if probe_engine:
        try:
            engine_info = await _probe_engine(base_url, api_key, backend)
        except Exception as exc:
            logger.warning("Engine probe failed: %s", exc)

    return RunContext(
        # Tier 1
        tool_version=_tool_version(),
        git_sha=_git_sha(),
        hostname=socket.gethostname(),
        platform_info=platform.platform(),
        python_version=platform.python_version(),
        # Tier 2
        model=model,
        backend=backend,
        base_url=display_url,
        temperature=temperature,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        seed=seed,
        scenario_selector=scenario_selector,
        trials=trials,
        parallel=parallel,
        error_rate=error_rate,
        thinking_enabled=thinking_enabled,
        extra_params=extra_params,
        context_pressure=context_pressure,
        # Tier 3
        server_model_id=engine_info.get("server_model_id"),
        server_model_root=engine_info.get("server_model_root"),
        engine_name=engine_info.get("engine_name"),
        engine_version=engine_info.get("engine_version"),
        max_model_len=engine_info.get("max_model_len"),
        quantization=engine_info.get("quantization"),
        gpu_count=engine_info.get("gpu_count"),
        spec_decoding=engine_info.get("spec_decoding"),
    )


# -- Legacy API (kept for backward compatibility) --

async def collect_run_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    """Collect run metadata (legacy interface).

    New code should use collect_run_context() instead.
    """
    return {
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "config": {
            "model": config.model,
            "backend": config.backend,
            "base_url": config.base_url,
        },
        "backend_probe": await _probe_models(config.base_url, config.api_key),
    }
