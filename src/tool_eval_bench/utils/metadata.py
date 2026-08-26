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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from tool_eval_bench.domain.models import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    BenchmarkConfig,
    RunContext,
)
from tool_eval_bench.utils.urls import metrics_url as _metrics_url
from tool_eval_bench.utils.urls import models_url as _models_url
from tool_eval_bench.utils.urls import root_url as _root_url

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT = 5  # seconds — tight timeout for engine probes

# Git exports repository-local variables while hooks run.  Those variables
# override both the subprocess working directory and ``git -C``, so inheriting
# them can make a nested ``git init`` mutate this checkout or make provenance
# resolve against the hook's repository instead of the installed package.
_GIT_REPOSITORY_ENV_VARS = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)


def _git_env_without_repository() -> dict[str, str]:
    """Return the process environment without an inherited Git repository."""
    env = os.environ.copy()
    for name in _GIT_REPOSITORY_ENV_VARS:
        env.pop(name, None)
    return env


# ---------------------------------------------------------------------------
# Tier 1: local environment
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Return the commit of *this package's* checkout, or None.

    Deliberately anchored to the installed package directory rather than the
    current working directory.  Running ``git rev-parse`` in the CWD reported the
    SHA of whatever unrelated repository the user happened to be standing in,
    which is worse than reporting nothing: the run claimed a provenance it never
    had.  Installed wheels have no git metadata, so they legitimately return None
    and rely on the setuptools-scm version string instead.

    A ``-dirty`` suffix is included when the working tree has uncommitted
    changes, because such a run is not reproducible from the SHA alone.
    """
    package_root = Path(__file__).resolve().parent.parent

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.check_output(  # noqa: S603 — args are module constants
                ["git", "-C", str(package_root), *args],
                stderr=subprocess.DEVNULL,
                env=_git_env_without_repository(),
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.decode().strip()

    if _git("rev-parse", "--is-inside-work-tree") != "true":
        logger.debug("Package at %s is not a git checkout", package_root)
        return None
    sha = _git("rev-parse", "--short", "HEAD")
    if not sha:
        return None
    if _git("status", "--porcelain"):
        return f"{sha}-dirty"
    return sha


def _tool_version() -> str:
    from tool_eval_bench import __version__

    return __version__


# ---------------------------------------------------------------------------
# Tier 3: inference engine probing (best-effort)
# ---------------------------------------------------------------------------


class _ProbeSession:
    """One connection pool shared across an endpoint's probes.

    Two problems, one object.  Each probe used to build its own
    ``AsyncClient``, so identifying a server cost six TCP and TLS handshakes to
    the same host.  And the ladder is sequential by design, so an endpoint that
    is simply not there used to burn ``_PROBE_TIMEOUT`` once per rung before
    the run could start.  ``unreachable`` latches on the first connect-level
    failure and every later probe returns immediately.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.unreachable = False


@asynccontextmanager
async def _probe_session(session: _ProbeSession | None) -> AsyncIterator[_ProbeSession]:
    """Yield *session*, or open a short-lived one for a standalone probe."""
    if session is not None:
        yield session
        return
    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
        yield _ProbeSession(client)


async def _probe_get(
    session: _ProbeSession, url: str, *, headers: dict[str, str], what: str
) -> Any | None:
    """GET *url*, returning ``None`` rather than raising on any failure.

    A refusal or a 404 says something about the server and leaves the session
    usable.  A connect failure says the host is not answering at all, so it
    latches ``unreachable`` and short-circuits the rest of the ladder.
    """
    if session.unreachable:
        return None
    try:
        return await session.client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as exc:
        session.unreachable = True
        logger.debug("%s probe failed, endpoint unreachable: %s", what, exc)
    except httpx.HTTPError as exc:
        logger.debug("%s probe failed: %s", what, exc)
    return None


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Return probe headers, adding bearer auth only when a key is present."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def _probe_models(
    base_url: str,
    api_key: str | None,
    *,
    session: _ProbeSession | None = None,
) -> dict[str, Any]:
    """Probe /v1/models for model metadata."""
    probe: dict[str, Any] = {}
    async with _probe_session(session) as active:
        resp = await _probe_get(
            active, _models_url(base_url), headers=_auth_headers(api_key), what="models"
        )
    if resp is None or resp.status_code != 200:
        return probe
    try:
        body = resp.json()
    except ValueError as exc:
        logger.debug("models probe returned non-JSON: %s", exc)
        return probe
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list) and data:
        first = data[0] if isinstance(data[0], dict) else {}
        probe["server_model_id"] = first.get("id")
        probe["server_model_root"] = first.get("root")
        probe["owned_by"] = first.get("owned_by")  # NInfer fingerprints here
        # vLLM exposes max_model_len in model metadata
        if "max_model_len" in first:
            probe["max_model_len"] = first["max_model_len"]
    return probe


async def _probe_vllm_version(
    base_url: str, api_key: str | None, *, session: _ProbeSession | None = None
) -> dict[str, Any]:
    """Probe /version (vLLM-specific endpoint)."""
    async with _probe_session(session) as active:
        resp = await _probe_get(
            active,
            f"{_root_url(base_url)}/version",
            headers=_auth_headers(api_key),
            what="vLLM /version",
        )
    if resp is None or resp.status_code != 200:
        return {}
    try:
        body = resp.json()
    except ValueError:
        return {}
    if isinstance(body, dict) and "version" in body:
        return {"engine_name": "vLLM", "engine_version": body["version"]}
    return {}


async def _probe_llamacpp(base_url: str, *, session: _ProbeSession | None = None) -> dict[str, Any]:
    """Probe /props or /health (llama.cpp endpoints)."""
    async with _probe_session(session) as active:
        for path in ("/props", "/health"):
            resp = await _probe_get(
                active, f"{_root_url(base_url)}{path}", headers={}, what=f"llama.cpp {path}"
            )
            if resp is None or resp.status_code != 200:
                continue
            try:
                body = resp.json()
            except ValueError:
                continue
            if not isinstance(body, dict):
                continue
            result: dict[str, Any] = {"engine_name": "llama.cpp"}
            if "build_info" in body:
                result["engine_version"] = str(body["build_info"])
            elif "build_number" in body:
                result["engine_version"] = f"b{body['build_number']}"
            if "total_slots" in body:
                result["gpu_count"] = body.get("total_slots")
            return result
    return {}


async def _probe_litellm(
    base_url: str, api_key: str | None, *, session: _ProbeSession | None = None
) -> dict[str, Any]:
    """Detect LiteLLM from response headers or /health."""
    async with _probe_session(session) as active:
        resp = await _probe_get(
            active,
            f"{_root_url(base_url)}/health",
            headers=_auth_headers(api_key),
            what="LiteLLM /health",
        )
    if resp is None:
        return {}
    # LiteLLM sets x-litellm-version header
    version = resp.headers.get("x-litellm-version")
    if version:
        return {"engine_name": "LiteLLM", "engine_version": version}
    if resp.status_code != 200:
        return {}
    try:
        body = resp.json()
    except ValueError:
        return {}
    if isinstance(body, dict) and "litellm_version" in body:
        return {"engine_name": "LiteLLM", "engine_version": body["litellm_version"]}
    return {}


# Each engine namespaces its own Prometheus metrics, which turns out to be a
# far more reliable fingerprint than HTTP headers: vLLM's OpenAI server runs on
# uvicorn and doesn't set an identifying ``Server`` header, and llama.cpp's
# cpp-httplib server doesn't either. Order matters only in that the first
# matching non-comment line wins.
_METRICS_BACKEND_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("vllm:", "vllm", "vLLM"),
    ("llamacpp:", "llamacpp", "llama.cpp"),
    ("sglang:", "sglang", "SGLang"),
    ("sglang_", "sglang", "SGLang"),  # SGLang >=0.5.4 renamed the metric prefix
)


def detect_backend_from_metrics(text: str) -> tuple[str, str] | None:
    """Identify the backend from its Prometheus ``/metrics`` namespace.

    Returns ``(backend, label)`` for the first recognized metric-name prefix,
    or ``None`` if the text doesn't match a known engine.
    """
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        for prefix, backend, label in _METRICS_BACKEND_PREFIXES:
            if line.startswith(prefix):
                return backend, label
    return None


async def probe_backend_hint(base_url: str, api_key: str | None = None) -> tuple[str, str] | None:
    """Best-effort identification of vllm/llamacpp/sglang for an arbitrary server.

    Tried in order of specificity, so a generic signal can never outvote a
    distinctive one:

    1. The ``/metrics`` namespace — see :func:`detect_backend_from_metrics`.
       Unambiguous, but llama.cpp's ``--metrics`` flag is opt-in.
    2. vLLM's ``/version`` — llama.cpp 404s this, so it cannot false-positive.
    3. NInfer's ``owned_by`` marker from ``/v1/models``.
    4. llama.cpp's ``/props``/``/health``.  Deliberately last: ``/health``
       is generic enough that other engines answer it, and vLLM only avoids
       matching here because its ``/health`` body is empty.

    All four share one connection pool, and the ladder stops early once a
    probe proves the endpoint unreachable rather than spending the timeout
    again on each remaining rung.  Best-effort throughout; returns ``None`` if
    nothing matched.
    """
    async with _probe_session(None) as active:
        headers = _auth_headers(api_key)
        resp = await _probe_get(active, _metrics_url(base_url), headers=headers, what="/metrics")
        if resp is not None and resp.status_code == 200:
            hit = detect_backend_from_metrics(resp.text)
            if hit:
                return hit

        if await _probe_vllm_version(base_url, api_key, session=active):
            return "vllm", "vLLM"

        # NInfer: /v1/models entries carry owned_by == "ninfer".  Checked before
        # the generic /health fallback, which otherwise fingerprints NInfer as
        # llama.cpp (NInfer answers /health with 200 {"status":"ok"} and does not
        # serve /metrics, /version, or /props).
        if await _probe_ninfer(base_url, api_key, session=active):
            return "ninfer", "NInfer"

        if await _probe_llamacpp(base_url, session=active):
            return "llamacpp", "llama.cpp"

    return None


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
    for q in [
        "AWQ",
        "GPTQ",
        "GGUF",
        "EXL2",
        "BNBQ4",
        "BNB4",
        "INT8",
        "INT4",
        "FP8",
        "FP16",
        "BF16",
    ]:
        if q in upper:
            return q
    return None


async def _probe_ninfer(
    base_url: str, api_key: str | None, *, session: _ProbeSession | None = None
) -> dict[str, Any]:
    """Detect NInfer from /v1/models: entries carry owned_by == 'ninfer'.

    More specific than the generic /health probe that otherwise fingerprints
    NInfer as llama.cpp (NInfer answers /health with 200 {'status':'ok'} and
    does not serve /metrics, /version, or /props).
    """
    async with _probe_session(session) as active:
        resp = await _probe_get(
            active,
            f"{_root_url(base_url)}/v1/models",
            headers=_auth_headers(api_key),
            what="ninfer /v1/models",
        )
    if resp is None or resp.status_code != 200:
        return {}
    try:
        body = resp.json()
    except ValueError:
        return {}
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if data[0].get("owned_by") == "ninfer":
            return {"engine_name": "NInfer"}
    return {}


async def _probe_engine(
    base_url: str,
    api_key: str | None,
    backend: str,
) -> dict[str, Any]:
    """Run the engine probes for *backend* and merge results. Best-effort.

    Every probe shares one connection pool, and an endpoint that stops
    answering ends the sequence rather than costing a timeout per probe.
    """
    result: dict[str, Any] = {}
    backend_l = backend.lower()

    if backend_l == "gemini":
        # A hosted API serves no engine metadata, and its native endpoint does
        # not answer /v1/models at all, so every probe below would be a wasted
        # round trip against Google's servers.
        return {"engine_name": "Google Gemini API"}

    async with _probe_session(None) as active:
        # Always probe /v1/models (works for all self-hosted backends)
        result.update(await _probe_models(base_url, api_key, session=active))

        # Backend-specific probes
        if backend_l == "vllm":
            result.update(await _probe_vllm_version(base_url, api_key, session=active))
        elif backend_l in ("llamacpp", "llama.cpp", "llama_cpp"):
            result.update(await _probe_llamacpp(base_url, session=active))
        elif backend_l == "litellm":
            result.update(await _probe_litellm(base_url, api_key, session=active))
        elif backend_l == "sglang":
            # No well-documented, stable metadata endpoint for version info yet;
            # the metrics-based detector that produced this label already confirms
            # the engine, so just record the name.
            result.setdefault("engine_name", "SGLang")
        elif backend_l == "ninfer":
            if result.get("owned_by") == "ninfer":
                result["engine_name"] = "NInfer"
        else:
            # Try specific engine endpoints before model ownership and generic health.
            for prober in (
                lambda: _probe_vllm_version(base_url, api_key, session=active),
                lambda: _probe_litellm(base_url, api_key, session=active),
            ):
                info = await prober()
                if info:
                    result.update(info)
                    break
            else:
                if result.get("owned_by") == "ninfer":
                    result["engine_name"] = "NInfer"
                else:
                    result.update(await _probe_llamacpp(base_url, session=active))

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
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    seed: int | None = None,
    scenario_selector: str = "all",
    trials: int = 1,
    parallel: int = 1,
    error_rate: float = 0.0,
    thinking_enabled: bool = True,
    extra_params: dict[str, Any] | None = None,
    context_pressure: float | None = None,
    label: str | None = None,
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
        label=label,
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
    from tool_eval_bench.utils.urls import redact_url as _redact

    return {
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "config": {
            "model": config.model,
            "backend": config.backend,
            # Persisted and exported, so it must not carry the endpoint host or
            # any credentials embedded in the URL's userinfo.
            "base_url": _redact(config.base_url),
        },
        "backend_probe": await _probe_models(config.base_url, config.api_key),
    }
