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
from typing import Any, Literal

import httpx

from tool_eval_bench.domain.models import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    BenchmarkConfig,
    RunContext,
    default_max_tokens,
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

    Responses are also remembered, because identifying a server and then
    reading its metadata asks the same endpoints twice.  The cache key is the
    URL plus the credential sent, so a keyless refusal never answers a keyed
    request.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.unreachable = False
        self.responses: dict[tuple[str, str | None], httpx.Response | None] = {}


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
) -> httpx.Response | None:
    """GET *url*, returning ``None`` rather than raising on any failure.

    A refusal or a 404 says something about the server and leaves the session
    usable.  A connect failure says the host is not answering at all, so it
    latches ``unreachable`` and short-circuits the rest of the ladder.  An
    answer already received stays valid after that.
    """
    key = (url, headers.get("Authorization"))
    if key in session.responses:
        return session.responses[key]
    if session.unreachable:
        return None
    resp = None
    try:
        resp = await session.client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as exc:
        session.unreachable = True
        logger.debug("%s probe failed, endpoint unreachable: %s", what, exc)
    except httpx.HTTPError as exc:
        logger.debug("%s probe failed: %s", what, exc)
    session.responses[key] = resp
    return resp


async def _probe_json(
    session: _ProbeSession, url: str, *, headers: dict[str, str], what: str
) -> dict[str, Any] | None:
    """GET *url* and return its body only when it is a 200 JSON object."""
    resp = await _probe_get(session, url, headers=headers, what=what)
    if resp is None or resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _positive_int(value: Any) -> int | None:
    """Return *value* when it is a real positive int; bools and strings do not count."""
    return value if type(value) is int and value > 0 else None


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
    identity = backend_from_response(resp)
    if identity:
        probe["engine_name"] = identity[1]
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list) and data:
        first = data[0] if isinstance(data[0], dict) else {}
        probe["server_model_id"] = first.get("id")
        probe["server_model_root"] = first.get("root")
        probe["owned_by"] = first.get("owned_by")
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


async def _probe_props(
    base_url: str, api_key: str | None = None, *, session: _ProbeSession | None = None
) -> dict[str, Any]:
    """Read llama.cpp build and slot metadata from ``/props``, then ``/health``."""
    async with _probe_session(session) as active:
        for path in ("/props", "/health"):
            resp = await _probe_get(
                active, f"{_root_url(base_url)}{path}", headers=_auth_headers(api_key), what=path
            )
            if resp is None or resp.status_code != 200:
                continue
            try:
                body = resp.json()
            except ValueError:
                continue
            if not isinstance(body, dict):
                continue
            # Strata and TabbyAPI serve llama-server-shaped /props. A response
            # that names its server is believed over its shape, and a build_info
            # that names another server is never a llama.cpp build.
            identity = _endpoint_identity(path, resp) or _declared(
                "build_info", body.get("build_info")
            )
            if identity and identity != ("llamacpp", "llama.cpp"):
                continue
            has_build = isinstance(body.get("build_info"), str) and bool(body["build_info"])
            has_build_number = type(body.get("build_number")) is int
            has_props = (
                isinstance(body.get("default_generation_settings"), dict)
                and type(body.get("total_slots")) is int
                and body["total_slots"] > 0
            )
            if identity != ("llamacpp", "llama.cpp") and not (
                has_build or has_build_number or has_props
            ):
                continue
            result: dict[str, Any] = {"engine_name": "llama.cpp"}
            if "build_info" in body:
                result["engine_version"] = str(body["build_info"])
            elif "build_number" in body:
                result["engine_version"] = f"b{body['build_number']}"
            if "total_slots" in body:
                result["slot_count"] = body.get("total_slots")
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


# Prefer an engine's native namespace over compatibility aliases. Halogen
# deliberately exports llama.cpp metrics too; scrape order must not decide identity.
_METRICS_BACKEND_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("halogen:", "halogen", "Halogen Flash"),
    ("tensorfold:", "tensorfold", "TensorFold"),
    ("vllm:", "vllm", "vLLM"),
    ("sglang:", "sglang", "SGLang"),
    ("sglang_", "sglang", "SGLang"),  # SGLang >=0.5.4 renamed the metric prefix
    ("llamacpp:", "llamacpp", "llama.cpp"),
)


def detect_backend_from_metrics(text: str) -> tuple[str, str] | None:
    """Identify the backend from its Prometheus ``/metrics`` namespace.

    Returns ``(backend, label)`` for the most specific recognized namespace,
    or ``None`` if the text doesn't match a known engine.
    """
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    for prefix, backend, label in _METRICS_BACKEND_PREFIXES:
        if any(line.startswith(prefix) for line in lines):
            return backend, label
    return None


_IdentitySource = Literal["owned_by", "server", "service", "software", "build_info"]

_STRATA = ("strata", "Strata")
_TABBYAPI = ("tabbyapi", "TabbyAPI")

# Names servers give themselves, keyed by where they give them. A lookup is an
# exact, case-insensitive match on the value's leading name token, so
# "Strata 0.1.40" names Strata and "tabby" names nothing. Declared identity is
# checked before llama.cpp's /props shape, which Strata and TabbyAPI imitate.
_DECLARED_IDENTITIES: dict[_IdentitySource, dict[str, tuple[str, str]]] = {
    # /v1/models data[0].owned_by
    "owned_by": {
        "llamacpp": ("llamacpp", "llama.cpp"),
        "ninfer": ("ninfer", "NInfer"),
        "sglang": ("sglang", "SGLang"),
        "tabbyapi": _TABBYAPI,
        "tensorfold": ("tensorfold", "TensorFold"),
    },
    # HTTP Server header, product token before any "/version"
    "server": {
        "llama.cpp": ("llamacpp", "llama.cpp"),
        "vllm": ("vllm", "vLLM"),
        "sglang": ("sglang", "SGLang"),
        "litellm": ("litellm", "LiteLLM"),
    },
    # /health service
    "service": {"strata": _STRATA},
    # /.well-known/serviceinfo software.name
    "software": {"tabbyapi": _TABBYAPI},
    # /props build_info; llama.cpp's "b6123-abc1234" names no server
    "build_info": {"strata": _STRATA},
}

# Endpoint-specific identity fields, as a key path into the JSON body. Every
# probed response is also checked for model ownership and its Server header.
_DECLARED_FIELDS: dict[str, tuple[_IdentitySource, tuple[str, ...]]] = {
    "/health": ("service", ("service",)),
    "/.well-known/serviceinfo": ("software", ("software", "name")),
    "/props": ("build_info", ("build_info",)),
}


def _declared(source: _IdentitySource, value: Any) -> tuple[str, str] | None:
    """Look up the server *value* names; anything but a non-empty string names none."""
    if not isinstance(value, str):
        return None
    token = re.split(r"[\s/]", value.strip(), maxsplit=1)[0].lower()
    return _DECLARED_IDENTITIES[source].get(token) if token else None


def backend_from_models(body: Any) -> tuple[str, str] | None:
    """Read a distinctive model owner without guessing from its model ID."""
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return _declared("owned_by", data[0].get("owned_by"))
    return None


def backend_from_response(resp: Any) -> tuple[str, str] | None:
    """Read declared model ownership or an identifying Server header."""
    try:
        owner = backend_from_models(resp.json())
    except (AttributeError, ValueError):
        owner = None
    return owner or _declared("server", resp.headers.get("server"))


def _endpoint_identity(path: str, resp: Any) -> tuple[str, str] | None:
    """Return the identity a 200 response from *path* declares, if any."""
    if resp is None or resp.status_code != 200:
        return None
    declared = backend_from_response(resp)
    if declared or path not in _DECLARED_FIELDS:
        return declared
    source, keys = _DECLARED_FIELDS[path]
    try:
        value: Any = resp.json()
    except ValueError:
        return None
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return _declared(source, value)


async def _probe_declared_identity(
    base_url: str, api_key: str | None, *, session: _ProbeSession
) -> tuple[str, str] | None:
    """Ask each endpoint where a server may name itself, in a fixed order.

    ``/.well-known/serviceinfo`` and ``/health`` answer without a key on
    TabbyAPI and Strata, so identity survives a missing ``--api-key``.
    """
    root = _root_url(base_url)
    for path in ("/v1/models", *_DECLARED_FIELDS):
        resp = await _probe_get(
            session, f"{root}{path}", headers=_auth_headers(api_key), what=f"identity {path}"
        )
        identity = _endpoint_identity(path, resp)
        if identity:
            return identity
    return None


async def probe_backend_hint(base_url: str, api_key: str | None = None) -> tuple[str, str] | None:
    """Best-effort identification of an arbitrary inference server.

    Check native metrics namespaces, then vLLM's version endpoint, then the
    identity a server declares: model ownership, a Server header, the
    ``/health`` service, ``/.well-known/serviceinfo``, or a ``/props`` build
    name. Only then fall back to characteristic llama.cpp props/build fields,
    since other servers copy that shape. A generic health response is not an
    identity. Compatibility metrics yield to an engine's own namespace,
    regardless of scrape order.

    All probes share one connection pool and stop after a connect failure.
    Returns ``None`` when the endpoint does not identify itself.
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

        identity = await _probe_declared_identity(base_url, api_key, session=active)
        if identity:
            return identity

        if await _probe_props(base_url, api_key, session=active):
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
    gguf_match = _GGUF_K_QUANT.search(upper)
    if gguf_match:
        return gguf_match.group(1)
    native_gguf = _GGUF_NATIVE_TYPE.search(upper)
    if native_gguf:
        return native_gguf.group(1)
    mlx_match = re.search(r"MLX-(\d+)BIT", upper)
    if mlx_match:
        return f"MLX-{mlx_match.group(1)}bit"
    # Simple keyword match
    for q in [
        "AWQ",
        "GPTQ",
        "GGUF",
        "EXL3",
        "EXL2",
        "NVFP4",
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


# Non-K GGUF file types from llama.cpp's llama_ftype. The Q4_0_x_y ARM layouts
# were dropped from GGUF, but files with those names exist and are not Q4_0.
_GGUF_NATIVE_TYPES = (
    "Q1_0",
    "Q2_0",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q4_0_4_4",
    "Q4_0_4_8",
    "Q4_0_8_8",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_NL",
    "IQ4_XS",
    "TQ1_0",
    "TQ2_0",
    "MXFP4",
)
# Longest first, and never followed by "_<digit>", so an unknown suffixed
# layout returns None instead of being truncated to its base type.
_GGUF_NATIVE_TYPE = re.compile(
    r"(?<![A-Z0-9])("
    + "|".join(sorted(_GGUF_NATIVE_TYPES, key=len, reverse=True))
    + r")(?![A-Z0-9]|_\d)"
)
# K-quants with llama.cpp's S/M/L sizes and Unsloth's XL. The lookbehind keeps
# ik_llama.cpp's IQ4_K, a different type, from reading as Q4_K.
_GGUF_K_QUANT = re.compile(r"(?<![A-Z0-9])(Q\d+_K(?:_(?:XL|[SML]))?)")


# Backend labels that name a vendor API rather than a self-hosted engine.
_HOSTED_ENGINE_NAMES: dict[str, str] = {
    "gemini": "Google Gemini API",
    "openai": "OpenAI API",
    "anthropic": "Anthropic Messages API",
}


async def _probe_tensorfold(
    base_url: str, api_key: str | None, *, session: _ProbeSession
) -> dict[str, Any]:
    """Collect declared capacity, not CUDA counters or MLX batch widths."""
    result: dict[str, Any] = {"engine_name": "TensorFold"}
    resp = await _probe_get(
        session,
        f"{_root_url(base_url)}/health",
        headers=_auth_headers(api_key),
        what="TensorFold /health",
    )
    if resp is None or resp.status_code != 200:
        return result
    try:
        body = resp.json()
    except ValueError:
        return result
    if not isinstance(body, dict):
        return result
    window = body.get("context_length")
    if type(window) is int and window > 0:
        result["max_model_len"] = window
    streams = body.get("streams")
    slots = streams.get("max") if isinstance(streams, dict) else None
    if type(slots) is int and slots > 0:
        result["slot_count"] = slots
    return result


def _props_capacity(props: dict[str, Any]) -> dict[str, Any]:
    """Read the context window and slot count from a llama-server-shaped /props."""
    result: dict[str, Any] = {}
    settings = props.get("default_generation_settings")
    window = _positive_int(settings.get("n_ctx")) if isinstance(settings, dict) else None
    if window:
        result["max_model_len"] = window
    slots = _positive_int(props.get("total_slots"))
    if slots:
        result["slot_count"] = slots
    return result


async def _probe_strata(
    base_url: str, api_key: str | None, *, session: _ProbeSession
) -> dict[str, Any]:
    """Read Strata's version and capacity; ``/props`` carries the version only when known."""
    result: dict[str, Any] = {"engine_name": "Strata"}
    root = _root_url(base_url)
    headers = _auth_headers(api_key)
    props = await _probe_json(session, f"{root}/props", headers=headers, what="Strata /props")
    if props:
        build = props.get("build_info")
        parts = build.split(maxsplit=1) if isinstance(build, str) else []
        if len(parts) == 2 and parts[0].lower() == "strata":
            result["engine_version"] = parts[1].strip()
        result.update(_props_capacity(props))
    if "max_model_len" not in result:
        health = await _probe_json(
            session, f"{root}/health", headers=headers, what="Strata /health"
        )
        window = _positive_int(health.get("max_context")) if health else None
        if window:
            result["max_model_len"] = window
    return result


async def _probe_tabbyapi(
    base_url: str, api_key: str | None, *, session: _ProbeSession
) -> dict[str, Any]:
    """Read TabbyAPI's capacity from ``/props``.

    TabbyAPI reports neither its own version nor its model backend's, so
    ``engine_version`` stays unknown rather than borrowed from elsewhere.
    """
    result: dict[str, Any] = {"engine_name": "TabbyAPI"}
    props = await _probe_json(
        session,
        f"{_root_url(base_url)}/props",
        headers=_auth_headers(api_key),
        what="TabbyAPI /props",
    )
    if props:
        result.update(_props_capacity(props))
    return result


_LLAMACPP_LABELS = ("llamacpp", "llama.cpp", "llama_cpp")
# Labels with a dedicated branch in _probe_engine; anything else is unlabelled.
_PROBED_LABELS = frozenset(
    {"tensorfold", "vllm", "strata", "tabbyapi", "halogen", "litellm", "sglang", "ninfer"}
    | set(_LLAMACPP_LABELS)
)


async def _identify_unlabelled(
    base_url: str, api_key: str | None, *, session: _ProbeSession
) -> tuple[str, str | None]:
    """Choose the metadata probe for a server the caller did not label.

    Returns ``(backend, engine_name)``. The name is None only for the llama.cpp
    fallback, which nothing declared and only ``/props`` can confirm.
    """
    if await _probe_vllm_version(base_url, api_key, session=session):
        return "vllm", "vLLM"
    if await _probe_litellm(base_url, api_key, session=session):
        return "litellm", "LiteLLM"
    identity = await _probe_declared_identity(base_url, api_key, session=session)
    # With no declared identity, only llama.cpp's characteristic /props shape is left.
    return identity if identity else ("llamacpp", None)


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

    if backend_l in _HOSTED_ENGINE_NAMES:
        # A hosted API serves no engine metadata, and Gemini's native endpoint
        # does not answer /v1/models at all, so every probe below would be a
        # wasted round trip against the vendor's servers.
        return {"engine_name": _HOSTED_ENGINE_NAMES[backend_l]}

    async with _probe_session(None) as active:
        # Always probe /v1/models (works for all self-hosted backends)
        result.update(await _probe_models(base_url, api_key, session=active))

        identified_name: str | None = None
        if str(result.get("owned_by", "")).lower() == "tensorfold":
            backend_l = "tensorfold"
        elif backend_l not in _PROBED_LABELS:
            backend_l, identified_name = await _identify_unlabelled(
                base_url, api_key, session=active
            )

        # Backend-specific probes. Responses are memoized, so a probe that
        # _identify_unlabelled already ran costs no second request.
        if backend_l == "tensorfold":
            result.update(await _probe_tensorfold(base_url, api_key, session=active))
        elif backend_l == "vllm":
            result.update(await _probe_vllm_version(base_url, api_key, session=active))
        elif backend_l == "strata":
            result.update(await _probe_strata(base_url, api_key, session=active))
        elif backend_l == "tabbyapi":
            result.update(await _probe_tabbyapi(base_url, api_key, session=active))
        elif backend_l in _LLAMACPP_LABELS:
            result.update(await _probe_props(base_url, api_key, session=active))
        elif backend_l == "halogen":
            result["engine_name"] = "Halogen Flash"
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

        # A server identified only by a declaration, such as a Server header,
        # has nothing for its metadata probe to read. It still has a name.
        if identified_name:
            result.setdefault("engine_name", identified_name)

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
    system_prompt: str | None = None,
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
        max_tokens=default_max_tokens(extra_params),
        extra_params=extra_params,
        context_pressure=context_pressure,
        system_prompt=system_prompt,
        label=label,
        # Tier 3
        server_model_id=engine_info.get("server_model_id"),
        server_model_root=engine_info.get("server_model_root"),
        engine_name=engine_info.get("engine_name"),
        engine_version=engine_info.get("engine_version"),
        max_model_len=engine_info.get("max_model_len"),
        quantization=engine_info.get("quantization"),
        gpu_count=engine_info.get("gpu_count"),
        slot_count=engine_info.get("slot_count"),
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
