"""Run metadata collection for benchmark context (issue #6).

Builds a RunContext with three tiers of metadata:
  1. Local environment (always available)
  2. CLI parameters (passed in by caller)
  3. Inference engine probe (best-effort, HTTP calls with tight timeouts)
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import socket
import subprocess
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, get_args

import httpx

from tool_eval_bench.domain.engines import (
    ENGINE_PROFILES,
    HALOGEN,
    LITELLM,
    LLAMACPP,
    METRICS_IDENTITY_ORDER,
    NINFER,
    SGLANG,
    STRATA,
    TABBYAPI,
    TENSORFOLD,
    VLLM,
    IdentitySource,
    engine_profile,
)
from tool_eval_bench.domain.models import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    BenchmarkConfig,
    RunContext,
    default_max_tokens,
)
from tool_eval_bench.utils.urls import metrics_request_target
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

    Being inside *a* work tree is not enough: a wheel installed into a
    gitignored ``.venv`` of some other project sits inside that project's work
    tree too. The work tree's top level must be the checkout that holds this
    package under ``src/``, and its ``pyproject.toml`` must name this project.
    """
    package_root = Path(__file__).resolve().parent.parent
    # <checkout>/src/tool_eval_bench -> <checkout>
    source_root = package_root.parent.parent

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
    toplevel = _git("rev-parse", "--show-toplevel")
    if not toplevel or not _is_project_checkout(Path(toplevel), source_root):
        logger.debug("Package at %s is not tracked by its own checkout", package_root)
        return None
    sha = _git("rev-parse", "--short", "HEAD")
    if not sha:
        return None
    if _git("status", "--porcelain"):
        return f"{sha}-dirty"
    return sha


def _is_project_checkout(toplevel: Path, source_root: Path) -> bool:
    """Return whether *toplevel* is *source_root* and holds this project's pyproject."""
    try:
        if not os.path.samefile(toplevel, source_root):
            return False
        project = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    table = project.get("project")
    return isinstance(table, dict) and table.get("name") == "tool-eval-bench"


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
    failure, or on ``_TIMEOUTS_BEFORE_UNREACHABLE`` timeouts in a row, and
    every later probe returns immediately.

    Responses are also remembered, because identifying a server and then
    reading its metadata asks the same endpoints twice.  The cache key is the
    URL plus the credential sent, so a keyless refusal never answers a keyed
    request.
    """

    def __init__(self, client: Any, purpose: str = "metadata") -> None:
        self.client = client
        self.purpose = purpose
        self.unreachable = False
        self.consecutive_timeouts = 0
        self.responses: dict[tuple[str, str | None], httpx.Response | None] = {}


# A host that accepts the connection and then never answers fails every probe
# by timeout, so it would otherwise cost _PROBE_TIMEOUT per rung. One timeout
# alone is not that: llama-server answers /metrics through its task queue,
# which waits behind a decode step, while /props and /models answer at once.
# Two in a row with no answer between them is.
_TIMEOUTS_BEFORE_UNREACHABLE = 2


@asynccontextmanager
async def _probe_session(
    session: _ProbeSession | None, purpose: str = "metadata"
) -> AsyncIterator[_ProbeSession]:
    """Yield *session*, or open a short-lived one for a standalone probe."""
    if session is not None:
        yield session
        return
    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
        yield _ProbeSession(client, purpose)


async def _probe_get(
    session: _ProbeSession, url: str, *, headers: dict[str, str], what: str
) -> httpx.Response | None:
    """GET *url*, returning ``None`` rather than raising on any failure.

    A refusal or a 404 says something about the server and leaves the session
    usable.  A connect failure says the host is not answering at all, so it
    latches ``unreachable`` and short-circuits the rest of the ladder, and so
    do consecutive timeouts.  An answer already received stays valid after that.
    """
    key = (url, headers.get("Authorization"))
    if key in session.responses:
        return session.responses[key]
    if session.unreachable:
        return None
    resp = None
    try:
        # httpx's timeout bounds each read, not the response: a server that
        # sends headers and then trickles its body would never trip it.
        async with asyncio.timeout(_PROBE_TIMEOUT):
            resp = await session.client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        session.unreachable = True
        logger.debug("%s probe failed, endpoint unreachable: %s", what, exc)
    # Before OSError, which the builtin TimeoutError subclasses.
    except (httpx.TimeoutException, TimeoutError) as exc:
        session.consecutive_timeouts += 1
        logger.debug("%s probe timed out: %r", what, exc)
        if session.consecutive_timeouts >= _TIMEOUTS_BEFORE_UNREACHABLE:
            session.unreachable = True
            # INFO, not WARNING: outside --json the CLI configures no logging,
            # so a warning would print as bare text in the middle of Rich output.
            logger.info(
                "The server did not answer %d probes in a row within %s s; "
                "skipping the remaining %s probes",
                session.consecutive_timeouts,
                _PROBE_TIMEOUT,
                session.purpose,
            )
    except OSError as exc:
        session.unreachable = True
        logger.debug("%s probe failed, endpoint unreachable: %s", what, exc)
    except httpx.HTTPError as exc:
        logger.debug("%s probe failed: %s", what, exc)
    else:
        session.consecutive_timeouts = 0
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


def _listing_entry(data: list[Any], model: str | None) -> dict[str, Any] | None:
    """Return the ``/v1/models`` entry that describes *model*, or None.

    Multi-model servers (llama-swap, LiteLLM, Ollama, vLLM with LoRA modules)
    list several entries in an order that says nothing about the model under
    test, so the entry must match by ``id``. A single entry is trusted even when
    its id differs, because llama.cpp lists its one model under a file path
    rather than the name the client sends. Several entries and no match leave
    the facts unknown rather than borrowed from another model.
    """
    entries = [entry for entry in data if isinstance(entry, dict)]
    if model is not None:
        match = next((entry for entry in entries if entry.get("id") == model), None)
        if match is not None:
            return match
    if len(data) == 1 and entries:
        return entries[0]
    return None


async def _probe_models(
    base_url: str,
    api_key: str | None,
    *,
    model: str | None = None,
    session: _ProbeSession | None = None,
) -> dict[str, Any]:
    """Probe /v1/models for the metadata of *model*; see ``_listing_entry``."""
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
        # owned_by declares the engine (domain/engines.py), not the model, so
        # it keeps coming from the first entry.
        first = data[0] if isinstance(data[0], dict) else {}
        probe["owned_by"] = first.get("owned_by")
        entry = _listing_entry(data, model)
        if entry is not None:
            probe["server_model_id"] = entry.get("id")
            probe["server_model_root"] = entry.get("root")
            # vLLM exposes max_model_len in model metadata
            if "max_model_len" in entry:
                probe["max_model_len"] = entry["max_model_len"]
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
        return {"engine_name": VLLM.display_name, "engine_version": body["version"]}
    return {}


async def _probe_props(
    base_url: str, api_key: str | None = None, *, session: _ProbeSession | None = None
) -> dict[str, Any]:
    """Read llama.cpp build, capacity and file type from ``/props``, then ``/health``.

    ``default_generation_settings.n_ctx`` is llama-server's per-slot context,
    which spans the whole KV cache under ``--kv-unified``, capped by
    ``--kv-unified-per-slot`` and the model's training context. Either way it
    is the longest single request, which is what ``max_model_len`` means for vLLM.
    """
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
            if identity and identity != LLAMACPP.identity:
                continue
            has_build = isinstance(body.get("build_info"), str) and bool(body["build_info"])
            has_build_number = type(body.get("build_number")) is int
            has_props = (
                isinstance(body.get("default_generation_settings"), dict)
                and type(body.get("total_slots")) is int
                and body["total_slots"] > 0
            )
            if identity != LLAMACPP.identity and not (has_build or has_build_number or has_props):
                continue
            result: dict[str, Any] = {"engine_name": LLAMACPP.display_name}
            if "build_info" in body:
                result["engine_version"] = str(body["build_info"])
            elif "build_number" in body:
                result["engine_version"] = f"b{body['build_number']}"
            result.update(_props_capacity(body))
            quantization = _llamacpp_quantization(body.get("model_ftype"))
            if quantization:
                result["quantization"] = quantization
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
        return {"engine_name": LITELLM.display_name, "engine_version": version}
    if resp.status_code != 200:
        return {}
    try:
        body = resp.json()
    except ValueError:
        return {}
    if isinstance(body, dict) and "litellm_version" in body:
        return {"engine_name": LITELLM.display_name, "engine_version": body["litellm_version"]}
    return {}


def detect_backend_from_metrics(text: str) -> tuple[str, str] | None:
    """Identify the backend from its Prometheus ``/metrics`` namespace.

    Returns ``(backend, label)`` for the first namespace in
    ``METRICS_IDENTITY_ORDER`` that starts a sample line, or ``None`` if the
    text doesn't match a known engine.

    Strata's Prometheus format (serve/prometheus.py, chosen by Accept: text/plain
    or ?format=prometheus) is what carries its ``strata:`` namespace. The probe
    asks for that format; builds that predate it serve JSON, which matches
    nothing here and leaves identity to /health.
    """
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    for profile in METRICS_IDENTITY_ORDER:
        for prefix in profile.metrics_prefixes:
            if any(line.startswith(prefix) for line in lines):
                return profile.identity
    return None


# Names servers give themselves, keyed by where they give them, from each
# profile's declared_names. A lookup is an exact, case-insensitive match on the
# value's leading name token, so "Strata 0.1.40" names Strata and "tabby" names
# nothing. Declared identity is checked before llama.cpp's /props shape, which
# Strata and TabbyAPI imitate.
_DECLARED_IDENTITIES: dict[IdentitySource, dict[str, tuple[str, str]]] = {
    source: {
        token: profile.identity
        for profile in ENGINE_PROFILES
        for declared_source, token in profile.declared_names
        if declared_source == source
    }
    for source in get_args(IdentitySource)
}

# Endpoint-specific identity fields, as a key path into the JSON body. Every
# probed response is also checked for model ownership and its Server header.
_DECLARED_FIELDS: dict[str, tuple[IdentitySource, tuple[str, ...]]] = {
    "/health": ("service", ("service",)),
    "/.well-known/serviceinfo": ("software", ("software", "name")),
    "/props": ("build_info", ("build_info",)),
}


def _declared(source: IdentitySource, value: Any) -> tuple[str, str] | None:
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

    All probes share one connection pool and stop after a connect failure or
    consecutive timeouts. Returns ``None`` when the endpoint does not identify
    itself.
    """
    async with _probe_session(None, "backend detection") as active:
        url, headers = metrics_request_target(base_url, None, api_key)
        resp = await _probe_get(active, url, headers=headers, what="/metrics")
        if resp is not None and resp.status_code == 200:
            hit = detect_backend_from_metrics(resp.text)
            if hit:
                return hit

        if await _probe_vllm_version(base_url, api_key, session=active):
            return VLLM.identity

        identity = await _probe_declared_identity(base_url, api_key, session=active)
        if identity:
            return identity

        if await _probe_props(base_url, api_key, session=active):
            return LLAMACPP.identity

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

# llama_ftype_name() in llama.cpp's src/llama-model-loader.cpp, which /props
# reports as model_ftype, mapped to the labels _guess_quantization produces.
# Exact matches only. "(guessed) ..." names a type inferred from tensor counts,
# which turns an unrecognized tensor type into "all F32", and anything unlisted
# falls back to the model-name heuristic rather than being stored raw.
_LLAMACPP_FTYPES: dict[str, str] = {
    "all F32": "FP32",
    "F16": "FP16",
    "BF16": "BF16",
    "Q1_0": "Q1_0",
    "Q2_0": "Q2_0",
    "Q4_0": "Q4_0",
    "Q4_1": "Q4_1",
    "Q5_0": "Q5_0",
    "Q5_1": "Q5_1",
    "Q8_0": "Q8_0",
    "MXFP4 MoE": "MXFP4",
    "NVFP4": "NVFP4",
    # llama-quantize calls the medium Q2_K preset plain "Q2_K".
    "Q2_K - Medium": "Q2_K",
    "Q2_K - Small": "Q2_K_S",
    "Q3_K - Small": "Q3_K_S",
    "Q3_K - Medium": "Q3_K_M",
    "Q3_K - Large": "Q3_K_L",
    "Q4_K - Small": "Q4_K_S",
    "Q4_K - Medium": "Q4_K_M",
    "Q5_K - Small": "Q5_K_S",
    "Q5_K - Medium": "Q5_K_M",
    "Q6_K": "Q6_K",
    "TQ1_0 - 1.69 bpw ternary": "TQ1_0",
    "TQ2_0 - 2.06 bpw ternary": "TQ2_0",
    "IQ2_XXS - 2.0625 bpw": "IQ2_XXS",
    "IQ2_XS - 2.3125 bpw": "IQ2_XS",
    "IQ2_S - 2.5 bpw": "IQ2_S",
    "IQ2_M - 2.7 bpw": "IQ2_M",
    "IQ3_XS - 3.3 bpw": "IQ3_XS",
    "IQ3_XXS - 3.0625 bpw": "IQ3_XXS",
    "IQ1_S - 1.5625 bpw": "IQ1_S",
    "IQ1_M - 1.75 bpw": "IQ1_M",
    "IQ4_NL - 4.5 bpw": "IQ4_NL",
    "IQ4_XS - 4.25 bpw": "IQ4_XS",
    "IQ3_S - 3.4375 bpw": "IQ3_S",
    # LLAMA_FTYPE_MOSTLY_IQ3_M, despite the name.
    "IQ3_S mix - 3.66 bpw": "IQ3_M",
}


def _llamacpp_quantization(ftype: Any) -> str | None:
    """Return the label for a llama.cpp ``model_ftype``, or None if it is not a known name."""
    return _LLAMACPP_FTYPES.get(ftype) if isinstance(ftype, str) else None


async def _probe_tensorfold(
    base_url: str, api_key: str | None, *, session: _ProbeSession
) -> dict[str, Any]:
    """Collect declared capacity, not CUDA counters or MLX batch widths."""
    result: dict[str, Any] = {"engine_name": TENSORFOLD.display_name}
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
    result: dict[str, Any] = {"engine_name": STRATA.display_name}
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
    """Read TabbyAPI's loaded model from ``/v1/model`` and its capacity from ``/props``.

    ``/v1/models`` lists the whole model directory for admin access, which every
    request gets when authentication is disabled, and puts dummy aliases such as
    ``gpt-3.5-turbo`` first when those are enabled. Its first entry is therefore
    not necessarily the loaded model, so the loaded card's ``id`` replaces it.

    TabbyAPI reports neither its own version nor its model backend's, so
    ``engine_version`` stays unknown rather than borrowed from elsewhere.
    """
    result: dict[str, Any] = {"engine_name": TABBYAPI.display_name}
    root = _root_url(base_url)
    headers = _auth_headers(api_key)
    loaded = await _probe_json(
        session, f"{root}/v1/model", headers=headers, what="TabbyAPI /v1/model"
    )
    model_id = loaded.get("id") if loaded else None
    if isinstance(model_id, str) and model_id.strip():
        result["server_model_id"] = model_id
        # Any root came from the same list entry the loaded ID replaces.
        result["server_model_root"] = None
    props = await _probe_json(session, f"{root}/props", headers=headers, what="TabbyAPI /props")
    if props:
        result.update(_props_capacity(props))
    return result


async def _identify_unlabelled(
    base_url: str, api_key: str | None, *, session: _ProbeSession
) -> tuple[str, str | None]:
    """Choose the metadata probe for a server the caller did not label.

    Returns ``(backend, engine_name)``. The name is None only for the llama.cpp
    fallback, which nothing declared and only ``/props`` can confirm.
    """
    if await _probe_vllm_version(base_url, api_key, session=session):
        return VLLM.identity
    if await _probe_litellm(base_url, api_key, session=session):
        return LITELLM.identity
    identity = await _probe_declared_identity(base_url, api_key, session=session)
    # With no declared identity, only llama.cpp's characteristic /props shape is left.
    return identity if identity else (LLAMACPP.key, None)


async def _probe_engine(
    base_url: str,
    api_key: str | None,
    backend: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the engine probes for *backend* and merge results. Best-effort.

    Every probe shares one connection pool, and an endpoint that stops
    answering ends the sequence rather than costing a timeout per probe.
    *model* selects which ``/v1/models`` entry describes the run.
    """
    result: dict[str, Any] = {}
    profile = engine_profile(backend)

    if profile is not None and profile.hosted:
        # A hosted API serves no engine metadata, and Gemini's native endpoint
        # does not answer /v1/models at all, so every probe below would be a
        # wasted round trip against the vendor's servers.
        return {"engine_name": profile.display_name}

    async with _probe_session(None, "engine metadata") as active:
        # Always probe /v1/models (works for all self-hosted backends)
        result.update(await _probe_models(base_url, api_key, model=model, session=active))

        identified_name: str | None = None
        if str(result.get("owned_by", "")).lower() == TENSORFOLD.declared_name("owned_by"):
            key = TENSORFOLD.key
        elif profile is None:
            key, identified_name = await _identify_unlabelled(base_url, api_key, session=active)
        else:
            key = profile.key

        # Backend-specific probes. Responses are memoized, so a probe that
        # _identify_unlabelled already ran costs no second request.
        if key == TENSORFOLD.key:
            result.update(await _probe_tensorfold(base_url, api_key, session=active))
        elif key == VLLM.key:
            result.update(await _probe_vllm_version(base_url, api_key, session=active))
        elif key == STRATA.key:
            result.update(await _probe_strata(base_url, api_key, session=active))
        elif key == TABBYAPI.key:
            result.update(await _probe_tabbyapi(base_url, api_key, session=active))
        elif key == LLAMACPP.key:
            props = await _probe_props(base_url, api_key, session=active)
            # A GGUF file type cannot tell UD-Q4_K_XL or Q4_K_L from Q4_K_M, so
            # it only fills in for a name that identifies no specific type.
            name = result.get("server_model_root") or result.get("server_model_id")
            if _guess_quantization(name) not in (None, "GGUF"):
                props.pop("quantization", None)
            result.update(props)
        elif key == HALOGEN.key:
            result["engine_name"] = HALOGEN.display_name
        elif key == LITELLM.key:
            result.update(await _probe_litellm(base_url, api_key, session=active))
        elif key == SGLANG.key:
            # No well-documented, stable metadata endpoint for version info yet;
            # the metrics-based detector that produced this label already confirms
            # the engine, so just record the name.
            result.setdefault("engine_name", SGLANG.display_name)
        elif key == NINFER.key:
            if result.get("owned_by") == "ninfer":
                result["engine_name"] = NINFER.display_name

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
            engine_info = await _probe_engine(base_url, api_key, backend, model=model)
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
        "backend_probe": await _probe_models(config.base_url, config.api_key, model=config.model),
    }
