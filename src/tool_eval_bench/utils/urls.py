"""Shared URL helpers for OpenAI-compatible endpoints."""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlparse, urlsplit

from tool_eval_bench.domain.redaction import redact_url as redact_url
from tool_eval_bench.domain.redaction import redact_urls as redact_urls
from tool_eval_bench.utils.ids import build_config_fingerprint

# Only plain HTTP(S) makes sense for an inference endpoint.  Anything else
# (file://, gopher://, …) can only be an attempt to make the benchmark read
# something it shouldn't.
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def validate_http_url(url: str, *, what: str = "URL") -> str:
    """Return ``url`` if it is a well-formed http(s) URL, else raise ValueError."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(
            f"{what} must use http or https, got "
            f"{parsed.scheme or 'no scheme'!r}: {redact_url(url)}"
        )
    if not parsed.hostname:
        raise ValueError(f"{what} is missing a host: {url}")
    return url


def is_same_origin(a: str, b: str) -> bool:
    """Whether two URLs share scheme, host, and effective port.

    Used to decide whether an API key may travel with a request: a token issued
    for the inference endpoint must not be forwarded to an unrelated host just
    because the user passed a different ``--metrics-url``.
    """
    pa, pb = urlparse(a), urlparse(b)
    default_ports = {"http": 80, "https": 443}
    if not pa.hostname or not pb.hostname:
        return False
    try:
        port_a = pa.port or default_ports.get(pa.scheme)
        port_b = pb.port or default_ports.get(pb.scheme)
    except ValueError:  # malformed port
        return False
    return (pa.scheme, pa.hostname.lower(), port_a) == (pb.scheme, pb.hostname.lower(), port_b)


def _normalize_base(base_url: str) -> str:
    """Strip trailing slash; ensure URL ends with /v1."""
    b = base_url.rstrip("/")
    if not b.endswith("/v1"):
        b = f"{b}/v1"
    return b


def chat_completions_url(base_url: str) -> str:
    """Build the /v1/chat/completions URL from a base URL.

    Handles both ``http://host:port`` and ``http://host:port/v1`` forms.
    """
    return f"{_normalize_base(base_url)}/chat/completions"


def models_url(base_url: str) -> str:
    """Build the /v1/models URL from a base URL.

    Handles both ``http://host:port`` and ``http://host:port/v1`` forms.
    """
    return f"{_normalize_base(base_url)}/models"


def systemone_url(base_url: str) -> str:
    """Build the /v1/systemone URL that llama.cpp serves decision models on."""
    return f"{_normalize_base(base_url)}/systemone"


def root_url(base_url: str) -> str:
    """Strip a trailing ``/v1`` (and any trailing slash) from a base URL.

    Engine-identity endpoints — vLLM's ``/version``, llama.cpp's ``/props``
    and ``/health`` — live at the server root, *not* under ``/v1``.  Callers
    that naively append to ``base_url`` request ``/v1/props`` and get a 404
    whenever the user passes the ``http://host:port/v1`` form, silently
    losing engine version metadata.
    """
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b


def metrics_url(base_url: str) -> str:
    """Build the Prometheus /metrics URL from a base URL (not under /v1)."""
    return f"{root_url(base_url)}/metrics"


# Strata answers /metrics with JSON unless Accept names text/plain or
# OpenMetrics. prometheus_client (vLLM, SGLang, LiteLLM), llama.cpp, TensorFold
# and NInfer return 0.0.4 text whatever the request asks for. OpenMetrics and
# text/plain version=1.0.0 are left out because prometheus_client switches
# encoders for them. The */* fallback mirrors Prometheus's own scrape header,
# so a strict negotiator has no reason to answer 406.
PROMETHEUS_TEXT_ACCEPT = "text/plain; version=0.0.4, */*;q=0.1"


def metrics_request_target(
    base_url: str,
    explicit_metrics_url: str | None,
    api_key: str | None,
) -> tuple[str, dict[str, str]]:
    """Resolve the /metrics URL plus the headers to send with it.

    The headers always ask for Prometheus text, which is what every caller
    parses. ``--metrics-url`` exists because the metrics endpoint may live on a
    proxy or sidecar rather than behind the inference API.  That means it can
    point at a different host, and forwarding the inference endpoint's bearer
    token there would hand the credential to a third party.  The token is
    therefore attached only when the target is same-origin with ``base_url``.

    Raises ValueError for a non-http(s) or hostless ``--metrics-url``.
    """
    accept = {"Accept": PROMETHEUS_TEXT_ACCEPT}
    if explicit_metrics_url is None:
        return metrics_url(base_url), accept | _bearer(api_key)
    validate_http_url(explicit_metrics_url, what="--metrics-url")
    if api_key and not is_same_origin(explicit_metrics_url, base_url):
        return explicit_metrics_url, accept
    return explicit_metrics_url, accept | _bearer(api_key)


def _bearer(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def canonical_endpoint_path(path: str, *, wire_format: str = "openai") -> str:
    """Reduce a base URL, or just its path, to the root its requests are built from.

    The OpenAI and Anthropic request builders treat ``…``, ``…/`` and ``…/v1``
    as one base (``_normalize_base``, ``anthropic_base_url``), and the
    Anthropic one also accepts a pasted ``…/v1/messages``, so those spellings
    collapse to the root form. A Gemini base keeps its version segment: there a
    bare host means ``v1beta``, so ``/v1`` is a different API.
    """
    trimmed = path.rstrip("/")
    if wire_format == "gemini":
        return trimmed
    if wire_format == "anthropic" and trimmed.endswith("/messages"):
        trimmed = trimmed[: -len("/messages")]
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed


def endpoint_identity(url: str, *, wire_format: str = "openai") -> str:
    """Return an opaque, credential-independent endpoint identity.

    Every spelling of a base URL that sends requests to the same place gets the
    same identity; see :func:`canonical_endpoint_path`. ``wire_format`` defaults
    to the OpenAI format, which is what plugin, mode-run and judge endpoints speak.
    """
    return _endpoint_hash(url, lambda path: canonical_endpoint_path(path, wire_format=wire_format))


def legacy_endpoint_identity(url: str) -> str:
    """Return the identity runs stored before endpoint paths were canonical.

    It kept ``/v1`` in the path, so it differs from :func:`endpoint_identity`
    only for a URL that names it. Resume accepts either, because an identity is
    an opaque hash that cannot be recomputed from the stored, redacted URL.
    """
    return _endpoint_hash(url, lambda path: path.rstrip("/"))


def _endpoint_hash(url: str, canonical_path: Callable[[str], str]) -> str:
    parsed = urlsplit(url)
    if not parsed.hostname:
        return "endpoint:invalid"
    try:
        port = parsed.port
    except ValueError:
        port = None
    default_ports = {"http": 80, "https": 443}
    effective_port = port if port is not None else default_ports.get(parsed.scheme.lower())
    canonical = "|".join(
        (
            parsed.scheme.lower(),
            parsed.hostname.lower(),
            str(effective_port or ""),
            canonical_path(parsed.path) or "/",
        )
    )
    return f"endpoint:{build_config_fingerprint({'endpoint': canonical})}"
