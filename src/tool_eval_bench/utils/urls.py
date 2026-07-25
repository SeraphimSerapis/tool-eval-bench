"""Shared URL helpers for OpenAI-compatible endpoints."""

from __future__ import annotations

from urllib.parse import urlparse

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


def metrics_url(base_url: str) -> str:
    """Build the Prometheus /metrics URL from a base URL (not under /v1)."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return f"{b}/metrics"


def metrics_request_target(
    base_url: str,
    explicit_metrics_url: str | None,
    api_key: str | None,
) -> tuple[str, dict[str, str]]:
    """Resolve the /metrics URL plus the headers that are safe to send with it.

    ``--metrics-url`` exists because the metrics endpoint may live on a proxy or
    sidecar rather than behind the inference API.  That means it can point at a
    different host, and forwarding the inference endpoint's bearer token there
    would hand the credential to a third party.  The token is therefore attached
    only when the target is same-origin with ``base_url``.

    Raises ValueError for a non-http(s) or hostless ``--metrics-url``.
    """
    if explicit_metrics_url is None:
        return metrics_url(base_url), _bearer(api_key)
    validate_http_url(explicit_metrics_url, what="--metrics-url")
    if api_key and not is_same_origin(explicit_metrics_url, base_url):
        return explicit_metrics_url, {}
    return explicit_metrics_url, _bearer(api_key)


def _bearer(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def redact_url(url: str) -> str:
    """Mask the host in a URL for display.

    e.g. http://192.168.10.5:8080 → http://***:8080
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    # Replace hostname with ***, keep port if present
    redacted_netloc = "***"
    if parsed.port:
        redacted_netloc = f"***:{parsed.port}"
    return urlunparse(parsed._replace(netloc=redacted_netloc))
