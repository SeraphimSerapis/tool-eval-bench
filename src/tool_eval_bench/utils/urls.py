"""Shared URL helpers for OpenAI-compatible endpoints."""

from __future__ import annotations


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
