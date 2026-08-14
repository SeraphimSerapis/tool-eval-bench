"""Which wire format an endpoint speaks, and how to work it out from a URL.

The benchmark speaks two request formats:

``openai``
    ``POST {base}/v1/chat/completions`` — vLLM, LiteLLM, llama.cpp, SGLang, and
    Google's own OpenAI compatibility layer.
``gemini``
    ``POST {base}/v1beta/models/{model}:generateContent`` — the native Gemini
    API described at https://ai.google.dev/api.

Detection is by endpoint host and path, which is unambiguous: Google serves the
compatibility layer under ``/v1beta/openai`` on the same host as the native API,
so the path is what separates them.  ``--format`` overrides the guess.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

WireFormat = Literal["openai", "gemini"]

WIRE_FORMATS: tuple[str, ...] = ("auto", "openai", "gemini")

# Hosts that serve the native Gemini API.  The OpenAI compatibility layer lives
# on the same hosts under an /openai path segment.
_GEMINI_HOSTS: frozenset[str] = frozenset({"generativelanguage.googleapis.com"})

# Version prefixes the native API is served under; a base URL missing one gets
# the default appended.
_GEMINI_API_VERSIONS: tuple[str, ...] = ("v1beta", "v1alpha", "v1")
DEFAULT_GEMINI_API_VERSION = "v1beta"


def detect_wire_format(base_url: str) -> WireFormat:
    """Infer the wire format an endpoint speaks from its URL."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host not in _GEMINI_HOSTS:
        return "openai"
    segments = [seg for seg in parsed.path.split("/") if seg]
    # ``…/v1beta/openai/v1`` is the OpenAI compatibility layer, not the native API.
    if "openai" in segments:
        return "openai"
    return "gemini"


def resolve_wire_format(requested: str | None, base_url: str) -> WireFormat:
    """Resolve an explicit ``--format`` choice, falling back to detection.

    Raises ValueError for an unknown format name.
    """
    choice = (requested or "auto").lower()
    if choice not in WIRE_FORMATS:
        raise ValueError(
            f"Unknown --format {requested!r}. Choose one of: {', '.join(WIRE_FORMATS)}"
        )
    if choice == "auto":
        return detect_wire_format(base_url)
    return "gemini" if choice == "gemini" else "openai"


def gemini_base_url(base_url: str) -> str:
    """Normalize a base URL to the versioned root of the native Gemini API.

    Accepts the bare host (``https://generativelanguage.googleapis.com``) as
    well as an already-versioned URL, and tolerates a trailing slash.
    """
    trimmed = base_url.rstrip("/")
    segments = [seg for seg in urlparse(trimmed).path.split("/") if seg]
    if segments and segments[-1] in _GEMINI_API_VERSIONS:
        return trimmed
    return f"{trimmed}/{DEFAULT_GEMINI_API_VERSION}"


def gemini_model_path(model: str) -> str:
    """Return the ``models/{id}`` path form the native API expects."""
    name = model.strip().lstrip("/")
    if name.startswith("models/") or name.startswith("tunedModels/"):
        return name
    return f"models/{name}"


def gemini_generate_url(base_url: str, model: str, *, stream: bool = False) -> str:
    """Build the generateContent / streamGenerateContent URL for a model."""
    method = "streamGenerateContent?alt=sse" if stream else "generateContent"
    return f"{gemini_base_url(base_url)}/{gemini_model_path(model)}:{method}"


def gemini_models_url(base_url: str) -> str:
    """Build the model-listing URL for the native API."""
    return f"{gemini_base_url(base_url)}/models"
