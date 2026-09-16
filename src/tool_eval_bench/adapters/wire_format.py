"""Which wire format an endpoint speaks, and how to work it out from a URL.

The benchmark speaks three request formats:

``openai``
    ``POST {base}/v1/chat/completions`` — vLLM, LiteLLM, llama.cpp, SGLang, and
    Google's own OpenAI compatibility layer.
``gemini``
    ``POST {base}/v1beta/models/{model}:generateContent`` — the native Gemini
    API described at https://ai.google.dev/api.
``anthropic``
    ``POST {base}/v1/messages`` — the Anthropic Messages API, and gateways that
    front other models with it (OpenCode Zen, LiteLLM, Bedrock Mantle).

Detection is by endpoint host and path.  Google serves the compatibility layer
under ``/v1beta/openai`` on the same host as the native API, and a gateway such
as OpenCode Zen serves ``/v1/chat/completions`` and ``/v1/messages`` side by
side, so the path is what separates them.  A base URL that names ``/messages``
is taken at its word; ``api.anthropic.com`` is native unless the path says
otherwise.  ``--format`` overrides the guess.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

WireFormat = Literal["openai", "gemini", "anthropic"]

WIRE_FORMATS: tuple[str, ...] = ("auto", "openai", "gemini", "anthropic")

# Hosts that serve the native Gemini API.  The OpenAI compatibility layer lives
# on the same hosts under an /openai path segment.
_GEMINI_HOSTS: frozenset[str] = frozenset({"generativelanguage.googleapis.com"})

# Version prefixes the native API is served under; a base URL missing one gets
# the default appended.
_GEMINI_API_VERSIONS: tuple[str, ...] = ("v1beta", "v1alpha", "v1")
DEFAULT_GEMINI_API_VERSION = "v1beta"

# Hosts that serve the Messages API natively.  Anthropic also serves an OpenAI
# compatibility layer at ``/v1/chat/completions`` on the same host; naming that
# path in the base URL selects it.
_ANTHROPIC_HOSTS: frozenset[str] = frozenset({"api.anthropic.com"})


def detect_wire_format(base_url: str) -> WireFormat:
    """Infer the wire format an endpoint speaks from its URL."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    segments = [seg for seg in parsed.path.split("/") if seg]
    if segments and segments[-1] == "messages":
        return "anthropic"
    if host in _ANTHROPIC_HOSTS:
        return "openai" if "chat" in segments else "anthropic"
    if host not in _GEMINI_HOSTS:
        return "openai"
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
    if choice == "gemini":
        return "gemini"
    return "anthropic" if choice == "anthropic" else "openai"


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


def anthropic_base_url(base_url: str) -> str:
    """Normalize a base URL to the ``…/v1`` root of a Messages API endpoint.

    Accepts the bare host, ``…/v1``, and a full ``…/v1/messages`` URL, and
    tolerates a trailing slash, so a pasted endpoint works as-is.
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/messages"):
        trimmed = trimmed[: -len("/messages")]
    if not trimmed.endswith("/v1"):
        trimmed = f"{trimmed}/v1"
    return trimmed


def anthropic_messages_url(base_url: str) -> str:
    """Build the ``/v1/messages`` URL from a base URL."""
    return f"{anthropic_base_url(base_url)}/messages"


def anthropic_models_url(base_url: str) -> str:
    """Build the model-listing URL; the Messages API lists under ``/v1/models``."""
    return f"{anthropic_base_url(base_url)}/models"
