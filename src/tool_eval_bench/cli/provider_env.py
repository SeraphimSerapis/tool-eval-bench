"""Provider-scoped endpoint settings from the environment.

One ``.env`` can hold several endpoints side by side so an A/B between, say,
Gemini and a local vLLM box is a flag change rather than an edit:

.. code-block:: bash

    TOOL_EVAL_GEMINI_BASE_URL=https://generativelanguage.googleapis.com
    TOOL_EVAL_GEMINI_API_KEY=...
    TOOL_EVAL_GEMINI_MODEL=gemini-2.5-pro

    TOOL_EVAL_LOCAL_BASE_URL=http://gpu-box:8080

``--provider gemini`` (or ``TOOL_EVAL_PROVIDER=gemini``) then reads the
``TOOL_EVAL_GEMINI_*`` triple.  The provider name is a free-form prefix, not
an enum: ``VLLM_A`` and ``VLLM_B`` are as valid as ``OPENAI``.  Explicit CLI
flags still win, and with no provider selected nothing here applies.

A gateway that wants extra headers keeps them next to its endpoint too:

.. code-block:: bash

    TOOL_EVAL_ZEN_HEADERS=User-Agent=my-agent/1.0
    TOOL_EVAL_ZEN_SESSION_HEADER=x-opencode-session
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from tool_eval_bench.utils.headers import parse_header_env

#: Provider names that name a hosted API rather than a self-hosted engine.
#: They double as the report's backend label, and engine probing is skipped
#: for them because ``/metrics`` and friends do not exist on a vendor API.
HOSTED_PROVIDERS: frozenset[str] = frozenset({"gemini", "openai", "anthropic"})

_PREFIX = "TOOL_EVAL_"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ProviderSettings:
    """The endpoint triple read from ``TOOL_EVAL_<PROVIDER>_*``."""

    name: str
    base_url: str
    api_key: str | None
    model: str | None
    headers: dict[str, str] = field(default_factory=dict)
    session_header: str | None = None

    @property
    def backend_label(self) -> str | None:
        """Report label implied by the provider, or None to leave detection alone."""
        return self.name if self.name in HOSTED_PROVIDERS else None


def provider_env_prefix(name: str) -> str:
    """Return the ``TOOL_EVAL_<NAME>_`` prefix for a provider name.

    Dots and dashes become underscores so ``llama.cpp`` and ``vllm-a`` are
    usable names.  Raises ValueError for anything that cannot form an env var.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid provider name {name!r}: use letters, digits, '_', '-' or '.', "
            "starting with a letter"
        )
    return _PREFIX + re.sub(r"[.-]", "_", name).upper() + "_"


def resolve_provider(name: str | None, env: Mapping[str, str]) -> ProviderSettings | None:
    """Read the provider's endpoint settings from *env*.

    Returns None when no provider is selected.  Raises ValueError when the
    provider has no ``_BASE_URL``, naming the variable that is missing, since a
    silent fall-through to localhost discovery would benchmark the wrong thing.
    """
    if not name:
        return None
    prefix = provider_env_prefix(name)
    base_url = env.get(prefix + "BASE_URL", "").strip()
    if not base_url:
        raise ValueError(
            f"Provider {name!r} selected but {prefix}BASE_URL is not set. "
            f"Add it to .env or the environment (optionally with {prefix}API_KEY "
            f"and {prefix}MODEL)."
        )
    return ProviderSettings(
        name=name.lower(),
        base_url=base_url,
        api_key=env.get(prefix + "API_KEY", "").strip() or None,
        model=env.get(prefix + "MODEL", "").strip() or None,
        headers=parse_header_env(env.get(prefix + "HEADERS")),
        session_header=env.get(prefix + "SESSION_HEADER", "").strip() or None,
    )
