"""Adapter selection: pick the wire format an endpoint speaks."""

from __future__ import annotations

from typing import Any

from tool_eval_bench.adapters.gemini import GeminiAdapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.adapters.wire_format import WireFormat, resolve_wire_format
from tool_eval_bench.domain.adapters import BackendAdapter


def build_adapter(
    base_url: str = "",
    *,
    wire_format: str | None = None,
    **kwargs: Any,
) -> BackendAdapter:
    """Return the adapter for ``base_url``, honoring an explicit format choice.

    ``wire_format`` accepts ``auto`` (or None) to detect from the URL, ``openai``,
    or ``gemini``.  Extra keyword arguments are passed to the adapter.
    """
    resolved: WireFormat = resolve_wire_format(wire_format, base_url)
    if resolved == "gemini":
        return GeminiAdapter(**kwargs)
    return OpenAICompatibleAdapter(**kwargs)
