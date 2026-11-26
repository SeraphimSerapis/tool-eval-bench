"""Adapter selection: pick the wire format an endpoint speaks."""

from __future__ import annotations

from typing import Any

from tool_eval_bench.adapters.anthropic import AnthropicAdapter
from tool_eval_bench.adapters.gemini import GeminiAdapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.adapters.systemone import SystemOneAdapter
from tool_eval_bench.adapters.wire_format import WireFormat, resolve_wire_format
from tool_eval_bench.domain.adapters import BackendAdapter


def build_adapter(
    base_url: str = "",
    *,
    wire_format: str | None = None,
    max_concurrency: int | None = None,
    **kwargs: Any,
) -> BackendAdapter:
    """Return the adapter for ``base_url``, honoring an explicit format choice.

    ``wire_format`` accepts ``auto`` (or None) to detect from the URL, ``openai``,
    ``gemini``, or ``anthropic``.  ``max_concurrency`` sizes the underlying HTTP
    connection pool when given.  Extra keyword arguments are passed to the
    adapter.
    """
    if max_concurrency is not None:
        kwargs["max_concurrency"] = max_concurrency
    resolved: WireFormat = resolve_wire_format(wire_format, base_url)
    if resolved == "gemini":
        return GeminiAdapter(**kwargs)
    if resolved == "anthropic":
        return AnthropicAdapter(**kwargs)
    return OpenAICompatibleAdapter(**kwargs)


def build_decision_adapter(
    *,
    wire_format: str | None = None,
    max_concurrency: int | None = None,
    **kwargs: Any,
) -> SystemOneAdapter:
    """Return the adapter for a decision-model endpoint.

    Decision models are served by llama.cpp at ``/v1/systemone``.  There is no
    wire-format choice to make, so ``wire_format`` is accepted only so callers
    can pass the same options they pass to :func:`build_adapter`, and is ignored.
    """
    if max_concurrency is not None:
        kwargs["max_concurrency"] = max_concurrency
    return SystemOneAdapter(**kwargs)
