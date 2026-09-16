"""Minimal single-shot requests used by pre-flight checks and warm-up.

These run before the benchmark's adapter exists (and before any retry policy
applies), but they still have to speak whichever wire format the endpoint uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tool_eval_bench.adapters.anthropic import ANTHROPIC_VERSION, _apply_extra_params
from tool_eval_bench.adapters.gemini import _generation_config
from tool_eval_bench.adapters.wire_format import anthropic_messages_url, gemini_generate_url
from tool_eval_bench.utils.headers import USER_AGENT
from tool_eval_bench.utils.urls import chat_completions_url

_WARMUP_SYSTEM = "You are a helpful assistant."
_WARMUP_USER = "Say hello."


def minimal_request(
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    wire_format: str = "openai",
    temperature: float = 0.0,
    max_tokens: int = 1,
    extra_params: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Build (url, payload, headers) for a trivial completion request.

    *headers* are the user's extra headers; they are applied last so they can
    override anything the wire format sets.
    """
    url, payload, format_headers = _minimal_request(
        base_url,
        model,
        api_key,
        wire_format=wire_format,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_params=extra_params,
    )
    return url, payload, {"User-Agent": USER_AGENT, **format_headers, **(headers or {})}


def _minimal_request(
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    wire_format: str,
    temperature: float,
    max_tokens: int,
    extra_params: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if wire_format == "gemini":
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": _WARMUP_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": _WARMUP_USER}]}],
            "generationConfig": _generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=None,
                extra_params=extra_params,
            ),
        }
        if api_key:
            headers["x-goog-api-key"] = api_key
        return gemini_generate_url(base_url, model), payload, headers

    if wire_format == "anthropic":
        payload = {
            "model": model,
            "system": _WARMUP_SYSTEM,
            "messages": [{"role": "user", "content": _WARMUP_USER}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if api_key:
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        # The warm-up's ``enable_thinking=False`` hint would become disabled
        # thinking, which some models reject outright; a probe must not fail
        # on a knob it does not need.
        probe_params = {
            k: v for k, v in (extra_params or {}).items() if k != "chat_template_kwargs"
        }
        _apply_extra_params(payload, headers, probe_params)
        return anthropic_messages_url(base_url), payload, headers

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _WARMUP_SYSTEM},
            {"role": "user", "content": _WARMUP_USER},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_params:
        payload.update(extra_params)
        if "max_completion_tokens" in extra_params:
            payload.pop("max_tokens", None)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return chat_completions_url(base_url), payload, headers
