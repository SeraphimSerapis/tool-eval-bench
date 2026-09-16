"""Small compatibility helpers for OpenAI-style request payloads."""

from __future__ import annotations

from typing import Any


def output_token_limit_reached(status_code: int, response_text: str) -> bool:
    """Return whether a valid request exhausted its output-token allowance.

    Some hosted reasoning endpoints report this as HTTP 400 instead of a
    successful response with ``finish_reason=length``. Reaching generation is
    sufficient evidence for availability checks and model warm-up.
    """
    if status_code not in (400, 422):
        return False
    text = response_text.lower()
    mentions_output_budget = any(
        marker in text for marker in ("max_tokens", "max_completion_tokens", "model output limit")
    )
    reports_exhaustion = any(
        marker in text
        for marker in ("limit was reached", "limit reached", "try again with higher max_")
    )
    return mentions_output_budget and reports_exhaustion


def max_tokens_retry_payload(
    payload: dict[str, Any], status_code: int, response_text: str
) -> dict[str, Any] | None:
    """Swap output-token keys when an endpoint explicitly requests the newer one."""
    text = response_text.lower()
    if (
        status_code not in (400, 422)
        or "max_tokens" not in payload
        or "max_tokens" not in text
        or "max_completion_tokens" not in text
        or not any(
            word in text for word in ("unsupported", "not supported", "unknown", "unrecognized")
        )
    ):
        return None
    retry = dict(payload)
    retry["max_completion_tokens"] = retry.pop("max_tokens")
    return retry


_SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def sampling_retry_payload(
    payload: dict[str, Any], status_code: int, response_text: str
) -> dict[str, Any] | None:
    """Drop sampling knobs when an endpoint rejects them by name.

    Current Claude models return HTTP 400 for ``temperature``, ``top_p``, and
    ``top_k`` rather than ignoring them.  The benchmark's ``temperature=0``
    never guaranteed determinism there anyway, so the request is worth more
    without the field than the field is worth.
    """
    if status_code not in (400, 422):
        return None
    text = response_text.lower()
    present = [key for key in _SAMPLING_KEYS if key in payload]
    if not present or not any(key in text for key in present):
        return None
    return {k: v for k, v in payload.items() if k not in _SAMPLING_KEYS}
