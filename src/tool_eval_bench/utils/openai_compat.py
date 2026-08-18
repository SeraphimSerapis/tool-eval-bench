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
