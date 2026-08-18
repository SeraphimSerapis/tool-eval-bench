"""Small compatibility helpers for OpenAI-style request payloads."""

from __future__ import annotations

from typing import Any


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
