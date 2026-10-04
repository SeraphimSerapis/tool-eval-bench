"""Adapter for decision models served on llama.cpp's ``/v1/systemone``.

Subclasses the OpenAI-compatible adapter so a run keeps one HTTP client, one
retry policy, and the user's headers.  Chat completions stay available for
comparing a decision model against a chat model on the same items.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.domain.decision import (
    DecisionBackend,
    DecisionQuestion,
    DecisionRequestError,
    DecisionResult,
    DecisionUnsupportedError,
    parse_answers,
)
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.utils.urls import redact_url, systemone_url

logger = logging.getLogger(__name__)

# A server without the decision endpoint answers one of these.
_UNSUPPORTED_STATUSES = frozenset({404, 405, 501})


class SystemOneAdapter(OpenAICompatibleAdapter, DecisionBackend):
    """Chat-capable adapter that also speaks the decision-model wire format."""

    async def decide(
        self,
        *,
        model: str,
        state: str,
        questions: Mapping[str, DecisionQuestion],
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        api_key: str | None = None,
        base_url: str = "",
    ) -> DecisionResult:
        url = systemone_url(base_url)
        # ``model`` selects a model in router mode and is ignored by a
        # single-model server, so it is always sent.
        payload: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": {name: q.to_wire() for name, q in questions.items()},
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers = self._request_headers(headers, None)

        client = self._get_client()
        req_timeout = httpx.Timeout(timeout_seconds)

        async def attempt() -> DecisionResult:
            await self._rate_limits.acquire()
            started = time.perf_counter()
            response = await client.post(url, json=payload, headers=headers, timeout=req_timeout)
            elapsed_ms = (time.perf_counter() - started) * 1000
            status = response.status_code
            if status in _UNSUPPORTED_STATUSES:
                raise DecisionUnsupportedError(
                    f"{redact_url(url)} returned {status}: the server does not serve "
                    "decision models (llama.cpp exposes /v1/systemone for them)"
                )
            if status == 400:
                raise DecisionRequestError(f"decision request rejected: {response.text[:200]}")
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("decision response is not a JSON object")
            usage = body.get("usage")
            input_tokens = int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
            return DecisionResult(
                answers=parse_answers(body, questions),
                input_tokens=input_tokens,
                elapsed_ms=elapsed_ms,
                raw_response=body,
            )

        return await self._with_retries(attempt, url=url)
