"""Shared test fixtures for tool-eval-bench.

Provides ``make_state`` and ``make_tool_call`` helpers that were previously
duplicated across 6+ test files.
"""

from __future__ import annotations

import json
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
    ToolResultRecord,
)


# Legacy runner unit tests historically passed a raw MockTransport client into
# measurement internals. Keep that compatibility in test code while production
# runners use only the semantic MeasurementClient port.
async def _test_tokenize(self: httpx.AsyncClient, *, model: str, text: str) -> httpx.Response:
    return await self.post("http://test/tokenize", json={"model": model, "prompt": text})


async def _test_models(self: httpx.AsyncClient) -> httpx.Response:
    return await self.get("http://test/v1/models")


async def _test_metrics(
    self: httpx.AsyncClient, *, metrics_url: str | None = None
) -> httpx.Response:
    return await self.get(metrics_url or "http://test/metrics")


async def _test_completion(self: httpx.AsyncClient, payload: dict[str, Any]) -> httpx.Response:
    return await self.post("http://test/v1/chat/completions", json=payload)


def _test_stream_completion(
    self: httpx.AsyncClient, payload: dict[str, Any]
) -> AbstractAsyncContextManager[httpx.Response]:
    return self.stream("POST", "http://test/v1/chat/completions", json=payload)


httpx.AsyncClient.tokenize = _test_tokenize  # type: ignore[attr-defined]
httpx.AsyncClient.models = _test_models  # type: ignore[attr-defined]
httpx.AsyncClient.metrics = _test_metrics  # type: ignore[attr-defined]
httpx.AsyncClient.completion = _test_completion  # type: ignore[attr-defined]
httpx.AsyncClient.stream_completion = _test_stream_completion  # type: ignore[attr-defined]


def make_state(
    *,
    tool_calls: list[ToolCallRecord] | list[dict] | None = None,
    tool_results: list[ToolResultRecord] | list[dict] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
    meta: dict | None = None,
) -> ScenarioState:
    """Build a ``ScenarioState`` for testing.

    Accepts either typed records *or* plain dicts (auto-converted).
    This unifies the various ``_make_state`` helpers that were duplicated
    across test modules.
    """
    state = ScenarioState()
    state.final_answer = final_answer
    state.assistant_messages = assistant_messages or ([final_answer] if final_answer else [])
    state.meta = meta or {}

    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, ToolCallRecord):
                state.tool_calls.append(tc)
            elif isinstance(tc, dict):
                state.tool_calls.append(
                    ToolCallRecord(
                        id=tc.get("id", f"call_{len(state.tool_calls)}"),
                        name=tc["name"],
                        raw_arguments=json.dumps(tc.get("arguments", {})),
                        arguments=tc.get("arguments", {}),
                        turn=tc.get("turn", 1),
                        user_phase=tc.get("user_phase"),
                    )
                )
            else:
                # Allow arbitrary objects (e.g. MagicMock) for flexible testing
                state.tool_calls.append(tc)

    if tool_results:
        for tr in tool_results:
            if isinstance(tr, ToolResultRecord):
                state.tool_results.append(tr)
            elif isinstance(tr, dict):
                state.tool_results.append(
                    ToolResultRecord(
                        call_id=tr.get("call_id", f"call_{len(state.tool_results)}"),
                        name=tr.get("name", "unknown"),
                        result=tr.get("result"),
                    )
                )
            else:
                # Allow arbitrary objects (e.g. MagicMock) for flexible testing
                state.tool_results.append(tr)

    return state


def make_tool_call(
    name: str = "unknown_tool",
    arguments: dict | None = None,
    turn: int = 1,
    call_id: str | None = None,
    user_phase: int | None = None,
) -> ToolCallRecord:
    """Build a ``ToolCallRecord`` for testing."""
    args = arguments or {}
    return ToolCallRecord(
        id=call_id or f"tc_{turn}",
        name=name,
        raw_arguments=json.dumps(args),
        arguments=args,
        turn=turn,
        user_phase=user_phase,
    )
