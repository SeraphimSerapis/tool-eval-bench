"""Shared test fixtures for tool-eval-bench.

Provides ``make_state`` and ``make_tool_call`` helpers that were previously
duplicated across 6+ test files.
"""

from __future__ import annotations

import importlib.util
import json
import os
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import pytest

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ToolCallRecord,
    ToolResultRecord,
)

_open_repositories: list[Any] = []


def open_repository(**kwargs: Any) -> Any:
    """A `RunRepository` closed at the end of the test, whatever happens.

    Windows refuses to delete a file that is still open, so a test that fails
    before reaching its own `close()` leaves `tmp_path` un-removable and
    reports `PermissionError` instead of the assertion that actually failed.
    Calling `close()` yourself as well is fine; it is idempotent.
    """
    from tool_eval_bench.storage.db import RunRepository

    repository = RunRepository(**kwargs)
    _open_repositories.append(repository)
    return repository


@pytest.fixture(autouse=True)
def _hardmode_states_carry_results(request, monkeypatch):
    """Fail a test that grades a Hard Mode state holding a call with no result.

    Evaluators read a missing result as "unknown" and grade it leniently, so a
    test built that way checks a path no production run takes. Record results
    with the scenario's handler, or pass the state through
    ``simulate_results``. A test that is about missing results opts out with
    ``@pytest.mark.allow_missing_results``.
    """
    if request.node.get_closest_marker("allow_missing_results"):
        yield
        return
    from tool_eval_bench.evals.scenarios import HARDMODE_SCENARIOS

    for scenario in HARDMODE_SCENARIOS:

        def guarded(state, _evaluate=scenario.evaluate, _sid=scenario.id):
            recorded = {result.call_id for result in state.tool_results}
            missing = [
                call.name
                for call in state.tool_calls
                if isinstance(call, ToolCallRecord) and call.id not in recorded
            ]
            assert not missing, (
                f"{_sid} graded calls without results: {missing}. "
                "Record each handler result or use simulate_results()."
            )
            return _evaluate(state)

        monkeypatch.setattr(scenario, "evaluate", guarded)
    yield


@pytest.fixture(autouse=True)
def _close_leaked_repositories():
    """Close anything `open_repository` handed out during the test."""
    yield
    while _open_repositories:
        _open_repositories.pop().close()


def utf8_subprocess_env() -> dict[str, str]:
    """Environment for a child Python that writes non-ASCII to a pipe.

    The CLI prints star ratings and em dashes. A child process on Windows
    encodes stdout as cp1252 by default and raises before the parent ever sees
    the output.
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


#: `termios` and `tty` are POSIX-only. The keypress reader returns None without
#: them by design, so tests that drive it have nothing to assert on Windows.
requires_termios = pytest.mark.skipif(
    importlib.util.find_spec("termios") is None,
    reason="termios and tty are POSIX-only",
)


class MeasurementTestClient(httpx.AsyncClient):
    """An ``httpx.AsyncClient`` that also satisfies the measurement port.

    Some runner tests hand a ``MockTransport`` client straight to measurement
    internals, which call the port's methods rather than raw ``get``/``post``.
    This used to be done by attaching those five methods to
    ``httpx.AsyncClient`` itself at import time, for the whole session: a
    future httpx release adding a same-named method would have been silently
    overridden, in every test, with no failure pointing at the cause.

    Subclassing keeps the convenience and confines it to the tests that opt in.
    """

    async def tokenize(self, *, model: str, text: str) -> httpx.Response:
        return await self.post("http://test/tokenize", json={"model": model, "prompt": text})

    async def models(self) -> httpx.Response:
        return await self.get("http://test/v1/models")

    async def metrics(self, *, metrics_url: str | None = None) -> httpx.Response:
        return await self.get(metrics_url or "http://test/metrics")

    async def completion(self, payload: dict[str, Any]) -> httpx.Response:
        return await self.post("http://test/v1/chat/completions", json=payload)

    def stream_completion(
        self, payload: dict[str, Any]
    ) -> AbstractAsyncContextManager[httpx.Response]:
        return self.stream("POST", "http://test/v1/chat/completions", json=payload)


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


def simulate_results(state: ScenarioState, scenario: Any) -> ScenarioState:
    """Replay ``state``'s tool calls through the scenario's handler, as the runner does.

    A state built from bare calls has no results, and evaluators read a
    missing result as "unknown" rather than as a failure, so such a test
    grades a more lenient path than any real run takes. This rebuilds the
    calls in order, recording each handler result and checkpoint, so the
    evaluator sees what production would have shown it. Results already
    recorded for a call are kept instead of simulated.
    """
    calls = list(state.tool_calls)
    recorded = {result.call_id for result in state.tool_results}
    state.tool_calls.clear()
    for call in calls:
        result = scenario.handle_tool_call(state, call)
        state.tool_calls.append(call)
        if call.id not in recorded:
            state.tool_results.append(
                ToolResultRecord(call_id=call.id, name=call.name, result=result)
            )
        if scenario.checkpoint:
            diagnostic = scenario.checkpoint(state, call)
            if diagnostic:
                state.meta.setdefault("state_checkpoints", []).append(diagnostic)
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


def disable_rate_limit_pacing(adapter: Any) -> None:
    """Pin the adapter's rate-limit spacing at zero for tests.

    Patching ``_rate_limit_delay`` only zeroes the post-429 pause.  The shared
    ``RateLimitCoordinator`` separately widens ``_min_interval`` on every 429
    (0.5s, 1s, 2s, 4s, ...) and enforces it with a real ``asyncio.sleep`` in
    ``acquire()``.  Tests that drive repeated 429s therefore burn wall-clock
    time proportional to the retry budget unless the pacing is neutralised too.
    """
    coordinator = adapter._rate_limits
    coordinator._max_interval = 0.0
    coordinator._min_interval = 0.0
