"""Tests for turn-level orchestrator callbacks, in-process SSE, and SQLite checkpoint tailing."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from tests.web_support import TEST_TOKEN, web_client
from tool_eval_bench.domain.adapters import ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
)
from tool_eval_bench.runner.orchestrator import run_all_scenarios
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.web import create_app
from tool_eval_bench.web.live import LiveRunHub, stream_run_events


class _StubAdapter:
    def __init__(self, responses: list[ChatCompletionResult]) -> None:
        self._responses = list(responses)

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        return self._responses.pop(0)


def _make_scenario(sid: str = "TC-01", *, held_out: bool = False) -> ScenarioDefinition:
    def _handler(state: ScenarioState, call: Any) -> dict[str, Any]:
        return {"temperature_c": 18, "condition": "Cloudy"}

    def _eval(state: ScenarioState) -> ScenarioEvaluation:
        return ScenarioEvaluation(ScenarioStatus.PASS, 2, "Good weather call")

    return ScenarioDefinition(
        id=sid,
        title="Weather Lookup",
        category=Category.A,
        user_message="What is the weather in Berlin?",
        description="Call get_weather",
        handle_tool_call=_handler,
        evaluate=_eval,
        held_out=held_out,
    )


def _parse_sse_frames(text: str) -> list[tuple[str, dict[str, Any]]]:
    frames: list[tuple[str, dict[str, Any]]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_type = "message"
        data_line = "{}"
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data_line = line.removeprefix("data: ").strip()
        frames.append((event_type, json.loads(data_line)))
    return frames


@pytest.mark.asyncio
async def test_run_all_scenarios_emits_turn_events_and_redacts_held_out() -> None:
    public_sc = _make_scenario("TC-01", held_out=False)
    held_sc = replace(_make_scenario("PACK-01", held_out=True), title="Secret Pack Title")

    adapter = _StubAdapter(
        [
            ChatCompletionResult(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        id="c1",
                        name="get_weather",
                        arguments_str='{"city": "Berlin"}',
                    )
                ],
                ttft_ms=42.0,
            ),
            ChatCompletionResult(
                content="It is 18C and cloudy in Berlin.",
                tool_calls=[],
                ttft_ms=30.0,
            ),
            ChatCompletionResult(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        id="c2",
                        name="get_weather",
                        arguments_str='{"city": "SecretCity"}',
                    )
                ],
                ttft_ms=25.0,
            ),
            ChatCompletionResult(
                content="Secret final answer.",
                tool_calls=[],
                ttft_ms=20.0,
            ),
        ]
    )

    turn_events: list[tuple[str, int, dict[str, Any]]] = []

    async def _on_turn(scenario: ScenarioDefinition, turn: int, payload: dict[str, Any]) -> None:
        turn_events.append((scenario.id, turn, payload))

    summary = await run_all_scenarios(
        adapter,
        model="test-model",
        base_url="http://localhost:8000",
        scenarios=[public_sc, held_sc],
        on_scenario_turn=_on_turn,
    )
    assert summary.final_score == 100
    assert len(turn_events) == 4

    # Public scenario turn 1 includes tool call and mock result
    assert turn_events[0][0] == "TC-01"
    assert turn_events[0][1] == 1
    assert turn_events[0][2]["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Berlin"}}
    ]
    assert turn_events[0][2]["tool_results"] == [{"temperature_c": 18, "condition": "Cloudy"}]

    # Public scenario turn 2 includes final content
    assert turn_events[1][0] == "TC-01"
    assert turn_events[1][1] == 2
    assert "18C and cloudy" in turn_events[1][2]["content"]

    # Held-out scenario turns withhold tool calls, results, and content
    assert turn_events[2][0] == "PACK-01"
    assert turn_events[2][2]["tool_calls"] == []
    assert turn_events[2][2]["tool_results"] == []
    assert turn_events[2][2]["content"] == ""
    assert turn_events[3][0] == "PACK-01"
    assert turn_events[3][2]["content"] == ""


@pytest.mark.asyncio
async def test_in_process_live_hub_streams_sse_events(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    app = create_app(token=TEST_TOKEN)
    hub = app.state.live_hub

    session = hub.register(
        "run-live-1",
        model="Qwen/Qwen3-8B",
        backend="vllm",
        total=2,
        held_out=frozenset({"PACK-01"}),
    )
    callbacks = hub.make_callbacks(session)
    sc1 = _make_scenario("TC-01", held_out=False)
    sc2 = _make_scenario("PACK-01", held_out=True)

    # A subscriber watching from the start sees every event as it happens.
    live = stream_run_events(hub, "run-live-1")
    live_frames = [await live.__anext__()]

    await callbacks["on_scenario_start"](sc1, 0, 2)
    await callbacks["on_scenario_turn"](
        sc1,
        1,
        {
            "max_turns": 8,
            "latency_ms": 120.0,
            "ttft_ms": 35.0,
            "content": "",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Berlin"}}],
            "tool_results": [{"temperature_c": 18}],
        },
    )
    await callbacks["on_scenario_result"](
        sc1,
        ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Passed weather lookup",
            raw_log="trace",
            duration_seconds=0.5,
        ),
        0,
        2,
    )
    await callbacks["on_scenario_start"](sc2, 1, 2)
    await callbacks["on_scenario_result"](
        sc2,
        ScenarioResult(
            scenario_id="PACK-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Secret held out summary",
            raw_log="SECRET TRACE",
            duration_seconds=0.6,
        ),
        1,
        2,
    )
    hub.mark_complete(
        "run-live-1",
        {
            "run_id": "run-live-1",
            "status": "completed",
            "model": "Qwen/Qwen3-8B",
            "config": {"model": "Qwen/Qwen3-8B", "backend": "vllm"},
            "scores": {
                "final_score": 100,
                "rating": "★★★★★ Excellent",
                "scenario_results": [
                    {"scenario_id": "TC-01", "status": "pass", "points": 2},
                    {"scenario_id": "PACK-01", "status": "pass", "points": 2},
                ],
            },
        },
    )

    live_frames.extend([frame async for frame in live])
    live_events = _parse_sse_frames("".join(live_frames))
    assert [ev for ev, _ in live_events] == [
        "run_started",
        "scenario_start",
        "scenario_turn",
        "scenario_result",
        "scenario_start",
        "scenario_result",
        "run_complete",
    ]
    assert live_events[4][1]["title"] == "held out"

    # A late subscriber gets the compacted history: one result per finished
    # scenario, without the turn-by-turn detail or any trace.
    async with web_client(app) as client:
        resp = await client.get("/api/runs/run-live-1/events")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        frames = _parse_sse_frames(resp.text)

    assert [ev for ev, _ in frames] == [
        "run_started",
        "scenario_result",
        "scenario_result",
        "run_complete",
    ]
    for text in ("".join(live_frames), resp.text):
        assert "Secret held out summary" not in text
        assert "SECRET TRACE" not in text
    pack_result = frames[2][1]["result"]
    assert pack_result["summary"] == "held out"
    assert "raw_log" not in pack_result
    assert "raw_log" not in frames[1][1]["result"]


@pytest.mark.asyncio
async def test_cross_process_sqlite_checkpoint_sse_tailing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": "cli-run-99",
                "status": "interrupted",
                "model": "Qwen/Qwen3-8B",
                "config": {
                    "model": "Qwen/Qwen3-8B",
                    "backend": "vllm",
                    "scenario_count": 2,
                    "scenario_packs": [
                        {
                            "name": "pack",
                            "scenario_count": 1,
                            "scenario_ids": ["PACK-77"],
                            "content_hash": "abcdef1234567890",
                        }
                    ],
                },
                "scores": {},
                "metadata": {"model": "Qwen/Qwen3-8B", "backend": "vllm"},
            }
        )
        repo.checkpoint_scenario_result(
            "cli-run-99",
            {
                "scenario_id": "TC-01",
                "status": "pass",
                "points": 2,
                "summary": "Checkpointed TC-01",
                "raw_log": "checkpoint trace",
                "duration_seconds": 1.1,
            },
        )
        repo.checkpoint_scenario_result(
            "cli-run-99",
            {
                "scenario_id": "PACK-77",
                "status": "fail",
                "points": 0,
                "summary": "Private checkpoint summary",
                "raw_log": "PRIVATE CHECKPOINT TRACE",
                "duration_seconds": 1.3,
            },
        )

    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )

    app = create_app(token=TEST_TOKEN)
    async with web_client(app) as client:
        prog_resp = await client.get("/api/runs/cli-run-99/progress")
        assert prog_resp.status_code == 200
        prog = prog_resp.json()
        assert len(prog["checkpoints"]) == 2
        assert "PRIVATE CHECKPOINT TRACE" not in prog_resp.text

        sse_resp = await client.get("/api/runs/cli-run-99/events")
        assert sse_resp.status_code == 200
        frames = _parse_sse_frames(sse_resp.text)
        assert [ev for ev, _ in frames] == [
            "run_started",
            "scenario_result",
            "scenario_result",
            "run_interrupted",
        ]
        assert "PRIVATE CHECKPOINT TRACE" not in sse_resp.text

        missing_resp = await client.get("/api/runs/no-such-run/events")
        assert missing_resp.status_code == 404


def _weather_turns() -> list[ChatCompletionResult]:
    return [
        ChatCompletionResult(
            content="",
            tool_calls=[
                ProviderToolCall(id="c1", name="get_weather", arguments_str='{"city": "Berlin"}')
            ],
            ttft_ms=42.0,
        ),
        ChatCompletionResult(content="It is 18C and cloudy.", tool_calls=[], ttft_ms=30.0),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_failing_turn_observer_never_changes_the_outcome(concurrency: int) -> None:
    calls: list[int] = []

    async def _broken_observer(
        scenario: ScenarioDefinition, turn: int, payload: dict[str, Any]
    ) -> None:
        calls.append(turn)
        raise RuntimeError("live view bug")

    summary = await run_all_scenarios(
        _StubAdapter(_weather_turns()),
        model="test-model",
        base_url="http://localhost:8000",
        scenarios=[_make_scenario("TC-01")],
        concurrency=concurrency,
        on_scenario_turn=_broken_observer,
    )
    assert calls == [1, 2]
    [result] = summary.scenario_results
    assert result.status is ScenarioStatus.PASS
    assert "live view bug" not in result.raw_log


def _event(kind: str, sid: str) -> dict[str, Any]:
    return {"event": kind, "scenario_id": sid}


def test_session_history_drops_turn_detail_once_a_scenario_finishes() -> None:
    hub = LiveRunHub()
    session = hub.register("r", model="m", backend="vllm", total=2)
    session.publish(_event("scenario_start", "TC-01"))
    session.publish(_event("scenario_turn", "TC-01"))
    session.publish(_event("scenario_start", "TC-02"))
    session.publish(_event("scenario_turn", "TC-02"))
    session.publish(_event("scenario_result", "TC-01"))
    assert [(e["event"], e.get("scenario_id")) for e in session.history] == [
        ("run_started", None),
        ("scenario_start", "TC-02"),
        ("scenario_turn", "TC-02"),
        ("scenario_result", "TC-01"),
    ]


def test_hub_keeps_only_recent_finished_sessions() -> None:
    hub = LiveRunHub(max_finished=2)
    for idx in range(4):
        hub.register(f"done-{idx}", model="m", backend="vllm", total=1)
        hub.mark_interrupted(f"done-{idx}")
    hub.register("active", model="m", backend="vllm", total=1)
    assert hub.get("done-0") is None and hub.get("done-1") is None
    assert hub.get("done-2") is not None and hub.get("done-3") is not None
    assert hub.get("active") is not None


def test_register_replays_preserved_results_after_run_started() -> None:
    hub = LiveRunHub()
    session = hub.register(
        "r",
        model="m",
        backend="vllm",
        total=3,
        preserved=[{"scenario_id": "TC-01", "status": "pass", "points": 2}],
    )
    kinds = [(e["event"], e.get("preserved")) for e in session.history]
    assert kinds == [("run_started", None), ("scenario_result", True)]
    assert session.history[0]["managed"] is True


@pytest.mark.asyncio
async def test_idle_session_stream_sends_heartbeats() -> None:
    hub = LiveRunHub()
    hub.register("r", model="m", backend="vllm", total=1)
    stream = stream_run_events(hub, "r", heartbeat_seconds=0.01)
    assert (await stream.__anext__()).startswith("event: run_started")
    assert await stream.__anext__() == ": keep-alive\n\n"
    await stream.aclose()


def _store(db_path: str, run_id: str, status: str, results: list[dict[str, Any]]) -> None:
    with RunRepository(db_path=db_path) as repo:
        repo.upsert_scenario_run(
            {
                "run_id": run_id,
                "status": status,
                "model": "Qwen/Qwen3-8B",
                "config": {"model": "Qwen/Qwen3-8B", "backend": "vllm", "scenario_count": 2},
                "scores": {"final_score": 50, "scenario_results": results},
                "metadata": {},
            }
        )


@pytest.mark.asyncio
async def test_tail_of_completed_cli_run_sends_stored_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    _store(
        db_path,
        "cli-done",
        "completed",
        [
            {"scenario_id": "TC-01", "status": "pass", "points": 2, "raw_log": "trace one"},
            {"scenario_id": "TC-02", "status": "fail", "points": 0, "raw_log": "trace two"},
        ],
    )
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    text = "".join([f async for f in stream_run_events(LiveRunHub(), "cli-done")])
    frames = _parse_sse_frames(text)
    assert [ev for ev, _ in frames] == [
        "run_started",
        "scenario_result",
        "scenario_result",
        "run_complete",
    ]
    assert frames[0][1]["managed"] is False
    assert frames[3][1]["passes"] == 1
    assert "trace one" not in text


@pytest.mark.asyncio
async def test_tail_gives_up_on_a_stale_running_row(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    _store(db_path, "cli-stale", "running", [])
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    frames = _parse_sse_frames(
        "".join(
            [
                f
                async for f in stream_run_events(
                    LiveRunHub(), "cli-stale", poll_interval=0.0, idle_timeout=0.0
                )
            ]
        )
    )
    assert [ev for ev, _ in frames] == ["run_started", "stream_idle"]
    assert "process may have exited" in frames[1][1]["message"]


@pytest.mark.asyncio
async def test_tail_reports_a_run_that_vanishes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = str(tmp_path / "benchmarks.sqlite")
    monkeypatch.setattr(
        "tool_eval_bench.application.run_queries._repository",
        lambda: RunRepository(db_path=db_path),
    )
    frames = _parse_sse_frames(
        "".join([f async for f in stream_run_events(LiveRunHub(), "never-existed")])
    )
    # Not "error": EventSource reserves that name for connection failures.
    assert [ev for ev, _ in frames] == ["stream_error"]


@pytest.mark.asyncio
async def test_resumed_session_reports_the_whole_run_total() -> None:
    """A resume runs a subset; progress must still count against the full protocol."""
    hub = LiveRunHub()
    session = hub.register(
        "r",
        model="m",
        backend="vllm",
        total=4,
        preserved=[{"scenario_id": "TC-01", "status": "pass", "points": 2}],
    )
    callbacks = hub.make_callbacks(session)
    scenario = _make_scenario("TC-02")
    # The orchestrator only knows the three scenarios it was asked to run.
    await callbacks["on_scenario_start"](scenario, 0, 3)
    await callbacks["on_scenario_result"](
        scenario,
        ScenarioResult(
            scenario_id="TC-02", status=ScenarioStatus.PASS, points=2, summary="ok", raw_log="t"
        ),
        0,
        3,
    )
    totals = [e["total"] for e in session.history if "total" in e]
    assert totals == [4, 4, 4]
