"""Integration tests for the multi-turn scenario orchestrator.

TEST-03: Tests run_scenario end-to-end with a mock adapter that returns
canned responses, verifying the full orchestration loop: message threading,
tool call dispatch, trace capture, and evaluation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.runner.orchestrator import run_scenario


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class MockAdapter(BackendAdapter):
    """Adapter that returns a sequence of pre-configured responses.

    Each entry in `responses` is either:
      - A ChatCompletionResult (returned as-is)
      - A dict with 'content' and optionally 'tool_calls' (auto-converted)
    """

    def __init__(self, responses: list[dict | ChatCompletionResult]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.captured_payloads: list[dict] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        # Deep-copy messages to capture state at call time (orchestrator mutates in-place)
        import copy
        captured = dict(kwargs)
        if "messages" in captured:
            captured["messages"] = copy.deepcopy(captured["messages"])
        self.captured_payloads.append(captured)
        if self._call_count >= len(self._responses):
            return ChatCompletionResult(content="[exhausted mock responses]")

        resp = self._responses[self._call_count]
        self._call_count += 1

        if isinstance(resp, ChatCompletionResult):
            return resp

        # Auto-convert dict shorthand
        tool_calls = []
        for tc in resp.get("tool_calls", []):
            tool_calls.append(ProviderToolCall(
                id=tc.get("id", f"tc_{len(tool_calls)}"),
                name=tc["name"],
                arguments_str=json.dumps(tc.get("arguments", {})),
            ))

        return ChatCompletionResult(
            content=resp.get("content", ""),
            tool_calls=tool_calls,
            elapsed_ms=resp.get("elapsed_ms", 10.0),
            ttft_ms=resp.get("ttft_ms"),
        )


# ---------------------------------------------------------------------------
# Test scenarios - simple deterministic definitions
# ---------------------------------------------------------------------------


def _simple_handler(state: ScenarioState, call: ToolCallRecord) -> Any:
    """Mock tool handler that returns a canned string."""
    if call.name == "get_weather":
        return {"temperature": "22C", "condition": "sunny"}
    if call.name == "calculator":
        return {"result": 42}
    return {"error": f"Unknown tool: {call.name}"}


def _simple_evaluator(state: ScenarioState) -> ScenarioEvaluation:
    """Pass if get_weather was called, fail otherwise."""
    if any(tc.name == "get_weather" for tc in state.tool_calls):
        return ScenarioEvaluation(
            status=ScenarioStatus.PASS,
            points=2,
            summary="Correct tool used",
        )
    return ScenarioEvaluation(
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Expected get_weather call",
    )


MOCK_SCENARIO = ScenarioDefinition(
    id="TEST-01",
    title="Test weather query",
    category=Category.A,
    user_message="What's the weather in Berlin?",
    description="Should call get_weather",
    handle_tool_call=_simple_handler,
    evaluate=_simple_evaluator,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_no_tools() -> None:
    """Model responds directly without calling any tools."""
    adapter = MockAdapter([
        {"content": "It's probably cold in Berlin."},
    ])

    result = await run_scenario(
        adapter, model="test", base_url="http://localhost:8000",
        api_key=None, scenario=MOCK_SCENARIO,
    )

    assert result.scenario_id == "TEST-01"
    assert result.status == ScenarioStatus.FAIL  # no tool call
    assert result.points == 0
    assert result.turn_count == 1
    assert result.duration_seconds > 0
    assert "final_answer=" in result.raw_log


@pytest.mark.asyncio
async def test_tool_call_pass() -> None:
    """Model calls get_weather → mock returns data → model summarizes → PASS."""
    adapter = MockAdapter([
        # Turn 1: model calls get_weather
        {
            "content": "",
            "tool_calls": [
                {"name": "get_weather", "arguments": {"location": "Berlin"}},
            ],
            "ttft_ms": 50.0,
        },
        # Turn 2: model summarizes the tool result
        {"content": "Berlin is 22C and sunny."},
    ])

    result = await run_scenario(
        adapter, model="test", base_url="http://localhost:8000",
        api_key=None, scenario=MOCK_SCENARIO,
    )

    assert result.status == ScenarioStatus.PASS
    assert result.points == 2
    assert result.turn_count == 2
    assert result.ttft_ms == 50.0
    assert len(result.turn_latencies_ms) == 2
    assert "get_weather" in result.raw_log
    assert len(result.tool_calls_made) == 1
    assert "Berlin" in result.tool_calls_made[0]


@pytest.mark.asyncio
async def test_multi_turn_tool_chain() -> None:
    """Model makes two sequential tool calls across turns."""
    def chain_evaluator(state: ScenarioState) -> ScenarioEvaluation:
        names = [tc.name for tc in state.tool_calls]
        if names == ["get_weather", "calculator"]:
            return ScenarioEvaluation(status=ScenarioStatus.PASS, points=2, summary="Chain correct")
        return ScenarioEvaluation(status=ScenarioStatus.FAIL, points=0, summary="Wrong chain")

    chain_scenario = ScenarioDefinition(
        id="TEST-02", title="Chain test", category=Category.C,
        user_message="Get weather then calculate something",
        description="Should chain two tools",
        handle_tool_call=_simple_handler,
        evaluate=chain_evaluator,
    )

    adapter = MockAdapter([
        {"content": "", "tool_calls": [{"name": "get_weather", "arguments": {"location": "NYC"}}]},
        {"content": "", "tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]},
        {"content": "Done! Weather is 22C, calculation is 42."},
    ])

    result = await run_scenario(
        adapter, model="test", base_url="http://localhost:8000",
        api_key=None, scenario=chain_scenario,
    )

    assert result.status == ScenarioStatus.PASS
    assert result.turn_count == 3
    assert len(result.tool_calls_made) == 2


@pytest.mark.asyncio
async def test_max_turns_exceeded() -> None:
    """If the model keeps calling tools without settling, it hits max_turns."""
    # Model always returns a tool call, never a final answer
    responses = [
        {"content": "", "tool_calls": [{"name": "get_weather", "arguments": {"location": "X"}}]}
        for _ in range(10)
    ]
    adapter = MockAdapter(responses)

    result = await run_scenario(
        adapter, model="test", base_url="http://localhost:8000",
        api_key=None, scenario=MOCK_SCENARIO,
        max_turns=3,
    )

    # Should still evaluate (with whatever state accumulated)
    assert result.turn_count == 3
    # The evaluator will PASS because get_weather was called
    assert result.status == ScenarioStatus.PASS


@pytest.mark.asyncio
async def test_adapter_exception_produces_fail() -> None:
    """If the adapter raises an exception, the scenario should FAIL gracefully."""
    class FailingAdapter(BackendAdapter):
        async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
            raise ConnectionError("Server down")

    result = await run_scenario(
        FailingAdapter(), model="test", base_url="http://localhost:8000",
        api_key=None, scenario=MOCK_SCENARIO,
    )

    assert result.status == ScenarioStatus.FAIL
    assert result.points == 0
    assert "Server down" in result.summary
    assert "error=" in result.raw_log


@pytest.mark.asyncio
async def test_messages_threaded_correctly() -> None:
    """Verify the adapter receives properly threaded messages across turns."""
    adapter = MockAdapter([
        {"content": "", "tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]},
        {"content": "Berlin is sunny."},
    ])

    await run_scenario(
        adapter, model="test", base_url="http://localhost:8000",
        api_key=None, scenario=MOCK_SCENARIO,
    )

    # Verify we got exactly 2 adapter calls (turn 1: tool call, turn 2: final answer)
    assert len(adapter.captured_payloads) == 2

    # First call should include system and user messages
    msgs_1 = adapter.captured_payloads[0]["messages"]
    roles_1 = [m["role"] for m in msgs_1]
    assert "system" in roles_1
    assert "user" in roles_1
    # No tool results yet in first call
    assert "tool" not in roles_1

    # Second call: must include the assistant tool_calls and tool result
    msgs_2 = adapter.captured_payloads[1]["messages"]
    roles_2 = [m["role"] for m in msgs_2]
    assert "tool" in roles_2  # tool result injected
    # Find the assistant message with tool_calls
    assistant_msgs = [m for m in msgs_2 if m["role"] == "assistant" and "tool_calls" in m]
    assert len(assistant_msgs) >= 1
    # Find the tool result message
    tool_msgs = [m for m in msgs_2 if m["role"] == "tool"]
    assert len(tool_msgs) >= 1
    # Tool result should contain the mock handler's JSON output
    tool_content = json.loads(tool_msgs[0]["content"])
    assert tool_content["temperature"] == "22C"


@pytest.mark.asyncio
async def test_scenario_timing_fields() -> None:
    """Result should include timing fields even for fast mock runs."""
    adapter = MockAdapter([
        {"content": "Direct answer.", "elapsed_ms": 25.0, "ttft_ms": 5.0},
    ])

    result = await run_scenario(
        adapter, model="test", base_url="http://localhost:8000",
        api_key=None, scenario=MOCK_SCENARIO,
    )

    assert result.duration_seconds > 0
    assert result.ttft_ms == 5.0
    assert len(result.turn_latencies_ms) == 1


@pytest.mark.asyncio
async def test_follow_up_messages() -> None:
    """Scenario with follow_up_messages triggers a multi-turn conversation.

    Turn 1: model calls a tool (create event).
    Turn 2: model summarizes the tool result.
    -- orchestrator injects follow-up user message --
    Turn 3: model answers the follow-up.
    """
    def follow_up_evaluator(state: ScenarioState) -> ScenarioEvaluation:
        """Pass if model created the event AND answered the follow-up."""
        created = any(tc.name == "create_calendar_event" for tc in state.tool_calls)
        # The final answer should be the response to the follow-up question
        if created and "no attendee" in state.final_answer.lower():
            return ScenarioEvaluation(
                status=ScenarioStatus.PASS, points=2,
                summary="Created event and answered follow-up correctly",
            )
        if created:
            return ScenarioEvaluation(
                status=ScenarioStatus.PARTIAL, points=1,
                summary="Created event but follow-up answer unclear",
            )
        return ScenarioEvaluation(
            status=ScenarioStatus.FAIL, points=0, summary="Did not create event",
        )

    follow_up_scenario = ScenarioDefinition(
        id="TEST-FU",
        title="Follow-up test",
        category=Category.I,
        user_message="Create a meeting titled Design Review at 3pm.",
        description="Multi-turn: create event, then answer follow-up about attendees",
        handle_tool_call=_simple_handler,
        evaluate=follow_up_evaluator,
        follow_up_messages=["Who is attending the Design Review?"],
    )

    # Mock a custom handler that recognizes create_calendar_event
    def _fu_handler(state: ScenarioState, call: ToolCallRecord) -> Any:
        if call.name == "create_calendar_event":
            return {"event_id": "evt_1", "status": "created", "attendees": []}
        return {"error": f"Unknown: {call.name}"}

    follow_up_scenario.handle_tool_call = _fu_handler

    adapter = MockAdapter([
        # Turn 1: model calls create_calendar_event
        {
            "content": "",
            "tool_calls": [
                {"name": "create_calendar_event", "arguments": {
                    "title": "Design Review", "date": "2026-03-21", "time": "15:00",
                }},
            ],
        },
        # Turn 2: model responds to user about the created event
        {"content": "I've created the Design Review meeting for 3pm tomorrow."},
        # Turn 3: model answers the follow-up question (injected by orchestrator)
        {"content": "No attendees have been added to the Design Review yet."},
    ])

    result = await run_scenario(
        adapter, model="test", base_url="http://localhost:8000",
        api_key=None, scenario=follow_up_scenario,
    )

    # Should pass — model created event and answered the follow-up
    assert result.status == ScenarioStatus.PASS
    assert result.points == 2
    assert result.turn_count == 3  # tool call + response + follow-up answer

    # Verify the follow-up was injected into the message thread
    assert len(adapter.captured_payloads) == 3
    # The 3rd call should include a user message with the follow-up
    msgs_3 = adapter.captured_payloads[2]["messages"]
    user_msgs = [m for m in msgs_3 if m["role"] == "user"]
    follow_up_found = any("attending" in m["content"].lower() for m in user_msgs)
    assert follow_up_found, "Follow-up user message should be in the 3rd adapter call"

    # Trace should include the follow-up
    assert "user_follow_up_1=" in result.raw_log
