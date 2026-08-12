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
    FailureKind,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
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
            tool_calls.append(
                ProviderToolCall(
                    id=tc.get("id", f"tc_{len(tool_calls)}"),
                    name=tc["name"],
                    arguments_str=json.dumps(tc.get("arguments", {})),
                )
            )

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
    adapter = MockAdapter(
        [
            {"content": "It's probably cold in Berlin."},
        ]
    )

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=MOCK_SCENARIO,
    )

    assert result.scenario_id == "TEST-01"
    assert result.status == ScenarioStatus.FAIL  # no tool call
    assert result.points == 0
    assert result.turn_count == 1
    assert result.duration_seconds > 0
    assert "available_tools=" in result.raw_log
    assert "get_weather" in result.raw_log
    assert "tool_choice=auto" in result.raw_log
    assert "final_answer=" in result.raw_log


@pytest.mark.asyncio
async def test_trace_marks_scenario_with_no_tools() -> None:
    """Traces distinguish no-tool scenarios from models that ignored offered tools."""
    no_tool_scenario = ScenarioDefinition(
        id="TEST-NO-TOOLS",
        title="No tools",
        category=Category.O,
        user_message="Return hello.",
        description="No tools are available.",
        handle_tool_call=_simple_handler,
        evaluate=lambda state: ScenarioEvaluation(
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
        ),
        tools_override=[],
    )
    adapter = MockAdapter([{"content": "hello"}])

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=no_tool_scenario,
    )

    assert "available_tools=(none)" in result.raw_log
    assert "tool_choice=" not in result.raw_log
    assert adapter.captured_payloads[0]["tools"] is None


@pytest.mark.asyncio
async def test_tool_call_pass() -> None:
    """Model calls get_weather → mock returns data → model summarizes → PASS."""
    adapter = MockAdapter(
        [
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
        ]
    )

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=MOCK_SCENARIO,
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
        id="TEST-02",
        title="Chain test",
        category=Category.C,
        user_message="Get weather then calculate something",
        description="Should chain two tools",
        handle_tool_call=_simple_handler,
        evaluate=chain_evaluator,
    )

    adapter = MockAdapter(
        [
            {
                "content": "",
                "tool_calls": [{"name": "get_weather", "arguments": {"location": "NYC"}}],
            },
            {
                "content": "",
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}],
            },
            {"content": "Done! Weather is 22C, calculation is 42."},
        ]
    )

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=chain_scenario,
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
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=MOCK_SCENARIO,
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
        FailingAdapter(),
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=MOCK_SCENARIO,
    )

    assert result.status == ScenarioStatus.FAIL
    assert result.points == 0
    assert "Server down" in result.summary
    assert "error=" in result.raw_log


@pytest.mark.asyncio
async def test_messages_threaded_correctly() -> None:
    """Verify the adapter receives properly threaded messages across turns."""
    adapter = MockAdapter(
        [
            {
                "content": "",
                "tool_calls": [{"name": "get_weather", "arguments": {"location": "Berlin"}}],
            },
            {"content": "Berlin is sunny."},
        ]
    )

    await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=MOCK_SCENARIO,
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
    adapter = MockAdapter(
        [
            {"content": "Direct answer.", "elapsed_ms": 25.0, "ttft_ms": 5.0},
        ]
    )

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=MOCK_SCENARIO,
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
                status=ScenarioStatus.PASS,
                points=2,
                summary="Created event and answered follow-up correctly",
            )
        if created:
            return ScenarioEvaluation(
                status=ScenarioStatus.PARTIAL,
                points=1,
                summary="Created event but follow-up answer unclear",
            )
        return ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Did not create event",
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

    adapter = MockAdapter(
        [
            # Turn 1: model calls create_calendar_event
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "create_calendar_event",
                        "arguments": {
                            "title": "Design Review",
                            "date": "2026-03-21",
                            "time": "15:00",
                        },
                    },
                ],
            },
            # Turn 2: model responds to user about the created event
            {"content": "I've created the Design Review meeting for 3pm tomorrow."},
            # Turn 3: model answers the follow-up question (injected by orchestrator)
            {"content": "No attendees have been added to the Design Review yet."},
        ]
    )

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=follow_up_scenario,
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


@pytest.mark.asyncio
async def test_parallel_tool_turns_records_same_turn_batch() -> None:
    adapter = MockAdapter(
        [
            {
                "content": "",
                "tool_calls": [
                    {"name": "get_weather", "arguments": {"location": "Berlin"}},
                    {"name": "calculator", "arguments": {"expression": "1+1"}},
                ],
            },
            {"content": "Done."},
        ]
    )

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=MOCK_SCENARIO,
    )

    assert result.parallel_tool_turns == [1]


@pytest.mark.asyncio
async def test_checkpoint_runs_after_each_tool_call() -> None:
    seen: list[str] = []

    def checkpoint(state: ScenarioState, call: ToolCallRecord) -> str | None:
        seen.append(call.name)
        if call.name == "calculator":
            return "calculator observed"
        return None

    scenario = ScenarioDefinition(
        id="TEST-CP",
        title="Checkpoint test",
        category=Category.P,
        user_message="Call two tools",
        description="Checkpoint after each call",
        handle_tool_call=_simple_handler,
        evaluate=_simple_evaluator,
        checkpoint=checkpoint,
    )
    adapter = MockAdapter(
        [
            {
                "content": "",
                "tool_calls": [
                    {"name": "get_weather", "arguments": {"location": "Berlin"}},
                    {"name": "calculator", "arguments": {"expression": "1+1"}},
                ],
            },
            {"content": "Done."},
        ]
    )

    result = await run_scenario(
        adapter,
        model="test",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=scenario,
    )

    assert seen == ["get_weather", "calculator"]
    assert result.state_checkpoints == ["calculator observed"]


# ---------------------------------------------------------------------------
# Per-scenario turn budget (max_turns_override) and budget-exhaustion signals
# ---------------------------------------------------------------------------


def _stalled_tool_adapter(tool_name: str = "get_weather") -> MockAdapter:
    """Adapter that always calls a tool and never produces a final answer."""
    return MockAdapter(
        [{"content": "", "tool_calls": [{"name": tool_name, "arguments": {}}]} for _ in range(30)]
    )


def _tool_then_final_adapter(tool_name: str = "get_weather") -> MockAdapter:
    """Adapter that calls a tool once, then gives a final answer."""
    return MockAdapter(
        [
            {"content": "", "tool_calls": [{"name": tool_name, "arguments": {}}]},
            {"content": "done"},
        ]
    )


def _tool_failing_evaluator(scenario_id: str) -> ScenarioDefinition:
    """Scenario whose evaluator always fails unless the tool was never used."""
    return ScenarioDefinition(
        id=scenario_id,
        title=scenario_id,
        description=scenario_id,
        category=Category.A,
        user_message="Use the tool.",
        handle_tool_call=lambda state, call: state.tool_calls.append(call),
        evaluate=lambda state: ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="never used required tool",
        ),
    )


@pytest.mark.asyncio
async def test_max_turns_override_extends_budget() -> None:
    """A scenario's max_turns_override replaces the global max_turns."""
    scenario = _tool_failing_evaluator("OVR-01")
    scenario.max_turns_override = 5
    result = await run_scenario(
        _stalled_tool_adapter(),
        model="test-model",
        base_url="http://localhost:8000",
        api_key="key",
        scenario=scenario,
        max_turns=2,
    )
    # 5 tool-call turns fit inside the override; the global 2 would have cut it.
    assert result.turn_count == 5
    assert result.turn_budget_exceeded is True
    assert result.failure_kind == FailureKind.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_max_turns_override_keeps_limit_finite() -> None:
    """Even with an override, a looping model stops at the override value."""
    scenario = _tool_failing_evaluator("OVR-02")
    scenario.max_turns_override = 6
    result = await run_scenario(
        _stalled_tool_adapter(),
        model="test-model",
        base_url="http://localhost:8000",
        api_key="key",
        scenario=scenario,
        max_turns=3,
    )
    assert result.turn_count == 6  # stopped at the override, not unbounded
    assert result.turn_budget_exceeded is True
    assert result.failure_kind == FailureKind.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_max_turns_override_not_set_uses_global_budget() -> None:
    """Scenarios without an override keep using the global max_turns."""
    scenario = _tool_failing_evaluator("OVR-03")
    result = await run_scenario(
        _stalled_tool_adapter(),
        model="test-model",
        base_url="http://localhost:8000",
        api_key="key",
        scenario=scenario,
        max_turns=4,
    )
    assert result.turn_count == 4  # global budget applied
    assert result.turn_budget_exceeded is True
    assert result.failure_kind == FailureKind.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_normal_completion_does_not_flag_budget_exceeded() -> None:
    """A model that reaches a final answer is not marked as budget-exhausted."""
    scenario = ScenarioDefinition(
        id="OVR-04",
        title="OVR-04",
        description="OVR-04",
        category=Category.A,
        user_message="Do the task.",
        handle_tool_call=lambda state, call: state.tool_calls.append(call),
        evaluate=lambda state: ScenarioEvaluation(
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
        ),
    )
    result = await run_scenario(
        _tool_then_final_adapter(),
        model="test-model",
        base_url="http://localhost:8000",
        api_key="key",
        scenario=scenario,
        max_turns=8,
    )
    assert result.turn_count == 2
    assert result.turn_budget_exceeded is False
    assert result.failure_kind is None


@pytest.mark.asyncio
async def test_tool_choice_can_relax_after_first_tool_call() -> None:
    """A scenario can require its first tool call and then permit final text."""
    scenario = ScenarioDefinition(
        id="CHOICE-01",
        title="CHOICE-01",
        description="Require one tool, then answer.",
        category=Category.H,
        user_message="Calculate 7 * 8.",
        handle_tool_call=_simple_handler,
        evaluate=lambda state: ScenarioEvaluation(
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
        ),
        tool_choice_override="required",
        tool_choice_after_first_call="auto",
    )
    adapter = MockAdapter(
        [
            {"content": "", "tool_calls": [{"name": "calculator", "arguments": {}}]},
            {"content": "56"},
        ]
    )

    result = await run_scenario(
        adapter,
        model="test-model",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=scenario,
    )

    assert result.turn_budget_exceeded is False
    assert [payload["tool_choice"] for payload in adapter.captured_payloads] == [
        "required",
        "auto",
    ]


@pytest.mark.asyncio
async def test_tool_calls_record_the_active_user_phase() -> None:
    """Tool provenance distinguishes initial work from follow-up authorization."""
    observed_phases: list[int | None] = []

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        observed_phases.extend(call.user_phase for call in state.tool_calls)
        return ScenarioEvaluation(status=ScenarioStatus.PASS, points=2, summary="ok")

    scenario = ScenarioDefinition(
        id="PHASE-01",
        title="PHASE-01",
        description="Act only after authorization.",
        category=Category.I,
        user_message="Prepare the action, but do not execute it.",
        follow_up_messages=["Execute it now."],
        handle_tool_call=_simple_handler,
        evaluate=evaluate,
    )
    adapter = MockAdapter(
        [
            {"content": "Draft prepared."},
            {"content": "", "tool_calls": [{"name": "calculator", "arguments": {}}]},
            {"content": "Done."},
        ]
    )

    await run_scenario(
        adapter,
        model="test-model",
        base_url="http://localhost:8000",
        api_key=None,
        scenario=scenario,
    )

    assert observed_phases == [1]


@pytest.mark.asyncio
async def test_budget_exhaustion_is_distinct_from_evaluator_failure() -> None:
    """Budget run-out sets BUDGET_EXCEEDED, not the evaluator's failure kind."""
    scenario = _tool_failing_evaluator("OVR-05")
    result = await run_scenario(
        _stalled_tool_adapter(),
        model="test-model",
        base_url="http://localhost:8000",
        api_key="key",
        scenario=scenario,
        max_turns=3,
    )
    # Evaluator says FAIL/missing-tool, but the reason must be budget, not tool.
    assert result.status == ScenarioStatus.FAIL
    assert result.turn_budget_exceeded is True
    assert result.failure_kind == FailureKind.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_tc46_has_scenario_budget_override() -> None:
    """TC-46 declares a per-scenario budget for its deep multi-turn workflow."""
    from tool_eval_bench.evals.scenarios_agentic import AGENTIC_SCENARIOS

    tc46 = next(s for s in AGENTIC_SCENARIOS if s.id == "TC-46")
    assert tc46.max_turns_override is not None
    # Canonical reference path: 5 user turns + tool rounds + final answers.
    # search, read2025, read2024, calc, contacts, email + final answers = 11.
    # With parallel calls the minimum is 8; the override keeps headroom finite.
    assert tc46.max_turns_override >= 9
    assert tc46.max_turns_override <= 12


@pytest.mark.asyncio
async def test_deep_multiturn_workflow_fits_override_budget() -> None:
    """A TC-46-like 11-turn reference path fits inside a 12-turn override."""
    # Canonical deep workflow: 1 user message + 4 follow-ups.
    # T1 search, T2 read2025, T3 final, T4 final, T5 read2024+calc (parallel),
    # T6 final, T7 final, T8 contacts+email (parallel), T9 final.
    # The canonical non-parallel path needs up to 11 assistant turns.
    responses = [
        {"content": "", "tool_calls": [{"name": "search_files", "arguments": {}}]},
        {"content": "", "tool_calls": [{"name": "read_file", "arguments": {"file": "2025"}}]},
        {"content": "found"},
        {"content": "read"},
        {
            "content": "",
            "tool_calls": [
                {"name": "read_file", "arguments": {"file": "2024"}},
                {"name": "calculator", "arguments": {"expr": "growth"}},
            ],
        },
        {"content": "compared"},
        {"content": "risks"},
        {
            "content": "",
            "tool_calls": [
                {"name": "get_contacts", "arguments": {}},
                {"name": "send_email", "arguments": {"to": "jordan.park@company.com"}},
            ],
        },
        {"content": "sent"},
    ]
    scenario = ScenarioDefinition(
        id="DEEP-01",
        title="DEEP-01",
        description="deep multi-turn",
        category=Category.A,
        user_message="Find the competitor analysis report.",
        follow_up_messages=[
            "Read the 2025 one.",
            "What's our market share growth compared to last year?",
            "Summarize the key risks from both reports.",
            "Email that summary to my manager.",
        ],
        handle_tool_call=lambda state, call: state.tool_calls.append(call),
        evaluate=lambda state: ScenarioEvaluation(
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
        ),
    )
    scenario.max_turns_override = 12
    result = await run_scenario(
        MockAdapter(responses),
        model="test-model",
        base_url="http://localhost:8000",
        api_key="key",
        scenario=scenario,
        max_turns=8,  # global default would cut the deep workflow
    )
    # The override lets the full 9-turn path complete; global 8 would fail it.
    assert result.status == ScenarioStatus.PASS
    assert result.turn_budget_exceeded is False
    assert result.turn_count == 9


@pytest.mark.asyncio
async def test_deep_multiturn_workflow_cut_by_global_budget() -> None:
    """The same deep workflow is cut at the global max_turns without override."""
    responses = [
        {"content": "", "tool_calls": [{"name": "search_files", "arguments": {}}]},
        {"content": "", "tool_calls": [{"name": "read_file", "arguments": {"file": "2025"}}]},
        {"content": "found"},
        {"content": "read"},
        {
            "content": "",
            "tool_calls": [
                {"name": "read_file", "arguments": {"file": "2024"}},
                {"name": "calculator", "arguments": {"expr": "growth"}},
            ],
        },
        {"content": "compared"},
        {"content": "risks"},
        {
            "content": "",
            "tool_calls": [
                {"name": "get_contacts", "arguments": {}},
                {"name": "send_email", "arguments": {"to": "jordan.park@company.com"}},
            ],
        },
        {"content": "sent"},
    ]
    scenario = ScenarioDefinition(
        id="DEEP-02",
        title="DEEP-02",
        description="deep multi-turn",
        category=Category.A,
        user_message="Find the competitor analysis report.",
        follow_up_messages=[
            "Read the 2025 one.",
            "What's our market share growth compared to last year?",
            "Summarize the key risks from both reports.",
            "Email that summary to my manager.",
        ],
        handle_tool_call=lambda state, call: state.tool_calls.append(call),
        evaluate=lambda state: ScenarioEvaluation(
            status=ScenarioStatus.FAIL,
            points=0,
            summary="did not finish",
        ),
    )
    # No override: the global budget of 6 cuts the path before completion.
    result = await run_scenario(
        MockAdapter(responses),
        model="test-model",
        base_url="http://localhost:8000",
        api_key="key",
        scenario=scenario,
        max_turns=6,
    )
    assert result.turn_count == 6
    assert result.turn_budget_exceeded is True
    assert result.failure_kind == FailureKind.BUDGET_EXCEEDED


def test_turn_budget_exceeded_roundtrips_through_serialization() -> None:
    """budget-exhausted results survive to_dict/from_dict resume round-trips."""
    source = ScenarioResult(
        scenario_id="TC-46",
        status=ScenarioStatus.FAIL,
        points=0,
        summary="Turn budget exceeded before final email.",
        turn_budget_exceeded=True,
    )

    restored = ScenarioResult.from_dict(source.to_dict())

    assert restored.scenario_id == "TC-46"
    assert restored.turn_budget_exceeded is True
    # Older persisted rows without the flag must still deserialize as False.
    legacy_dict = source.to_dict()
    legacy_dict.pop("turn_budget_exceeded", None)
    legacy = ScenarioResult.from_dict(legacy_dict)
    assert legacy.turn_budget_exceeded is False
