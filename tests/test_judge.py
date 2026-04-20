"""Tests for runner/judge.py — LLM-as-judge re-evaluation.

Covers runner/judge.py which was at 47% coverage. Uses a mock adapter
that returns deterministic JSON responses to test the judge pipeline.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tool_eval_bench.adapters.base import BackendAdapter, ChatCompletionResult
from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.runner.judge import judge_failed_scenarios


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_scenario(
    scenario_id: str = "TC-01",
    title: str = "Direct Specialist Match",
    category: Category = Category.A,
    user_message: str = "What's the weather in Berlin?",
    description: str = "Use get_weather for Berlin.",
) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=scenario_id,
        title=title,
        category=category,
        user_message=user_message,
        description=description,
        handle_tool_call=lambda state, call: None,
        evaluate=lambda state: ScenarioEvaluation(ScenarioStatus.FAIL, 0, "No tool call."),
    )


def _make_state(
    tool_calls: list[ToolCallRecord] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
    tool_results: list | None = None,
    meta: dict | None = None,
) -> ScenarioState:
    return ScenarioState(
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        assistant_messages=assistant_messages or [],
        final_answer=final_answer,
        meta=meta or {},
    )


def _make_result(
    scenario_id: str = "TC-01",
    status: ScenarioStatus = ScenarioStatus.FAIL,
    points: int = 0,
    summary: str = "No tool call made.",
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        status=status,
        points=points,
        summary=summary,
        tool_call_arg_bytes=0,
    )


# ===========================================================================
# Mock adapter for judge calls
# ===========================================================================


class JudgeMockAdapter(BackendAdapter):
    """Returns a deterministic JSON response for judge calls."""

    def __init__(self, verdict: str = "partial", reason: str = "Model showed intent.") -> None:
        self._verdict = verdict
        self._reason = reason
        self._call_count = 0
        self._last_messages: list[dict] | None = None

    async def chat_completion(self, **kwargs) -> ChatCompletionResult:
        self._call_count += 1
        self._last_messages = kwargs.get("messages", [])
        content = json.dumps({
            "verdict": self._verdict,
            "reason": self._reason,
        })
        return ChatCompletionResult(
            content=content,
            elapsed_ms=100.0,
        )


class JudgeMockAdapterCodeFence(JudgeMockAdapter):
    """Returns JSON wrapped in markdown code fences."""

    async def chat_completion(self, **kwargs) -> ChatCompletionResult:
        self._call_count += 1
        self._last_messages = kwargs.get("messages", [])
        content = "```json\n{\"verdict\": \"partial\", \"reason\": \"Model showed intent.\"}\n```"
        return ChatCompletionResult(
            content=content,
            elapsed_ms=100.0,
        )


class JudgeMockAdapterNonJson(JudgeMockAdapter):
    """Returns non-JSON response to test error handling."""

    async def chat_completion(self, **kwargs) -> ChatCompletionResult:
        self._call_count += 1
        return ChatCompletionResult(
            content="I think the model did okay but I'm not sure.",
            elapsed_ms=100.0,
        )


class JudgeMockAdapterUnexpectedVerdict(JudgeMockAdapter):
    """Returns an unexpected verdict value."""

    async def chat_completion(self, **kwargs) -> ChatCompletionResult:
        self._call_count += 1
        content = json.dumps({"verdict": "pass", "reason": "Actually this is a pass."})
        return ChatCompletionResult(
            content=content,
            elapsed_ms=100.0,
        )


class JudgeMockAdapterException(JudgeMockAdapter):
    """Raises an exception to test error handling."""

    async def chat_completion(self, **kwargs) -> ChatCompletionResult:
        raise RuntimeError("API connection failed")


# ===========================================================================
# _build_judge_prompt
# ===========================================================================


class TestBuildJudgePrompt:
    """Tests for the _build_judge_prompt function."""

    def test_includes_scenario_id(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario(scenario_id="TC-42", title="Extra Parameter Injection")
        result = _make_result(scenario_id="TC-42", summary="Called with extra param.")
        state = _make_state()

        prompt = _build_judge_prompt(scenario, result, state)

        assert "TC-42" in prompt
        assert "Extra Parameter Injection" in prompt

    def test_includes_user_message(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario(user_message="Send an email to alice@example.com.")
        result = _make_result()
        state = _make_state()

        prompt = _build_judge_prompt(scenario, result, state)

        assert "alice@example.com" in prompt

    def test_includes_expected_behavior(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario(description="Should call send_email with recipient.")
        result = _make_result()
        state = _make_state()

        prompt = _build_judge_prompt(scenario, result, state)

        assert "send_email" in prompt

    def test_includes_tool_calls_when_present(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario()
        result = _make_result()
        state = _make_state(
            tool_calls=[
                ToolCallRecord(
                    id="call_1",
                    name="send_email",
                    raw_arguments='{"to": "alice@example.com"}',
                    arguments={"to": "alice@example.com"},
                    turn=1,
                )
            ]
        )

        prompt = _build_judge_prompt(scenario, result, state)

        assert "send_email" in prompt
        assert "alice@example.com" in prompt
        assert "turn 1" in prompt

    def test_includes_tool_results_when_present(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario()
        result = _make_result()
        state = _make_state(
            tool_results=[
                MagicMock(name="send_email", result={"status": "sent"}, __repr__=lambda self: "ToolResult"),
            ]
        )

        prompt = _build_judge_prompt(scenario, result, state)

        assert "sent" in prompt

    def test_includes_final_answer(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario()
        result = _make_result()
        state = _make_state(final_answer="I sent the email to Alice.")

        prompt = _build_judge_prompt(scenario, result, state)

        assert "I sent the email to Alice" in prompt

    def test_includes_evaluator_reason(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario()
        result = _make_result(summary="Model did not call send_email.")
        state = _make_state()

        prompt = _build_judge_prompt(scenario, result, state)

        assert "Model did not call send_email" in prompt

    def test_long_tool_results_truncated(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt

        scenario = _make_scenario()
        result = _make_result()
        state = _make_state(
            tool_results=[
                MagicMock(name="read_file", result={"content": "x" * 600}),
            ]
        )

        prompt = _build_judge_prompt(scenario, result, state)

        # Should be truncated to ~500 chars
        assert len(prompt) < 2000


# ===========================================================================
# judge_failed_scenarios
# ===========================================================================


class TestJudgeFailedScenarios:
    """Tests for the judge_failed_scenarios async function."""

    @pytest.mark.asyncio
    async def test_upgrades_fail_to_partial(self) -> None:
        """FAIL results should be upgraded to PARTIAL when judge agrees."""
        adapter = JudgeMockAdapter(verdict="partial", reason="Model showed partial understanding.")

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
            judge_model="judge-model",
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.PARTIAL
        assert updated[0].points == 1
        assert "[judge override]" in updated[0].note

    @pytest.mark.asyncio
    async def test_preserves_already_passed(self) -> None:
        """PASS results should not be re-evaluated."""
        adapter = JudgeMockAdapter()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.PASS
        assert updated[0].points == 2

    @pytest.mark.asyncio
    async def test_preserves_already_partial(self) -> None:
        """PARTIAL results should not be re-evaluated."""
        adapter = JudgeMockAdapter()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.PARTIAL, points=1)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.PARTIAL
        assert updated[0].points == 1

    @pytest.mark.asyncio
    async def test_confirms_fail_when_judge_says_fail(self) -> None:
        """FAIL results should stay FAIL when judge disagrees with upgrade."""
        adapter = JudgeMockAdapter(verdict="fail", reason="Model was completely wrong.")

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.FAIL
        assert updated[0].points == 0

    @pytest.mark.asyncio
    async def test_handles_non_json_response(self) -> None:
        """Non-JSON judge response should preserve original FAIL."""
        adapter = JudgeMockAdapterNonJson()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.FAIL
        assert updated[0].points == 0

    @pytest.mark.asyncio
    async def test_handles_unexpected_verdict(self) -> None:
        """Unexpected verdict (e.g., 'pass') should preserve original FAIL."""
        adapter = JudgeMockAdapterUnexpectedVerdict()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.FAIL
        assert updated[0].points == 0

    @pytest.mark.asyncio
    async def test_handles_api_exception(self) -> None:
        """API exception should preserve original FAIL."""
        adapter = JudgeMockAdapterException()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.FAIL
        assert updated[0].points == 0

    @pytest.mark.asyncio
    async def test_handles_code_fence_json(self) -> None:
        """JSON wrapped in markdown code fences should be parsed correctly."""
        adapter = JudgeMockAdapterCodeFence()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_multiple_results(self) -> None:
        """Multiple FAIL results should all be re-evaluated."""
        adapter = JudgeMockAdapter(verdict="partial")

        scenario_01 = _make_scenario(scenario_id="TC-01")
        scenario_02 = _make_scenario(scenario_id="TC-02")
        scenario_03 = _make_scenario(scenario_id="TC-03")
        results = [
            _make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0),
            _make_result(scenario_id="TC-02", status=ScenarioStatus.FAIL, points=0),
            _make_result(scenario_id="TC-03", status=ScenarioStatus.PASS, points=2),
        ]
        states = {
            "TC-01": _make_state(),
            "TC-02": _make_state(),
            "TC-03": _make_state(),
        }

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario_01, scenario_02, scenario_03],
            results=results,
            states=states,
        )

        assert len(updated) == 3
        assert updated[0].status == ScenarioStatus.PARTIAL
        assert updated[1].status == ScenarioStatus.PARTIAL
        assert updated[2].status == ScenarioStatus.PASS

    @pytest.mark.asyncio
    async def test_mixed_results(self) -> None:
        """Mix of FAIL, PARTIAL, PASS should only re-evaluate FAILs."""
        adapter = JudgeMockAdapter(verdict="partial")

        scenario = _make_scenario(scenario_id="TC-01")
        results = [
            _make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0),
            _make_result(scenario_id="TC-02", status=ScenarioStatus.PARTIAL, points=1),
            _make_result(scenario_id="TC-03", status=ScenarioStatus.PASS, points=2),
        ]
        states = {
            "TC-01": _make_state(),
            "TC-02": _make_state(),
            "TC-03": _make_state(),
        }

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 3
        assert updated[0].status == ScenarioStatus.PARTIAL
        assert updated[1].status == ScenarioStatus.PARTIAL
        assert updated[2].status == ScenarioStatus.PASS

    @pytest.mark.asyncio
    async def test_preserves_tool_call_arg_bytes(self) -> None:
        """Upgraded results should preserve tool_call_arg_bytes."""
        adapter = JudgeMockAdapter(verdict="partial")

        scenario = _make_scenario(scenario_id="TC-01")
        result = _make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)
        result.tool_call_arg_bytes = 256
        results = [result]
        states = {"TC-01": _make_state()}

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert updated[0].tool_call_arg_bytes == 256

    @pytest.mark.asyncio
    async def test_uses_judge_model_not_main_model(self) -> None:
        """Judge should use judge_model parameter, not the main model."""
        adapter = JudgeMockAdapter()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-01": _make_state()}

        await judge_failed_scenarios(
            adapter,
            model="main-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
            judge_model="judge-model",
        )

        # The adapter's last_messages should have been called with judge_model
        assert adapter._last_messages is not None

    @pytest.mark.asyncio
    async def test_no_scenarios_returns_original(self) -> None:
        """Empty results list should return empty list."""
        adapter = JudgeMockAdapter()

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[],
            results=[],
            states={},
        )

        assert updated == []

    @pytest.mark.asyncio
    async def test_missing_scenario_state_preserves_result(self) -> None:
        """If scenario_id is missing from states dict, original result is preserved."""
        adapter = JudgeMockAdapter()

        scenario = _make_scenario(scenario_id="TC-01")
        results = [_make_result(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0)]
        states = {"TC-99": _make_state()}  # Missing TC-01

        updated = await judge_failed_scenarios(
            adapter,
            model="test-model",
            base_url="http://localhost:8080",
            scenarios=[scenario],
            results=results,
            states=states,
        )

        assert len(updated) == 1
        assert updated[0].status == ScenarioStatus.FAIL


# ===========================================================================
# _call_judge
# ===========================================================================


class TestCallJudge:
    """Tests for the internal _call_judge function."""

    @pytest.mark.asyncio
    async def test_returns_parsed_verdict(self) -> None:
        adapter = JudgeMockAdapter()
        scenario = _make_scenario()
        result = _make_result()
        state = _make_state()

        from tool_eval_bench.runner.judge import _call_judge
        judge_result = await _call_judge(
            adapter,
            judge_model="judge",
            base_url="http://localhost:8080",
            api_key=None,
            scenario=scenario,
            result=result,
            state=state,
        )

        assert judge_result is not None
        assert judge_result["verdict"] == "partial"
        assert "Model showed intent" in judge_result["reason"]

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self) -> None:
        adapter = JudgeMockAdapterException()
        scenario = _make_scenario()
        result = _make_result()
        state = _make_state()

        from tool_eval_bench.runner.judge import _call_judge
        judge_result = await _call_judge(
            adapter,
            judge_model="judge",
            base_url="http://localhost:8080",
            api_key=None,
            scenario=scenario,
            result=result,
            state=state,
        )

        assert judge_result is None

    @pytest.mark.asyncio
    async def test_calls_adapter_with_correct_params(self) -> None:
        adapter = JudgeMockAdapter()
        scenario = _make_scenario(scenario_id="TC-42")
        result = _make_result(scenario_id="TC-42", summary="Test summary")
        state = _make_state(final_answer="Test answer")

        from tool_eval_bench.runner.judge import _call_judge
        await _call_judge(
            adapter,
            judge_model="judge-model",
            base_url="http://example.com",
            api_key="test-key",
            scenario=scenario,
            result=result,
            state=state,
            timeout_seconds=30.0,
        )

        assert adapter._last_messages is not None
        assert len(adapter._last_messages) == 2  # system + user
        assert adapter._last_messages[0]["role"] == "system"
        assert adapter._last_messages[1]["role"] == "user"
