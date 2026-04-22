"""Tests for Category O — Structured Output scenarios (TC-64 to TC-69)."""

import json

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)


def _make_state(
    tool_calls: list[dict] | None = None,
    tool_results: list[dict] | None = None,
    final_answer: str = "",
    assistant_messages: list[str] | None = None,
) -> ScenarioState:
    state = ScenarioState()
    state.final_answer = final_answer
    state.assistant_messages = assistant_messages or ([final_answer] if final_answer else [])
    if tool_calls:
        for tc in tool_calls:
            state.tool_calls.append(
                ToolCallRecord(
                    id=tc.get("id", f"call_{len(state.tool_calls)}"),
                    name=tc["name"],
                    raw_arguments="{}",
                    arguments=tc.get("arguments", {}),
                    turn=tc.get("turn", 1),
                )
            )
    if tool_results:
        for tr in tool_results:
            state.tool_results.append(
                ToolResultRecord(
                    call_id=tr.get("call_id", "call_0"),
                    name=tr.get("name", "unknown"),
                    result=tr.get("result"),
                )
            )
    return state


def _get_scenario(sc_id: str):
    from tool_eval_bench.evals.scenarios_structured import STRUCTURED_SCENARIOS
    return next(s for s in STRUCTURED_SCENARIOS if s.id == sc_id)


# ==========================================================================
# TC-64: Simple Schema Compliance
# ==========================================================================

class TestTC64SimpleSchema:
    scenario = _get_scenario("TC-64")

    def test_pass_valid_json(self) -> None:
        data = {
            "title": "The Matrix",
            "year": 1999,
            "rating": 8.7,
            "genre": "sci-fi",
            "summary": "A computer hacker learns about the true nature of reality.",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_pass_json_in_code_fence(self) -> None:
        data = {
            "title": "The Matrix",
            "year": 1999,
            "rating": 8.7,
            "genre": "sci-fi",
            "summary": "A computer hacker learns about the true nature of reality.",
        }
        state = _make_state(final_answer=f"```json\n{json.dumps(data, indent=2)}\n```")
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_not_json(self) -> None:
        state = _make_state(final_answer="The Matrix is a great movie. I'd give it an 8/10.")
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_fail_used_tools(self) -> None:
        data = {"title": "The Matrix", "year": 1999, "rating": 8.7, "genre": "sci-fi", "summary": "Great."}
        state = _make_state(
            tool_calls=[{"name": "web_search", "arguments": {"query": "matrix review"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_partial_missing_field(self) -> None:
        data = {"title": "The Matrix", "year": 1999, "rating": 8.7, "genre": "sci-fi"}
        state = _make_state(final_answer=json.dumps(data))
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_partial_invalid_genre(self) -> None:
        data = {
            "title": "The Matrix",
            "year": 1999,
            "rating": 8.7,
            "genre": "cyberpunk",  # not in enum
            "summary": "A computer hacker learns about reality.",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_partial_extra_fields(self) -> None:
        data = {
            "title": "The Matrix",
            "year": 1999,
            "rating": 8.7,
            "genre": "sci-fi",
            "summary": "A great film.",
            "director": "Wachowskis",  # not allowed
        }
        state = _make_state(final_answer=json.dumps(data))
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL


# ==========================================================================
# TC-65: Tool → Structured Output
# ==========================================================================

class TestTC65ToolToStructured:
    scenario = _get_scenario("TC-65")

    def test_pass_correct_flow(self) -> None:
        data = {
            "location": "Tokyo",
            "temperature_celsius": 28,
            "condition": "Sunny",
            "recommendation": "Wear light, breathable clothing.",
        }
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_no_tool_call(self) -> None:
        data = {
            "location": "Tokyo",
            "temperature_celsius": 28,
            "condition": "Sunny",
            "recommendation": "Wear light clothes.",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_partial_wrong_temp(self) -> None:
        data = {
            "location": "Tokyo",
            "temperature_celsius": 25,  # wrong — tool returned 28
            "condition": "Sunny",
            "recommendation": "Wear light clothes.",
        }
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL


# ==========================================================================
# TC-66: Nested Schema
# ==========================================================================

class TestTC66NestedSchema:
    scenario = _get_scenario("TC-66")

    def test_pass_correct(self) -> None:
        data = {
            "query": "engineering",
            "total": 3,
            "contacts": [
                {"name": "Alice Zhang", "email": "alice.zhang@company.com", "department": "Engineering"},
                {"name": "Bob Martinez", "email": "bob.martinez@company.com", "department": "Design"},
                {"name": "Carol Singh", "email": "carol.singh@company.com", "department": "Engineering"},
            ],
        }
        state = _make_state(
            tool_calls=[{"name": "get_contacts", "arguments": {"query": "engineering"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_no_tool(self) -> None:
        state = _make_state(final_answer='{"query": "eng", "total": 0, "contacts": []}')
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_partial_wrong_total(self) -> None:
        data = {
            "query": "engineering",
            "total": 5,  # doesn't match array length
            "contacts": [
                {"name": "Alice Zhang", "email": "alice.zhang@company.com", "department": "Engineering"},
                {"name": "Bob Martinez", "email": "bob.martinez@company.com", "department": "Design"},
                {"name": "Carol Singh", "email": "carol.singh@company.com", "department": "Engineering"},
            ],
        }
        state = _make_state(
            tool_calls=[{"name": "get_contacts", "arguments": {"query": "engineering"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL


# ==========================================================================
# TC-67: Enum Constraint + Analysis
# ==========================================================================

class TestTC67EnumConstraint:
    scenario = _get_scenario("TC-67")

    def test_pass_correct(self) -> None:
        data = {
            "ticker": "NVDA",
            "price": 892.50,
            "currency": "USD",
            "signal": "buy",
            "reasoning": "Strong revenue growth of 265% YoY driven by AI demand with analyst targets above current price.",
        }
        state = _make_state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_no_tool(self) -> None:
        state = _make_state(final_answer="NVDA looks good.")
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_partial_invalid_signal(self) -> None:
        data = {
            "ticker": "NVDA",
            "price": 892.50,
            "currency": "USD",
            "signal": "moderate_buy",  # not in enum
            "reasoning": "Looking good but volatile.",
        }
        state = _make_state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL


# ==========================================================================
# TC-68: Schema Violation Resistance
# ==========================================================================

class TestTC68ViolationResistance:
    scenario = _get_scenario("TC-68")

    def test_pass_no_extra_fields(self) -> None:
        data = {
            "task_id": "PROJ-127",
            "status": "in_progress",
            "assignee": "me",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_partial_extra_fields(self) -> None:
        data = {
            "task_id": "PROJ-127",
            "status": "in_progress",
            "assignee": "me",
            "priority": "high",
            "due_date": "2026-04-30",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_fail_not_json(self) -> None:
        state = _make_state(final_answer="Task PROJ-127 is in progress.")
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL


# ==========================================================================
# TC-69: Multi-Tool → Complex Schema
# ==========================================================================

class TestTC69MultiToolComplex:
    scenario = _get_scenario("TC-69")

    def test_pass_correct(self) -> None:
        data = {
            "date": "2026-04-19",
            "weather": {
                "location": "San Francisco",
                "temperature": 18,
                "condition": "Foggy",
            },
            "market": {
                "ticker": "AAPL",
                "price": 192.30,
                "direction": "down",
            },
            "action_items": [
                "Bring a jacket — fog expected in SF.",
                "Monitor AAPL position — stock dropped 1.11%.",
            ],
        }
        state = _make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "San Francisco"}},
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
            ],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PASS

    def test_fail_missing_tool(self) -> None:
        data = {
            "date": "2026-04-19",
            "weather": {"location": "SF", "temperature": 18, "condition": "Foggy"},
            "market": {"ticker": "AAPL", "price": 192.30, "direction": "down"},
            "action_items": ["Check weather"],
        }
        state = _make_state(
            tool_calls=[{"name": "get_weather", "arguments": {"location": "SF"}}],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.FAIL

    def test_partial_wrong_direction(self) -> None:
        data = {
            "date": "2026-04-19",
            "weather": {"location": "San Francisco", "temperature": 18, "condition": "Foggy"},
            "market": {"ticker": "AAPL", "price": 192.30, "direction": "up"},  # wrong
            "action_items": ["Check weather"],
        }
        state = _make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "San Francisco"}},
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
            ],
            final_answer=json.dumps(data),
        )
        result = self.scenario.evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL


# ==========================================================================
# Issue #5 regression: unhashable type 'list' crash
# ==========================================================================
# When a model returns a list value for an enum field (e.g. "genre": ["sci-fi"])
# the evaluators previously crashed with TypeError instead of returning PARTIAL.

class TestIssue5ListValuedEnums:
    """Verify list-valued enum fields produce PARTIAL, not TypeError."""

    def test_tc64_genre_as_list(self) -> None:
        """TC-64: 'genre' is a list instead of a string."""
        data = {
            "title": "The Matrix", "year": 1999, "rating": 9.0,
            "genre": ["sci-fi", "action"],  # list, not string
            "summary": "A classic movie about reality.",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = _get_scenario("TC-64").evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL
        assert "genre" in result.summary

    def test_tc67_signal_as_list(self) -> None:
        """TC-67: 'signal' is a list instead of a string."""
        data = {
            "ticker": "NVDA", "price": 892.50, "currency": "USD",
            "signal": ["buy"],  # list, not string
            "reasoning": "Strong AI demand with record revenue growth.",
        }
        state = _make_state(
            tool_calls=[{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}],
            final_answer=json.dumps(data),
        )
        result = _get_scenario("TC-67").evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_tc68_status_as_list(self) -> None:
        """TC-68: 'status' is a list instead of a string."""
        data = {
            "task_id": "PROJ-127",
            "status": ["in_progress"],  # list, not string
            "assignee": "me",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = _get_scenario("TC-68").evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_tc69_direction_as_list(self) -> None:
        """TC-69: 'direction' is a list instead of a string."""
        data = {
            "date": "2026-04-19",
            "weather": {"location": "San Francisco", "temperature": 18, "condition": "Foggy"},
            "market": {"ticker": "AAPL", "price": 192.30, "direction": ["down"]},
            "action_items": ["Check weather"],
        }
        state = _make_state(
            tool_calls=[
                {"name": "get_weather", "arguments": {"location": "San Francisco"}},
                {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
            ],
            final_answer=json.dumps(data),
        )
        result = _get_scenario("TC-69").evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL

    def test_tc64_genre_as_int(self) -> None:
        """Non-string scalar (int) also handled gracefully."""
        data = {
            "title": "The Matrix", "year": 1999, "rating": 9.0,
            "genre": 42,  # int, not string
            "summary": "A classic movie.",
        }
        state = _make_state(final_answer=json.dumps(data))
        result = _get_scenario("TC-64").evaluate(state)
        assert result.status == ScenarioStatus.PARTIAL


class TestOrchestratorEvaluatorSafetyNet:
    """Verify the orchestrator catches evaluator exceptions (issue #5)."""

    def test_evaluator_crash_returns_fail(self) -> None:
        """An evaluator that raises should produce a FAIL, not crash the run."""
        import asyncio
        from unittest.mock import AsyncMock

        from tool_eval_bench.adapters.base import ChatCompletionResult
        from tool_eval_bench.domain.scenarios import Category, ScenarioDefinition
        from tool_eval_bench.runner.orchestrator import run_scenario

        def exploding_evaluator(state):
            raise RuntimeError("intentional evaluator crash")

        scenario = ScenarioDefinition(
            id="TC-TEST", title="Exploding Evaluator", category=Category.A,
            user_message="test prompt", description="test",
            handle_tool_call=lambda s, c: {}, evaluate=exploding_evaluator,
        )

        mock_result = ChatCompletionResult(
            content="some answer", tool_calls=[], raw_response={}, elapsed_ms=100.0,
        )
        adapter = AsyncMock()
        adapter.chat_completion = AsyncMock(return_value=mock_result)

        result = asyncio.run(run_scenario(
            adapter, model="m", base_url="http://x", api_key=None, scenario=scenario,
        ))

        assert result.status == ScenarioStatus.FAIL
        assert "Evaluator error" in result.summary
        assert "intentional evaluator crash" in result.summary

    def test_unhashable_evaluator_returns_fail(self) -> None:
        """The original issue #5 crash path: TypeError in evaluator."""
        import asyncio
        from unittest.mock import AsyncMock

        from tool_eval_bench.adapters.base import ChatCompletionResult
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        from tool_eval_bench.runner.orchestrator import run_scenario

        tc64 = next(s for s in ALL_SCENARIOS if s.id == "TC-64")

        # Model returns genre as list — was crashing the entire run
        mock_result = ChatCompletionResult(
            content=json.dumps({
                "title": "The Matrix", "year": 1999, "rating": 9.0,
                "genre": ["sci-fi", "action"], "summary": "A classic.",
            }),
            tool_calls=[], raw_response={}, elapsed_ms=100.0,
        )
        adapter = AsyncMock()
        adapter.chat_completion = AsyncMock(return_value=mock_result)

        result = asyncio.run(run_scenario(
            adapter, model="m", base_url="http://x", api_key=None, scenario=tc64,
        ))

        # Should be PARTIAL (not a crash), because the fix in the evaluator
        # itself handles list values.  The safety net is a second layer.
        assert result.status == ScenarioStatus.PARTIAL
        assert "genre" in result.summary

