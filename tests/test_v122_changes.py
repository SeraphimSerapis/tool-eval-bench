"""Tests for v1.2.2 changes.

Covers:
  - TC-15 evaluator: flexible query matching (population + iceland)
  - Context pressure: updated _RESERVED_FOR_SCENARIO constant
  - _resolve_scenarios: --categories, --scenarios, --short filtering
  - --backend-kwargs: JSON parsing and deep-merge
  - --metrics-url: URL override in speculative decoding
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock

import pytest

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sc(sid: str):
    """Get a scenario by ID."""
    return next(s for s in ALL_SCENARIOS if s.id == sid)


def _state(tool_calls: list[dict] | None = None) -> ScenarioState:
    state = ScenarioState()
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
    return state


# ---------------------------------------------------------------------------
# TC-15: Flexible query matching
# ---------------------------------------------------------------------------


class TestTC15FlexibleQuery:
    sc = _sc("TC-15")

    def test_pass_exact_phrase(self) -> None:
        """Original phrasing should still pass."""
        s = _state(tool_calls=[
            {"name": "web_search", "arguments": {"query": "population of iceland"}},
            {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_reversed_word_order(self) -> None:
        """'Iceland population' (reversed order) should also pass."""
        s = _state(tool_calls=[
            {"name": "web_search", "arguments": {"query": "Iceland population 2026"}},
            {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_pass_mixed_case(self) -> None:
        """Case-insensitive matching should work."""
        s = _state(tool_calls=[
            {"name": "web_search", "arguments": {"query": "Population Iceland current"}},
            {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
        ])
        assert self.sc.evaluate(s).status == ScenarioStatus.PASS

    def test_fail_missing_iceland(self) -> None:
        """Query without 'iceland' should fail."""
        s = _state(tool_calls=[
            {"name": "web_search", "arguments": {"query": "population of europe"}},
            {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
        ])
        assert self.sc.evaluate(s).status != ScenarioStatus.PASS

    def test_fail_missing_population(self) -> None:
        """Query without 'population' should fail."""
        s = _state(tool_calls=[
            {"name": "web_search", "arguments": {"query": "iceland facts"}},
            {"name": "calculator", "arguments": {"expression": "372520 * 0.02"}},
        ])
        assert self.sc.evaluate(s).status != ScenarioStatus.PASS


# ---------------------------------------------------------------------------
# Context pressure: _RESERVED_FOR_SCENARIO constant
# ---------------------------------------------------------------------------


class TestReservedForScenario:
    def test_constant_value(self) -> None:
        """Reservation must be >= 8000 to cover LARGE_TOOLSET tool definitions."""
        from tool_eval_bench.runner.context_pressure import _RESERVED_FOR_SCENARIO
        assert _RESERVED_FOR_SCENARIO >= 8000

    def test_small_context_returns_zero(self) -> None:
        """Context too small for any fill should still return 0, not negative."""
        from tool_eval_bench.runner.context_pressure import compute_fill_budget
        fill = compute_fill_budget(8000, 0.75)
        assert fill == 0

    def test_budget_accounts_for_reservation(self) -> None:
        """Fill budget should respect the increased reservation."""
        from tool_eval_bench.runner.context_pressure import (
            _RESERVED_FOR_OUTPUT,
            _RESERVED_FOR_SCENARIO,
            compute_fill_budget,
        )
        ctx = 32768
        fill = compute_fill_budget(ctx, 1.0)
        available = ctx - _RESERVED_FOR_OUTPUT - _RESERVED_FOR_SCENARIO
        assert fill == available
        assert fill < ctx - 12000  # Must leave substantial room


# ---------------------------------------------------------------------------
# _resolve_scenarios
# ---------------------------------------------------------------------------


class TestResolveScenarios:
    def _args(self, **kwargs) -> argparse.Namespace:
        defaults = {"short": False, "scenarios": None, "categories": None}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_all_scenarios_default(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        result = _resolve_scenarios(self._args())
        assert len(result) == len(ALL_SCENARIOS)

    def test_short_flag(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        from tool_eval_bench.evals.scenarios import SCENARIOS
        result = _resolve_scenarios(self._args(short=True))
        assert len(result) == len(SCENARIOS)
        assert len(result) < len(ALL_SCENARIOS)

    def test_categories_single(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        result = _resolve_scenarios(self._args(categories=["K"]))
        assert all(s.category == Category.K for s in result)
        assert len(result) > 0

    def test_categories_multiple(self) -> None:
        from tool_eval_bench.cli.bench import _resolve_scenarios
        result = _resolve_scenarios(self._args(categories=["A", "B"]))
        cats = {s.category for s in result}
        assert cats == {Category.A, Category.B}

    def test_categories_lowercase(self) -> None:
        """Should accept lowercase category letters."""
        from tool_eval_bench.cli.bench import _resolve_scenarios
        result = _resolve_scenarios(self._args(categories=["k", "a"]))
        cats = {s.category for s in result}
        assert cats == {Category.K, Category.A}

    def test_scenarios_override_categories(self) -> None:
        """--scenarios takes priority over --categories."""
        from tool_eval_bench.cli.bench import _resolve_scenarios
        result = _resolve_scenarios(self._args(
            scenarios=["TC-01"],
            categories=["K"],
        ))
        assert len(result) == 1
        assert result[0].id == "TC-01"

    def test_categories_with_short(self) -> None:
        """--categories should filter from --short base when both set."""
        from tool_eval_bench.cli.bench import _resolve_scenarios
        result = _resolve_scenarios(self._args(short=True, categories=["A"]))
        assert all(s.category == Category.A for s in result)
        assert len(result) > 0

    def test_empty_categories_returns_empty(self) -> None:
        """Non-existent category should return empty list (not crash)."""
        from tool_eval_bench.cli.bench import _resolve_scenarios
        # Category Z doesn't exist, but categories are uppercased
        # and filtered — no match means empty list
        result = _resolve_scenarios(self._args(categories=["Z"]))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# --backend-kwargs: JSON parsing
# ---------------------------------------------------------------------------


class TestBackendKwargsParsing:
    def test_valid_json_merges(self) -> None:
        """Valid JSON dict should merge into extra_params."""
        import json
        bk = json.loads('{"temperature": 0.6, "top_p": 0.9}')
        extra_params: dict = {}
        for k, v in bk.items():
            extra_params[k] = v
        assert extra_params == {"temperature": 0.6, "top_p": 0.9}

    def test_deep_merge_dict_values(self) -> None:
        """Dict-valued keys should be deep-merged."""
        import json
        extra_params = {"chat_template_kwargs": {"enable_thinking": False}}
        bk = json.loads('{"chat_template_kwargs": {"max_thinking_tokens": 1024}}')
        for k, v in bk.items():
            if isinstance(v, dict) and isinstance(extra_params.get(k), dict):
                extra_params[k].update(v)
            else:
                extra_params[k] = v
        assert extra_params == {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "max_thinking_tokens": 1024,
            }
        }

    def test_override_scalar(self) -> None:
        """Scalar values in --backend-kwargs should override individual flags."""
        import json
        extra_params = {"top_p": 0.9}
        bk = json.loads('{"top_p": 0.5}')
        for k, v in bk.items():
            extra_params[k] = v
        assert extra_params["top_p"] == 0.5


# ---------------------------------------------------------------------------
# --metrics-url: threading through speculative module
# ---------------------------------------------------------------------------


class TestMetricsUrlOverride:
    def test_metrics_url_auto(self) -> None:
        """Without override, _metrics_url derives from base_url."""
        from tool_eval_bench.runner.speculative import _metrics_url
        assert _metrics_url("http://localhost:8080/v1") == "http://localhost:8080/metrics"
        assert _metrics_url("http://host:4000") == "http://host:4000/metrics"

    def test_metrics_url_strips_v1(self) -> None:
        """Should strip /v1 before appending /metrics."""
        from tool_eval_bench.runner.speculative import _metrics_url
        assert _metrics_url("http://host:8080/v1/") == "http://host:8080/metrics"

    @pytest.mark.asyncio
    async def test_scrape_uses_override(self) -> None:
        """scrape_spec_metrics should use metrics_url when provided."""
        from tool_eval_bench.runner.speculative import scrape_spec_metrics

        client = AsyncMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "spec_decode_num_accepted_tokens_total 100\nspec_decode_num_draft_tokens_total 200\n"
        client.get = AsyncMock(return_value=resp)

        result = await scrape_spec_metrics(
            client, "http://litellm:4000",
            metrics_url="http://vllm:8080/metrics",
        )
        # Should have called the override URL, not litellm:4000/metrics
        client.get.assert_called_once()
        called_url = client.get.call_args[0][0]
        assert called_url == "http://vllm:8080/metrics"
        assert result is not None
        assert result.accepted_tokens == 100.0

    @pytest.mark.asyncio
    async def test_detect_uses_override(self) -> None:
        """detect_spec_decoding should use metrics_url when provided."""
        from tool_eval_bench.runner.speculative import detect_spec_decoding

        client = AsyncMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "spec_decode_num_accepted_tokens_total 50\n"
        client.get = AsyncMock(return_value=resp)

        info = await detect_spec_decoding(
            client, "http://litellm:4000",
            metrics_url="http://vllm:8080/metrics",
        )
        assert info.active is True
        assert info.has_prometheus is True
        called_url = client.get.call_args[0][0]
        assert called_url == "http://vllm:8080/metrics"
