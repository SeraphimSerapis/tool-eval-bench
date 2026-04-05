"""Tests for CLI utilities, math parser edge cases, noise variation, and report generation.

TEST-03: CLI argument validation (_parse_int_list)
TEST-05: Property-based edge cases for parse_math_expression
CRED-03: Noise variation per payload
TEST-02: Report output verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tool_eval_bench.cli.bench import _parse_int_list
from tool_eval_bench.evals.helpers import (
    asks_for_clarification,
    contains_refusal,
    parse_math_expression,
)
from tool_eval_bench.evals.noise import _seed_from_payload, enrich_payload


# ---------------------------------------------------------------------------
# TEST-03: _parse_int_list
# ---------------------------------------------------------------------------


class TestParseIntList:
    def test_comma_separated(self) -> None:
        assert _parse_int_list("0,4096,8192") == [0, 4096, 8192]

    def test_space_separated(self) -> None:
        assert _parse_int_list("1 2 4") == [1, 2, 4]

    def test_mixed_separators(self) -> None:
        assert _parse_int_list("1, 2,4 8") == [1, 2, 4, 8]

    def test_single_value(self) -> None:
        assert _parse_int_list("42") == [42]

    def test_empty_string(self) -> None:
        assert _parse_int_list("") == []

    def test_whitespace_only(self) -> None:
        assert _parse_int_list("   ") == []

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_int_list("1,abc,3")


# ---------------------------------------------------------------------------
# TEST-05: Math parser edge cases
# ---------------------------------------------------------------------------


class TestMathParserEdgeCases:
    """Boundary and edge case tests for the AST-based safe math parser."""

    def test_empty_string(self) -> None:
        assert parse_math_expression("") is None

    def test_dot_only(self) -> None:
        assert parse_math_expression(".") is None

    def test_division_by_zero(self) -> None:
        # Should raise ZeroDivisionError internally, return None
        assert parse_math_expression("1/0") is None

    def test_very_long_number(self) -> None:
        result = parse_math_expression("9" * 100)
        assert result is not None
        assert result == float("9" * 100)

    def test_nested_parens(self) -> None:
        result = parse_math_expression("((((1+2))))")
        assert result == 3.0

    def test_floating_point_precision(self) -> None:
        result = parse_math_expression("0.1 + 0.2")
        assert result is not None
        assert abs(result - 0.3) < 1e-10

    def test_negative_number(self) -> None:
        result = parse_math_expression("-5")
        assert result == -5.0

    def test_modulo(self) -> None:
        result = parse_math_expression("10 % 3")
        assert result == 1.0

    def test_complex_expression(self) -> None:
        result = parse_math_expression("(2 + 3) * 4 - 6 / 2")
        assert result == 17.0

    def test_rejects_function_calls(self) -> None:
        assert parse_math_expression("__import__('os')") is None

    def test_rejects_attribute_access(self) -> None:
        assert parse_math_expression("a.b") is None

    def test_rejects_string_literals(self) -> None:
        assert parse_math_expression("'hello'") is None

    def test_rejects_comparison_operators(self) -> None:
        assert parse_math_expression("1 < 2") is None

    def test_commas_stripped(self) -> None:
        # 1,000 → 1000
        result = parse_math_expression("1,000 + 500")
        assert result == 1500.0


# ---------------------------------------------------------------------------
# Refusal and clarification detection tests
# ---------------------------------------------------------------------------


class TestRefusalDetection:
    """Ensure the expanded refusal detection covers edge cases from CRED-02."""

    def test_wont_refusal(self) -> None:
        assert contains_refusal("I won't do that because there's no tool for it.")

    def test_sorry_refusal(self) -> None:
        assert contains_refusal("I'm sorry, but I can't help with that request.")

    def test_unfortunately_refusal(self) -> None:
        assert contains_refusal("Unfortunately, email deletion is not available.")

    def test_no_tool_refusal(self) -> None:
        assert contains_refusal("There is no tool available for deleting files.")

    def test_positive_not_detected(self) -> None:
        assert not contains_refusal("The weather in Berlin is 8°C and overcast.")

    def test_empty_string(self) -> None:
        assert not contains_refusal("")


class TestClarificationDetection:
    def test_please_provide(self) -> None:
        assert asks_for_clarification("Please provide the recipient's email address.")

    def test_can_you_tell_me(self) -> None:
        assert asks_for_clarification("Can you tell me which format you'd like?")

    def test_need_more_info(self) -> None:
        assert asks_for_clarification("I need more information to proceed.")

    def test_positive_statement(self) -> None:
        assert not asks_for_clarification("The weather is sunny today.")


# ---------------------------------------------------------------------------
# CRED-03: Noise variation per payload
# ---------------------------------------------------------------------------


class TestNoiseVariation:
    """Verify that different payloads produce different noise values."""

    def test_different_weather_payloads_get_different_ids(self) -> None:
        p1 = enrich_payload("get_weather", {"temperature": 8, "condition": "overcast"})
        p2 = enrich_payload("get_weather", {"temperature": 22, "condition": "sunny"})

        # Same tool, different data → different request IDs
        assert p1["request_id"] != p2["request_id"]
        assert p1["station_id"] != p2["station_id"]

    def test_same_payload_produces_same_noise(self) -> None:
        """Determinism: identical input → identical output."""
        p1 = enrich_payload("get_weather", {"temperature": 8, "condition": "overcast"})
        p2 = enrich_payload("get_weather", {"temperature": 8, "condition": "overcast"})
        assert p1 == p2

    def test_search_payloads_vary(self) -> None:
        p1 = enrich_payload("web_search", {"results": [{"snippet": "result a"}]})
        p2 = enrich_payload("web_search", {"results": [{"snippet": "result b"}]})
        assert p1["request_id"] != p2["request_id"]

    def test_error_payloads_vary(self) -> None:
        e1 = enrich_payload("unknown_tool", {"error": "Tool X not relevant"})
        e2 = enrich_payload("unknown_tool_2", {"error": "Tool Y not relevant"})
        assert e1["request_id"] != e2["request_id"]
        assert e1["trace_id"] != e2["trace_id"]

    def test_seed_determinism(self) -> None:
        s1 = _seed_from_payload({"key": "value"}, "salt")
        s2 = _seed_from_payload({"key": "value"}, "salt")
        assert s1 == s2

    def test_seed_varies_with_salt(self) -> None:
        s1 = _seed_from_payload({"key": "value"}, "a")
        s2 = _seed_from_payload({"key": "value"}, "b")
        assert s1 != s2


# ---------------------------------------------------------------------------
# TEST-02: Report output verification
# ---------------------------------------------------------------------------


class TestMarkdownReporter:
    """Verify that reports are well-formed and contain expected content."""

    def test_report_contains_all_scenario_ids(self) -> None:
        from tool_eval_bench.domain.scenarios import (
            CategoryScore,
            ModelScoreSummary,
            ScenarioResult,
            ScenarioStatus,
        )
        from tool_eval_bench.storage.reports import MarkdownReporter

        sr = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="Correct tool used",
            raw_log="model→get_weather→result",
        )
        cs = CategoryScore(
            category=__import__("tool_eval_bench.domain.scenarios", fromlist=["Category"]).Category.A,
            label="Tool Selection",
            earned=2,
            max_points=2,
            percent=100,
        )
        summary = ModelScoreSummary(
            scenario_results=[sr],
            category_scores=[cs],
            final_score=100,
            total_points=2,
            max_points=2,
            rating="★★★★★ Excellent",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = MarkdownReporter(root=tmpdir)
            path = reporter.write_scenario_report("test-run-001", "test-model", summary)

            assert Path(path).exists()
            content = Path(path).read_text()

            # Must contain key sections
            assert "TC-01" in content
            assert "test-model" in content
            assert "Tool Selection" in content
            assert "100" in content  # score
            assert "✅" in content  # pass emoji

    def test_report_includes_safety_warnings(self) -> None:
        from tool_eval_bench.domain.scenarios import (
            Category,
            CategoryScore,
            ModelScoreSummary,
            ScenarioResult,
            ScenarioStatus,
        )
        from tool_eval_bench.storage.reports import MarkdownReporter

        sr = ScenarioResult(
            scenario_id="TC-34",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Obeyed injected instructions",
        )
        cs = CategoryScore(
            category=Category.K,
            label="Safety & Boundaries",
            earned=0,
            max_points=2,
            percent=0,
        )
        summary = ModelScoreSummary(
            scenario_results=[sr],
            category_scores=[cs],
            final_score=0,
            total_points=0,
            max_points=2,
            rating="★ Poor",
            safety_warnings=["TC-34 (Prompt Injection): Obeyed injected instructions"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = MarkdownReporter(root=tmpdir)
            path = reporter.write_scenario_report("test-safety-001", "test-model", summary)

            content = Path(path).read_text()
            assert "TC-34" in content
            assert "Safety" in content or "⚠" in content
            assert "❌" in content

    def test_report_path_structure(self) -> None:
        """Reports should be written to YYYY/MM/ subdirectory."""
        from tool_eval_bench.domain.scenarios import ModelScoreSummary
        from tool_eval_bench.storage.reports import MarkdownReporter

        summary = ModelScoreSummary(
            scenario_results=[], category_scores=[],
            final_score=0, total_points=0, max_points=0, rating="★ Poor",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = MarkdownReporter(root=tmpdir)
            path = reporter.write_scenario_report("run-id-test", "model", summary)

            rel = str(Path(path).relative_to(tmpdir))
            # Should be like 2026/04/run-id-test.md
            parts = rel.split("/")
            assert len(parts) == 3  # YYYY/MM/filename.md
            assert parts[2].endswith(".md")


# ---------------------------------------------------------------------------
# SEC-02: Reference date validation
# ---------------------------------------------------------------------------


class TestReferenceDateValidation:
    def test_invalid_date_raises(self) -> None:
        """Invalid --reference-date should raise ValueError, not silently fall through."""
        from unittest.mock import MagicMock
        from tool_eval_bench.runner.service import BenchmarkService

        service = BenchmarkService(
            repo=MagicMock(),
            reporter=MagicMock(),
        )

        with pytest.raises(ValueError, match="Invalid --reference-date"):
            import asyncio
            asyncio.run(service.run_benchmark(
                model="test",
                backend="vllm",
                base_url="http://localhost:9999",
                reference_date="not-a-date",
            ))


# ---------------------------------------------------------------------------
# METH-02: Bootstrap confidence intervals
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Test the bootstrap CI and trial aggregation functions."""

    def test_single_value_returns_point(self) -> None:
        from tool_eval_bench.cli.bench import _bootstrap_ci
        lo, hi = _bootstrap_ci([85.0])
        assert lo == 85.0
        assert hi == 85.0

    def test_identical_values(self) -> None:
        from tool_eval_bench.cli.bench import _bootstrap_ci
        lo, hi = _bootstrap_ci([90.0, 90.0, 90.0, 90.0])
        assert lo == 90.0
        assert hi == 90.0

    def test_ci_contains_mean(self) -> None:
        from tool_eval_bench.cli.bench import _bootstrap_ci
        from statistics import mean
        values = [80.0, 85.0, 90.0, 82.0, 88.0]
        lo, hi = _bootstrap_ci(values)
        m = mean(values)
        assert lo <= m <= hi

    def test_ci_narrows_with_low_variance(self) -> None:
        from tool_eval_bench.cli.bench import _bootstrap_ci
        narrow = _bootstrap_ci([89.0, 90.0, 91.0, 90.0, 90.0])
        wide = _bootstrap_ci([60.0, 80.0, 100.0, 70.0, 90.0])
        narrow_range = narrow[1] - narrow[0]
        wide_range = wide[1] - wide[0]
        assert narrow_range < wide_range

    def test_deterministic(self) -> None:
        """Same input → same CI (uses seeded RNG)."""
        from tool_eval_bench.cli.bench import _bootstrap_ci
        r1 = _bootstrap_ci([70.0, 80.0, 90.0])
        r2 = _bootstrap_ci([70.0, 80.0, 90.0])
        assert r1 == r2


class TestMedian:
    def test_odd(self) -> None:
        from tool_eval_bench.cli.bench import _median
        assert _median([1.0, 3.0, 5.0]) == 3.0

    def test_even(self) -> None:
        from tool_eval_bench.cli.bench import _median
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_single(self) -> None:
        from tool_eval_bench.cli.bench import _median
        assert _median([42.0]) == 42.0

