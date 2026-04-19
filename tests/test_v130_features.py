"""Tests for v1.3.0 features: leaderboard, export, judge, async tools, and arg bytes tracking."""

import csv
import io
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)


# ===========================================================================
# Leaderboard: _extract_leaderboard_rows
# ===========================================================================

class TestLeaderboardExtraction:
    """Tests for the leaderboard row extraction logic."""

    def _make_run(
        self,
        model: str = "test-model",
        final_score: float = 85.0,
        rating: str = "★★★★ Good",
        total_points: int = 100,
        max_points: int = 126,
        cat_scores: list | None = None,
        scenario_results: list | None = None,
    ) -> dict:
        return {
            "model": model,
            "run_id": f"run_{model}",
            "created_at": "2026-04-19T12:00:00",
            "config": {"scenario_count": 69, "backend": "vllm"},
            "scores": {
                "final_score": final_score,
                "rating": rating,
                "total_points": total_points,
                "max_points": max_points,
                "category_scores": cat_scores or [
                    {"category": "A", "percent": 100},
                    {"category": "B", "percent": 67},
                ],
                "scenario_results": scenario_results or [
                    {"status": "pass", "scenario_id": "TC-01"},
                    {"status": "partial", "scenario_id": "TC-02"},
                    {"status": "fail", "scenario_id": "TC-03"},
                ],
                "total_tokens": 5000,
                "safety_warnings": [],
            },
        }

    def test_single_model(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        runs = [self._make_run()]
        rows = _extract_leaderboard_rows(runs)
        assert len(rows) == 1
        assert rows[0]["model"] == "test-model"
        assert rows[0]["final_score"] == 85.0
        assert rows[0]["passes"] == 1
        assert rows[0]["partials"] == 1
        assert rows[0]["fails"] == 1
        assert rows[0]["num_runs"] == 1

    def test_multiple_models_sorted_by_score(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        runs = [
            self._make_run(model="weak", final_score=40),
            self._make_run(model="strong", final_score=95),
            self._make_run(model="mid", final_score=70),
        ]
        rows = _extract_leaderboard_rows(runs)
        assert len(rows) == 3
        assert rows[0]["model"] == "strong"
        assert rows[1]["model"] == "mid"
        assert rows[2]["model"] == "weak"

    def test_deduplicates_by_best_run(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        runs = [
            self._make_run(model="test", final_score=60),
            self._make_run(model="test", final_score=85),
            self._make_run(model="test", final_score=70),
        ]
        rows = _extract_leaderboard_rows(runs)
        assert len(rows) == 1
        assert rows[0]["final_score"] == 85
        assert rows[0]["num_runs"] == 3

    def test_category_scores_extracted(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        runs = [self._make_run(cat_scores=[
            {"category": "A", "percent": 100},
            {"category": "K", "percent": 50},
            {"category": "O", "percent": 83},
        ])]
        rows = _extract_leaderboard_rows(runs)
        assert rows[0]["cat_scores"]["A"] == 100
        assert rows[0]["cat_scores"]["K"] == 50
        assert rows[0]["cat_scores"]["O"] == 83

    def test_empty_runs(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        rows = _extract_leaderboard_rows([])
        assert rows == []


# ===========================================================================
# Leaderboard: color helpers
# ===========================================================================

class TestLeaderboardColors:
    """Tests for score-to-color mapping functions."""

    def test_score_color_excellent(self) -> None:
        from tool_eval_bench.cli.leaderboard import _score_color
        assert "green" in _score_color(95)

    def test_score_color_good(self) -> None:
        from tool_eval_bench.cli.leaderboard import _score_color
        assert "green" in _score_color(80)

    def test_score_color_adequate(self) -> None:
        from tool_eval_bench.cli.leaderboard import _score_color
        assert "yellow" in _score_color(65)

    def test_score_color_weak(self) -> None:
        from tool_eval_bench.cli.leaderboard import _score_color
        assert "red" in _score_color(45)

    def test_score_color_poor(self) -> None:
        from tool_eval_bench.cli.leaderboard import _score_color
        assert "red" in _score_color(20)

    def test_rating_short_all_variants(self) -> None:
        from tool_eval_bench.cli.leaderboard import _rating_short
        assert "★★★★★" in _rating_short("★★★★★ Excellent")
        assert "★★★★" in _rating_short("★★★★ Good")
        assert "★★★" in _rating_short("★★★ Adequate")
        assert "ⓢ" in _rating_short("★★★ Adequate (safety-capped)")
        assert "★★" in _rating_short("★★ Weak")
        assert "★" in _rating_short("★ Poor")


# ===========================================================================
# Export: CSV format
# ===========================================================================

class TestExportCSV:
    """Tests for CSV export logic."""

    def test_csv_output(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        runs = [{
            "model": "test-model",
            "run_id": "run_001",
            "created_at": "2026-04-19T12:00:00",
            "config": {"scenario_count": 69, "backend": "vllm"},
            "scores": {
                "final_score": 85,
                "rating": "Good",
                "total_points": 100,
                "max_points": 138,
                "category_scores": [{"category": "A", "percent": 100}],
                "scenario_results": [{"status": "pass", "scenario_id": "TC-01"}],
                "total_tokens": 5000,
                "safety_warnings": [],
            },
        }]
        rows = _extract_leaderboard_rows(runs)
        assert len(rows) == 1
        assert rows[0]["model"] == "test-model"

    def test_export_to_file(self) -> None:
        from rich.console import Console

        from tool_eval_bench.cli.leaderboard import export_runs

        console = Console(file=io.StringIO(), force_terminal=False)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmpfile = f.name

        # Patch RunRepository to return test data
        mock_runs = [{
            "model": "export-test",
            "run_id": "run_export",
            "created_at": "2026-04-19T12:00:00",
            "config": {"scenario_count": 69, "backend": "vllm"},
            "scores": {
                "final_score": 90,
                "rating": "Excellent",
                "total_points": 120,
                "max_points": 138,
                "category_scores": [{"category": "A", "percent": 100}],
                "scenario_results": [{"status": "pass", "scenario_id": "TC-01"}],
                "total_tokens": 3000,
                "safety_warnings": [],
            },
        }]

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = mock_runs
            MockRepo.return_value = mock_repo
            export_runs(console, fmt="csv", output=tmpfile)

        with open(tmpfile) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["model"] == "export-test"
        assert rows[0]["final_score"] == "90"

    def test_export_json_to_file(self) -> None:
        from rich.console import Console

        from tool_eval_bench.cli.leaderboard import export_runs

        console = Console(file=io.StringIO(), force_terminal=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmpfile = f.name

        mock_runs = [{
            "model": "json-test",
            "run_id": "run_json",
            "created_at": "2026-04-19T12:00:00",
            "config": {"scenario_count": 69, "backend": "vllm"},
            "scores": {
                "final_score": 75,
                "rating": "Good",
                "total_points": 100,
                "max_points": 138,
                "category_scores": [{"category": "A", "percent": 100}],
                "scenario_results": [{"status": "pass", "scenario_id": "TC-01"}],
                "total_tokens": 4000,
                "safety_warnings": [],
            },
        }]

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = mock_runs
            MockRepo.return_value = mock_repo
            export_runs(console, fmt="json", output=tmpfile)

        with open(tmpfile) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["model"] == "json-test"
        assert data[0]["final_score"] == 75
        assert "categories" in data[0]

    def test_export_no_runs(self) -> None:
        from rich.console import Console

        from tool_eval_bench.cli.leaderboard import export_runs

        console = Console(file=io.StringIO(), force_terminal=False)

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = []
            MockRepo.return_value = mock_repo
            # Should not crash
            export_runs(console, fmt="csv")


# ===========================================================================
# LLM-as-Judge: prompt building
# ===========================================================================

class TestJudgePromptBuilding:
    """Tests for judge prompt construction."""

    def test_build_judge_prompt_basic(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt
        from tool_eval_bench.domain.scenarios import ScenarioDefinition

        scenario = ScenarioDefinition(
            id="TC-01",
            title="Test Scenario",
            category=Category.A,
            user_message="What's the weather?",
            description="Call get_weather.",
            handle_tool_call=lambda s, c: {},
            evaluate=lambda s: None,
        )
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Did not call get_weather.",
        )
        state = ScenarioState()
        state.final_answer = "I think it's sunny."

        prompt = _build_judge_prompt(scenario, result, state)
        assert "TC-01" in prompt
        assert "Test Scenario" in prompt
        assert "What's the weather?" in prompt
        assert "Did not call get_weather" in prompt
        assert "I think it's sunny" in prompt
        assert "No tool calls made" in prompt

    def test_build_judge_prompt_with_tool_calls(self) -> None:
        from tool_eval_bench.runner.judge import _build_judge_prompt
        from tool_eval_bench.domain.scenarios import ScenarioDefinition, ToolResultRecord

        scenario = ScenarioDefinition(
            id="TC-07",
            title="Chained Calls",
            category=Category.C,
            user_message="Find contacts then email them.",
            description="Chain get_contacts → send_email.",
            handle_tool_call=lambda s, c: {},
            evaluate=lambda s: None,
        )
        result = ScenarioResult(
            scenario_id="TC-07",
            status=ScenarioStatus.FAIL,
            points=0,
            summary="Called wrong tools.",
        )
        state = ScenarioState()
        state.tool_calls.append(ToolCallRecord(
            id="call_1", name="web_search",
            raw_arguments='{"query": "contacts"}',
            arguments={"query": "contacts"}, turn=1,
        ))
        state.tool_results.append(ToolResultRecord(
            call_id="call_1", name="web_search",
            result={"results": [{"snippet": "No contacts found"}]},
        ))
        state.final_answer = "I searched but found nothing."

        prompt = _build_judge_prompt(scenario, result, state)
        assert "web_search" in prompt
        assert "contacts" in prompt
        assert "No tool calls made" not in prompt


# ===========================================================================
# LLM-as-Judge: verdict handling
# ===========================================================================

class TestJudgeVerdicts:
    """Tests for judge result processing."""

    @pytest.mark.asyncio
    async def test_judge_skips_non_fail(self) -> None:
        from tool_eval_bench.runner.judge import judge_failed_scenarios
        from tool_eval_bench.domain.scenarios import ScenarioDefinition

        scenario = ScenarioDefinition(
            id="TC-01", title="Test", category=Category.A,
            user_message="test", description="test",
            handle_tool_call=lambda s, c: {}, evaluate=lambda s: None,
        )
        pass_result = ScenarioResult(
            scenario_id="TC-01", status=ScenarioStatus.PASS,
            points=2, summary="ok",
        )
        partial_result = ScenarioResult(
            scenario_id="TC-02", status=ScenarioStatus.PARTIAL,
            points=1, summary="partial ok",
        )

        adapter = AsyncMock()
        updated = await judge_failed_scenarios(
            adapter,
            model="test",
            base_url="http://localhost",
            scenarios=[scenario],
            results=[pass_result, partial_result],
            states={},
        )
        assert len(updated) == 2
        assert updated[0].status == ScenarioStatus.PASS
        assert updated[1].status == ScenarioStatus.PARTIAL
        # Adapter should not be called since there are no FAILs
        adapter.chat_completion.assert_not_called()


# ===========================================================================
# AsyncToolExecutor
# ===========================================================================

class TestAsyncToolExecutor:
    """Tests for the experimental async tool executor."""

    def test_register_and_start(self) -> None:
        from tool_eval_bench.runner.async_tools import (
            AsyncToolExecutor,
            AsyncToolSpec,
            AsyncToolStatus,
        )

        executor = AsyncToolExecutor()
        spec = AsyncToolSpec(
            tool_name="search_files",
            duration_ms=1000.0,
            final_result={"results": []},
        )
        executor.register_tool(spec)

        result = executor.start_tool("search_files")
        assert result.status == AsyncToolStatus.PENDING
        assert result.handle.startswith("async_search_files_")
        assert result.progress_percent == 0.0

    def test_unregistered_tool_completes_immediately(self) -> None:
        from tool_eval_bench.runner.async_tools import AsyncToolExecutor, AsyncToolStatus

        executor = AsyncToolExecutor()
        result = executor.start_tool("unknown_tool")
        assert result.status == AsyncToolStatus.COMPLETED
        assert "not registered" in str(result.result.get("error", ""))

    def test_cancel_tool(self) -> None:
        from tool_eval_bench.runner.async_tools import (
            AsyncToolExecutor,
            AsyncToolSpec,
            AsyncToolStatus,
        )

        executor = AsyncToolExecutor()
        executor.register_tool(AsyncToolSpec(
            tool_name="search_files",
            duration_ms=10000.0,
            final_result={},
        ))

        started = executor.start_tool("search_files")
        cancelled = executor.cancel_tool(started.handle)
        assert cancelled.status == AsyncToolStatus.CANCELLED

    def test_format_async_status_pending(self) -> None:
        from tool_eval_bench.runner.async_tools import (
            AsyncToolResult,
            AsyncToolStatus,
            format_async_status,
        )

        result = AsyncToolResult(
            status=AsyncToolStatus.PENDING,
            handle="async_test_1",
            progress_percent=0.0,
        )
        formatted = format_async_status(result)
        data = json.loads(formatted)
        assert data["status"] == "pending"
        assert data["handle"] == "async_test_1"

    def test_format_async_status_completed(self) -> None:
        from tool_eval_bench.runner.async_tools import (
            AsyncToolResult,
            AsyncToolStatus,
            format_async_status,
        )

        result = AsyncToolResult(
            status=AsyncToolStatus.COMPLETED,
            handle="async_test_2",
            result={"data": "hello"},
        )
        formatted = format_async_status(result)
        data = json.loads(formatted)
        assert data["status"] == "completed"
        assert data["result"]["data"] == "hello"

    def test_format_async_status_failed(self) -> None:
        from tool_eval_bench.runner.async_tools import (
            AsyncToolResult,
            AsyncToolStatus,
            format_async_status,
        )

        result = AsyncToolResult(
            status=AsyncToolStatus.FAILED,
            handle="async_test_3",
            error="Timeout exceeded",
        )
        formatted = format_async_status(result)
        data = json.loads(formatted)
        assert data["status"] == "failed"
        assert "Timeout" in data["error"]

    def test_format_async_status_unknown(self) -> None:
        from tool_eval_bench.runner.async_tools import format_async_status

        # This should not crash even with a weird mock
        class FakeResult:
            status = "something_weird"
            handle = "x"

        formatted = format_async_status(FakeResult())
        assert "unknown" in formatted

    def test_example_specs(self) -> None:
        from tool_eval_bench.runner.async_tools import create_example_async_specs

        specs = create_example_async_specs()
        assert len(specs) == 3
        names = {s.tool_name for s in specs}
        assert names == {"search_files", "run_code", "web_search"}
        # web_search spec should simulate failure
        ws_spec = next(s for s in specs if s.tool_name == "web_search")
        assert ws_spec.simulate_failure is True


# ===========================================================================
# tool_call_arg_bytes tracking
# ===========================================================================

class TestArgBytesTracking:
    """Tests for per-tool-call argument size tracking."""

    def test_arg_bytes_in_scenario_result(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            tool_call_arg_bytes=150,
        )
        assert result.tool_call_arg_bytes == 150

    def test_arg_bytes_in_to_dict(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            tool_call_arg_bytes=250,
        )
        d = result.to_dict()
        assert d["tool_call_arg_bytes"] == 250

    def test_arg_bytes_zero_not_in_dict(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
            tool_call_arg_bytes=0,
        )
        d = result.to_dict()
        assert "tool_call_arg_bytes" not in d

    def test_arg_bytes_default_zero(self) -> None:
        result = ScenarioResult(
            scenario_id="TC-01",
            status=ScenarioStatus.PASS,
            points=2,
            summary="ok",
        )
        assert result.tool_call_arg_bytes == 0


# ===========================================================================
# Category O registration
# ===========================================================================

class TestCategoryORegistration:
    """Verify Category O is fully integrated into the system."""

    def test_category_enum_has_O(self) -> None:
        assert hasattr(Category, "O")
        assert Category.O.value == "O"

    def test_category_label_defined(self) -> None:
        from tool_eval_bench.domain.scenarios import CATEGORY_LABELS
        assert Category.O in CATEGORY_LABELS
        assert CATEGORY_LABELS[Category.O] == "Structured Output"

    def test_display_color_defined(self) -> None:
        from tool_eval_bench.cli.display import CATEGORY_COLORS
        assert Category.O in CATEGORY_COLORS

    def test_structured_scenarios_in_all(self) -> None:
        from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
        cat_o = [s for s in ALL_SCENARIOS if s.category == Category.O]
        assert len(cat_o) == 6
        ids = {s.id for s in cat_o}
        assert ids == {"TC-64", "TC-65", "TC-66", "TC-67", "TC-68", "TC-69"}

    def test_display_details_present(self) -> None:
        from tool_eval_bench.evals.scenarios import ALL_DISPLAY_DETAILS
        for tc_id in ("TC-64", "TC-65", "TC-66", "TC-67", "TC-68", "TC-69"):
            assert tc_id in ALL_DISPLAY_DETAILS, f"Missing display details for {tc_id}"

    def test_leaderboard_labels_include_O(self) -> None:
        from tool_eval_bench.cli.leaderboard import _CAT_LABELS, _CAT_FULL
        assert "O" in _CAT_LABELS
        assert "O" in _CAT_FULL
        assert _CAT_LABELS["O"] == "Out"
        assert _CAT_FULL["O"] == "Structured Output"

    def test_tc68_has_no_response_format(self) -> None:
        """TC-68 tests MODEL restraint (refusing extra fields).
        If response_format is set, the SERVER enforces the constraint,
        making the test trivially passable."""
        from tool_eval_bench.evals.scenarios_structured import STRUCTURED_SCENARIOS
        tc68 = next(s for s in STRUCTURED_SCENARIOS if s.id == "TC-68")
        assert tc68.response_format_override is None, (
            "TC-68 must NOT use response_format — it tests model restraint, "
            "not server enforcement"
        )

    def test_other_structured_scenarios_have_response_format(self) -> None:
        """TC-64, 65, 66, 67, 69 should all have response_format_override."""
        from tool_eval_bench.evals.scenarios_structured import STRUCTURED_SCENARIOS
        for s in STRUCTURED_SCENARIOS:
            if s.id == "TC-68":
                continue  # TC-68 intentionally omits it
            assert s.response_format_override is not None, (
                f"{s.id} should have response_format_override"
            )

    def test_schemas_embedded_in_user_messages(self) -> None:
        """All Category O user messages should contain the actual JSON schema
        text so models see it even if the backend ignores response_format."""
        from tool_eval_bench.evals.scenarios_structured import STRUCTURED_SCENARIOS
        for s in STRUCTURED_SCENARIOS:
            assert "Schema:" in s.user_message, (
                f"{s.id} user message should embed the schema"
            )
            assert '"type"' in s.user_message, (
                f"{s.id} user message should contain schema body"
            )


# ===========================================================================
# Version consistency
# ===========================================================================

class TestVersionConsistency:
    """Ensure version strings are consistent across the project."""

    def test_init_version(self) -> None:
        from tool_eval_bench import __version__
        assert __version__ == "1.3.0"

    def test_pyproject_version_matches(self) -> None:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            assert data["project"]["version"] == "1.3.0"
