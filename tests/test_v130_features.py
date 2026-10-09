"""Tests for v1.3.0 features: leaderboard, export, and arg bytes tracking."""

import csv
import io
import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioResult,
    ScenarioStatus,
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
                "category_scores": cat_scores
                or [
                    {"category": "A", "percent": 100},
                    {"category": "B", "percent": 67},
                ],
                "scenario_results": scenario_results
                or [
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

        runs = [
            self._make_run(
                cat_scores=[
                    {"category": "A", "percent": 100},
                    {"category": "K", "percent": 50},
                    {"category": "O", "percent": 83},
                ]
            )
        ]
        rows = _extract_leaderboard_rows(runs)
        assert rows[0]["cat_scores"]["A"] == 100
        assert rows[0]["cat_scores"]["K"] == 50
        assert rows[0]["cat_scores"]["O"] == 83

    def test_empty_runs(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        rows = _extract_leaderboard_rows([])
        assert rows == []

    def test_separates_runs_with_different_config_fingerprints(self) -> None:
        from tool_eval_bench.cli.leaderboard import _extract_leaderboard_rows

        first = self._make_run(model="same", final_score=90)
        first["config"]["config_fingerprint"] = "first"
        second = self._make_run(model="same", final_score=70)
        second["config"]["config_fingerprint"] = "second"

        rows = _extract_leaderboard_rows([first, second])

        assert len(rows) == 2


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

        runs = [
            {
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
            }
        ]
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
        mock_runs = [
            {
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
            }
        ]

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = mock_runs
            MockRepo.return_value = mock_repo
            export_runs(console, fmt="csv", output=tmpfile)

        with open(tmpfile, encoding="utf-8") as f:
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

        mock_runs = [
            {
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
            }
        ]

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.list.return_value = mock_runs
            MockRepo.return_value = mock_repo
            export_runs(console, fmt="json", output=tmpfile)

        with open(tmpfile, encoding="utf-8") as f:
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
        from tool_eval_bench.cli.leaderboard import _CAT_FULL, _CAT_LABELS

        assert "O" in _CAT_LABELS
        assert "O" in _CAT_FULL
        assert _CAT_LABELS["O"] == "Out"
        assert _CAT_FULL["O"] == "Structured Output"

    def test_tc68_has_no_response_format(self) -> None:
        """TC-68 tests MODEL restraint (refusing extra fields).
        If response_format is set, the SERVER enforces the constraint,
        making the test trivially passable."""
        from tool_eval_bench.evals.scenarios import STRUCTURED_SCENARIOS

        tc68 = next(s for s in STRUCTURED_SCENARIOS if s.id == "TC-68")
        assert tc68.response_format_override is None, (
            "TC-68 must NOT use response_format — it tests model restraint, not server enforcement"
        )

    def test_other_structured_scenarios_have_response_format(self) -> None:
        """TC-64, 65, 66, 67, 69 should all have response_format_override."""
        from tool_eval_bench.evals.scenarios import STRUCTURED_SCENARIOS

        for s in STRUCTURED_SCENARIOS:
            if s.id == "TC-68":
                continue  # TC-68 intentionally omits it
            assert s.response_format_override is not None, (
                f"{s.id} should have response_format_override"
            )

    def test_schemas_embedded_in_user_messages(self) -> None:
        """All Category O user messages should contain the actual JSON schema
        text so models see it even if the backend ignores response_format."""
        from tool_eval_bench.evals.scenarios import STRUCTURED_SCENARIOS

        for s in STRUCTURED_SCENARIOS:
            assert "Schema:" in s.user_message, f"{s.id} user message should embed the schema"
            assert '"type"' in s.user_message, f"{s.id} user message should contain schema body"


# ===========================================================================
# Version consistency
# ===========================================================================


class TestVersionConsistency:
    """The reported version must identify the code that produced a run."""

    def test_init_version_is_pep440(self) -> None:
        import re

        from tool_eval_bench import __version__

        # Release builds look like 2.2.0; builds from a commit look like
        # 2.2.1.dev11+g528272d — both must parse, neither may be empty.
        assert re.fullmatch(r"\d+\.\d+\.\d+(\.[a-z0-9]+)*(\+[a-zA-Z0-9.]+)?", __version__), (
            f"Not a PEP 440 version: {__version__}"
        )

    def test_version_is_derived_from_git_not_hardcoded(self) -> None:
        """A hardcoded version makes every dev build claim to be the last release."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("running from an installed distribution")
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        assert "version" in data["project"]["dynamic"]
        assert "version" not in data["project"]
        assert "setuptools_scm" in data["tool"]

    def test_non_release_builds_carry_their_commit(self) -> None:
        from tool_eval_bench import __version__

        if ".dev" not in __version__:
            pytest.skip("this is a tagged release build")
        assert "+g" in __version__, f"Dev build must identify its commit, got {__version__}"
