"""Tests for CLI leaderboard display and export functions.

Covers cli/leaderboard.py which was at 53% coverage. Tests the display
and export functions using mocked RunRepository.
"""

from __future__ import annotations

import csv
import io
import json
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from tool_eval_bench.cli.leaderboard import (
    _extract_leaderboard_rows,
    _rating_short,
    _score_bg,
    _score_color,
    _shorten_model_name,
    export_runs,
    print_leaderboard,
)

# ===========================================================================
# Fixtures
# ===========================================================================


def _make_run(
    model: str = "test-model",
    final_score: float = 85.0,
    rating: str = "★★★★ Good",
    total_points: int = 100,
    max_points: int = 126,
    cat_scores: list | None = None,
    scenario_results: list | None = None,
    total_tokens: int = 5000,
    num_runs: int = 1,
    scenario_count: int = 69,
) -> dict:
    return {
        "model": model,
        "run_id": f"run_{model}",
        "created_at": "2026-04-19T12:00:00",
        "config": {"scenario_count": scenario_count, "backend": "vllm"},
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
            "total_tokens": total_tokens,
            "safety_warnings": [],
        },
    }


# ===========================================================================
# _score_color
# ===========================================================================


class TestScoreColor:
    """Tests for the _score_color helper."""

    def test_excellent(self) -> None:
        assert _score_color(95) == "bold green"

    def test_good(self) -> None:
        assert _score_color(80) == "green"

    def test_adequate(self) -> None:
        assert _score_color(65) == "yellow"

    def test_weak(self) -> None:
        assert _score_color(45) == "red"

    def test_poor(self) -> None:
        assert _score_color(20) == "bold red"

    def test_boundary_90(self) -> None:
        assert _score_color(90) == "bold green"

    def test_boundary_75(self) -> None:
        assert _score_color(75) == "green"

    def test_boundary_60(self) -> None:
        assert _score_color(60) == "yellow"

    def test_boundary_40(self) -> None:
        assert _score_color(40) == "red"


# ===========================================================================
# _score_bg
# ===========================================================================


class TestScoreBg:
    """Tests for the _score_bg helper."""

    def test_excellent_format(self) -> None:
        result = _score_bg(95)
        assert "95" in result
        assert "bold green" in result

    def test_weak_format(self) -> None:
        result = _score_bg(30)
        assert "30" in result
        assert "bold red" in result

    def test_adequate_format(self) -> None:
        result = _score_bg(65)
        assert "65" in result
        assert "yellow" in result


# ===========================================================================
# _rating_short
# ===========================================================================


class TestRatingShort:
    """Tests for the _rating_short helper."""

    def test_excellent(self) -> None:
        assert "★★★★★" in _rating_short("★★★★★ Excellent")

    def test_good(self) -> None:
        assert "★★★★" in _rating_short("★★★★ Good")

    def test_safety_capped(self) -> None:
        assert "ⓢ" in _rating_short("★★★ Adequate (safety-capped)")

    def test_adequate(self) -> None:
        assert "★★★" in _rating_short("★★★ Adequate")

    def test_weak(self) -> None:
        assert "★★" in _rating_short("★★ Weak")

    def test_poor(self) -> None:
        assert "★" in _rating_short("★ Poor")


# ===========================================================================
# print_leaderboard
# ===========================================================================


class TestPrintLeaderboard:
    """Tests for the print_leaderboard CLI sub-command."""

    def test_no_runs_shows_message(self) -> None:
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = []
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "No benchmark runs found" in output

    def test_shows_ranked_models(self) -> None:
        runs = [
            _make_run(model="strong", final_score=95),
            _make_run(model="weak", final_score=40),
            _make_run(model="mid", final_score=70),
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "strong" in output
            assert "weak" in output
            assert "mid" in output

    def test_shows_medals(self) -> None:
        runs = [
            _make_run(model="gold", final_score=95),
            _make_run(model="silver", final_score=85),
            _make_run(model="bronze", final_score=75),
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "🥇" in output
            assert "🥈" in output
            assert "🥉" in output

    def test_shows_category_heatmap(self) -> None:
        runs = [
            _make_run(
                cat_scores=[{"category": "A", "percent": 100}, {"category": "B", "percent": 50}]
            )
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "Sel" in output  # Category A label
            assert "Prm" in output  # Category B label

    def test_shows_legend(self) -> None:
        runs = [_make_run()]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "P/F" in output
            assert "Config" in output
            assert "Scores:" in output
            assert "safety-capped" in output

    def test_shows_config_for_partial_runs(self) -> None:
        runs = [_make_run(scenario_count=15)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            # Config column shows backend/scenario_count
            assert "vllm/15" in output

    def test_shows_token_usage(self) -> None:
        runs = [_make_run(total_tokens=50000)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "50K" in output

    def test_hides_token_when_zero(self) -> None:
        runs = [_make_run(total_tokens=0)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "—" in output

    def test_shows_num_runs(self) -> None:
        runs = [
            _make_run(model="same-model", final_score=85),
            _make_run(model="same-model", final_score=80),
            _make_run(model="same-model", final_score=75),
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "3" in output  # 3 runs for same model

    def test_limits_rows(self) -> None:
        runs = [_make_run(model=f"model-{i}", final_score=float(i)) for i in range(100)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console, limit=5)

            output = console.file.getvalue()
            # Sorted by score DESC — top 5 are model-99..model-95
            for i in range(95, 100):
                assert f"model-{i}" in output
            # model-0 (lowest score) should NOT appear
            assert "model-0 " not in output

    def test_no_valid_results_shows_message(self) -> None:
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = []  # empty runs list
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            assert "No benchmark runs found" in output or "No valid results" in output

    def test_excludes_interrupted_and_incomplete_persisted_runs(self) -> None:
        interrupted = _make_run(model="interrupted", final_score=100)
        interrupted["status"] = "interrupted"
        incomplete = _make_run(model="incomplete", final_score=100)
        incomplete["status"] = "completed"
        incomplete["config"]["scenario_count"] = 4
        complete = _make_run(model="complete", final_score=75)
        complete["status"] = "completed"
        complete["config"]["scenario_count"] = 3

        rows = _extract_leaderboard_rows([interrupted, incomplete, complete])

        assert [row["model"] for row in rows] == ["complete"]
        assert rows[0]["completion_count"] == 3

    def test_persisted_retries_use_latest_not_best_score(self) -> None:
        first = _make_run(model="same", final_score=99)
        first.update({"status": "completed", "created_at": "2026-01-01T00:00:00"})
        first["config"]["scenario_count"] = 3
        first["config"]["config_fingerprint"] = "shared"
        latest = _make_run(model="same", final_score=60)
        latest.update({"status": "completed", "created_at": "2026-01-02T00:00:00"})
        latest["config"]["scenario_count"] = 3
        latest["config"]["config_fingerprint"] = "shared"

        rows = _extract_leaderboard_rows([first, latest])

        assert rows[0]["final_score"] == 60

    def test_excludes_completed_run_with_infrastructure_incompletion(self) -> None:
        incomplete = _make_run(model="incomplete", final_score=100)
        incomplete["status"] = "completed"
        incomplete["config"]["scenario_count"] = 3
        incomplete["scores"]["completion_rate"] = 67

        assert _extract_leaderboard_rows([incomplete]) == []


# ===========================================================================
# export_runs
# ===========================================================================


class TestExportRuns:
    """Tests for the export_runs CLI sub-command."""

    def test_export_csv_format(self) -> None:
        runs = [_make_run(model="test", final_score=85)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            # export_runs prints CSV to stdout via print(), not to console
            import sys

            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                from tool_eval_bench.cli.leaderboard import export_runs

                export_runs(Console(file=StringIO(), width=200, no_color=True), fmt="csv")
            finally:
                sys.stdout = old_stdout

            output = buf.getvalue()
            assert "test" in output
            assert "85" in output

    def test_export_csv_has_headers(self) -> None:
        runs = [_make_run(model="test")]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            # Capture stdout
            import sys

            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()

            try:
                from tool_eval_bench.cli.leaderboard import export_runs

                export_runs(MagicMock(), fmt="csv")
            finally:
                sys.stdout = old_stdout

            csv_text = buf.getvalue()
            reader = csv.reader(io.StringIO(csv_text))
            headers = next(reader)
            assert "model" in headers
            assert "final_score" in headers
            assert "passes" in headers
            assert "rank" in headers

    def test_export_csv_has_data_rows(self) -> None:
        runs = [_make_run(model="test-model", final_score=85)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            import sys

            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()

            try:
                from tool_eval_bench.cli.leaderboard import export_runs

                export_runs(MagicMock(), fmt="csv")
            finally:
                sys.stdout = old_stdout

            csv_text = buf.getvalue()
            reader = csv.reader(io.StringIO(csv_text))
            rows = list(reader)
            assert len(rows) >= 2  # header + 1 data row

    def test_export_json_format(self) -> None:
        runs = [_make_run(model="test", final_score=85)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            import sys

            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()

            try:
                from tool_eval_bench.cli.leaderboard import export_runs

                export_runs(MagicMock(), fmt="json")
            finally:
                sys.stdout = old_stdout

            json_text = buf.getvalue()
            data = json.loads(json_text)
            assert len(data) >= 1
            assert data[0]["model"] == "test"
            assert data[0]["final_score"] == 85

    def test_export_json_has_categories(self) -> None:
        runs = [
            _make_run(
                cat_scores=[{"category": "A", "percent": 100}, {"category": "B", "percent": 50}]
            )
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            import sys

            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()

            try:
                from tool_eval_bench.cli.leaderboard import export_runs

                export_runs(MagicMock(), fmt="json")
            finally:
                sys.stdout = old_stdout

            json_text = buf.getvalue()
            data = json.loads(json_text)
            assert "categories" in data[0]
            assert data[0]["categories"]["A"] == 100
            assert data[0]["categories"]["B"] == 50

    def test_export_to_file_csv(self, tmp_path) -> None:
        runs = [_make_run(model="test")]
        output_file = str(tmp_path / "export.csv")
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            export_runs(console, fmt="csv", output=output_file)

            with open(output_file) as f:
                assert "model" in f.read()

    def test_export_to_file_json(self, tmp_path) -> None:
        runs = [_make_run(model="test")]
        output_file = str(tmp_path / "export.json")
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            export_runs(console, fmt="json", output=output_file)

            with open(output_file) as f:
                data = json.loads(f.read())
            assert len(data) >= 1

    def test_export_unknown_format_exits(self) -> None:
        runs = [_make_run()]  # Need at least one run to reach format check
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                try:
                    export_runs(console, fmt="xml")
                except SystemExit:
                    pass
                mock_exit.assert_called_once_with(1)

    def test_export_no_runs_shows_message(self) -> None:
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = []
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            export_runs(console, fmt="csv")

            output = console.file.getvalue()
            assert "No benchmark runs found" in output

    def test_export_csv_includes_category_columns(self) -> None:
        runs = [
            _make_run(
                cat_scores=[{"category": "A", "percent": 100}, {"category": "K", "percent": 75}]
            )
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            import sys

            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()

            try:
                from tool_eval_bench.cli.leaderboard import export_runs

                export_runs(MagicMock(), fmt="csv")
            finally:
                sys.stdout = old_stdout

            csv_text = buf.getvalue()
            assert "cat_A" in csv_text
            assert "cat_K" in csv_text


# ===========================================================================
# _shorten_model_name
# ===========================================================================


class TestShortenModelName:
    """Tests for the _shorten_model_name helper (#15)."""

    def test_hf_style_name_unchanged(self) -> None:
        assert _shorten_model_name("Qwen/Qwen3.6-35B-A3B-FP8") == "Qwen/Qwen3.6-35B-A3B-FP8"

    def test_bare_alias_unchanged(self) -> None:
        assert _shorten_model_name("gemma4") == "gemma4"

    def test_hf_cache_path(self) -> None:
        path = "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B/snapshots/abc123"
        assert _shorten_model_name(path) == "Qwen/Qwen3.6-35B"

    def test_hf_cache_path_two_parts(self) -> None:
        path = "/tmp/models--Meta--Llama-3-70B/main"
        assert _shorten_model_name(path) == "Meta/Llama-3-70B"

    def test_absolute_unix_path(self) -> None:
        path = "/data/models/Qwen/Qwen3.6-35B-A3B-FP8"
        assert _shorten_model_name(path) == "Qwen/Qwen3.6-35B-A3B-FP8"

    def test_absolute_single_component(self) -> None:
        path = "/models/my-model"
        assert _shorten_model_name(path) == "models/my-model"

    def test_whitespace_stripped(self) -> None:
        assert _shorten_model_name("  Qwen/Qwen3  ") == "Qwen/Qwen3"

    def test_empty_string(self) -> None:
        assert _shorten_model_name("") == ""

    def test_hf_cache_path_only_org(self) -> None:
        """Edge case: models--OrgOnly (no model part)."""
        path = "/cache/models--OrgOnly/snapshot"
        assert _shorten_model_name(path) == "OrgOnly"

    def test_windows_backslash_hf_cache(self) -> None:
        """Windows-style backslash path with models-- pattern."""
        path = r"C:\Users\user\.cache\huggingface\hub\models--Org--Model\snapshots\abc"
        assert _shorten_model_name(path) == "Org/Model"

    def test_relative_path_unchanged(self) -> None:
        """Relative path without leading / should pass through unchanged."""
        assert _shorten_model_name("local/model-name") == "local/model-name"

    def test_windows_drive_path(self) -> None:
        """Windows drive-letter path should be shortened to last two components."""
        path = r"C:\Users\user\models\my-model"
        assert _shorten_model_name(path) == "models/my-model"

    def test_windows_unc_path(self) -> None:
        r"""Windows UNC path (\\server\share\…) should be shortened."""
        path = r"\\server\share\models\Qwen-35B"
        assert _shorten_model_name(path) == "models/Qwen-35B"


class TestLeaderboardLongModelNames:
    """Integration test: long model names should appear shortened (#15)."""

    def test_long_path_shortened_in_output(self) -> None:
        long_path = "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B/snapshots/abc"
        runs = [_make_run(model=long_path, final_score=90)]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.list.return_value = runs
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)
            print_leaderboard(console)

            output = console.file.getvalue()
            # Shortened name should appear, full path should not
            assert "Qwen/Qwen3.6-35B" in output
            assert "huggingface" not in output
            assert "snapshots" not in output
