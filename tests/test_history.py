"""Tests for CLI history sub-commands: print_history, print_diff, compare_runs.

Covers the entire history module (cli/history.py) which was at 3% coverage.
Uses mocked RunRepository to avoid needing a live database.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from tool_eval_bench.cli.history import compare_runs, print_diff, print_history
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus


def _make_repo_mock(runs: list[dict] | None = None) -> MagicMock:
    """Create a MagicMock that mimics RunRepository with given runs."""
    repo = MagicMock()
    runs = runs or []
    repo.list.return_value = runs
    if runs:
        repo.get_latest.return_value = runs[0]
    else:
        repo.get_latest.return_value = None
    return repo


# ===========================================================================
# print_history
# ===========================================================================


class TestPrintHistory:
    """Tests for the print_history CLI sub-command."""

    def test_no_runs_shows_message(self) -> None:
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock([])
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)

            print_history(console)

            repo.list.assert_called_once_with(limit=15)
            output = console.file.getvalue()
            assert "No previous runs found" in output

    def test_shows_recent_runs(self) -> None:
        runs = [
            {
                "run_id": "2026-04-20T12-00-00Z_abc123",
                "model": "qwen3-35b",
                "scores": {"final_score": 85, "rating": "★★★★ Good"},
                "created_at": "2026-04-20T12:00:00",
            },
            {
                "run_id": "2026-04-19T10-00-00Z_def456",
                "model": "llama3-70b",
                "scores": {"final_score": 72, "rating": "★★★ Adequate"},
                "created_at": "2026-04-19T10:00:00",
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(runs)
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)

            print_history(console)

            repo.list.assert_called_once_with(limit=15)
            output = console.file.getvalue()
            assert "qwen3-35b" in output
            assert "llama3-70b" in output
            assert "85" in output
            assert "72" in output

    def test_truncates_created_at_to_datetime(self) -> None:
        runs = [
            {
                "run_id": "2026-04-20T12-00-00Z_abc123",
                "model": "test-model",
                "scores": {"final_score": 50},
                "created_at": "2026-04-20T12:00:00.123456+00:00",
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(runs)
            MockRepo.return_value = repo

            console = Console(file=StringIO(), width=200, no_color=True)

            print_history(console)

            # Should show truncated datetime (first 19 chars)
            output = console.file.getvalue()
            assert "2026-04-20T12:00:00" in output


# ===========================================================================
# print_diff
# ===========================================================================


class TestPrintDiff:
    """Tests for the print_diff CLI sub-command."""

    def test_latest_resolves_to_actual_id(self) -> None:
        """print_diff with 'latest' should resolve to the actual run ID."""
        prev_runs = [
            {
                "run_id": "2026-04-19T10-00-00Z_abc123",
                "model": "test-model",
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "points": 2, "status": "pass"},
                        {"scenario_id": "TC-02", "points": 0, "status": "fail"},
                    ]
                },
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(prev_runs)
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="pass"),
                ScenarioResult(scenario_id="TC-02", status=ScenarioStatus.PASS, points=2, summary="improved"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "latest")

            repo.get_scenario_results.assert_called_once_with("2026-04-19T10-00-00Z_abc123")

    def test_latest_no_previous_runs(self) -> None:
        """print_diff with 'latest' when no runs exist should show message."""
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock([])
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="pass"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "latest")

            output = console.file.getvalue()
            assert "No previous runs found" in output

    def test_run_not_found(self) -> None:
        """print_diff with non-existent run ID should show message."""
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock()
            repo.get_scenario_results.return_value = None
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="pass"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "nonexistent_id")

            output = console.file.getvalue()
            assert "not found" in output

    def test_improved_scenario_shows_green(self) -> None:
        """A scenario that improved from 0 to 2 should show improved."""
        prev_runs = [
            {
                "run_id": "2026-04-19T10-00-00Z_abc123",
                "model": "test-model",
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "points": 0, "status": "fail"},
                    ]
                },
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(prev_runs)
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="pass"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "2026-04-19T10-00-00Z_abc123")

            output = console.file.getvalue()
            assert "improved" in output

    def test_regressed_scenario_shows_red(self) -> None:
        """A scenario that regressed from 2 to 0 should show regressed."""
        prev_runs = [
            {
                "run_id": "2026-04-19T10-00-00Z_abc123",
                "model": "test-model",
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "points": 2, "status": "pass"},
                    ]
                },
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(prev_runs)
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.FAIL, points=0, summary="fail"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "2026-04-19T10-00-00Z_abc123")

            output = console.file.getvalue()
            assert "regressed" in output

    def test_unchanged_scenario_shows_dim(self) -> None:
        """A scenario that stayed the same should show unchanged."""
        prev_runs = [
            {
                "run_id": "2026-04-19T10-00-00Z_abc123",
                "model": "test-model",
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "points": 1, "status": "partial"},
                    ]
                },
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(prev_runs)
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PARTIAL, points=1, summary="partial"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "2026-04-19T10-00-00Z_abc123")

            output = console.file.getvalue()
            assert "=" in output

    def test_new_scenario_detected(self) -> None:
        """A scenario in current results but not in previous should show as new."""
        prev_runs = [
            {
                "run_id": "2026-04-19T10-00-00Z_abc123",
                "model": "test-model",
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "points": 2, "status": "pass"},
                    ]
                },
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(prev_runs)
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="pass"),
                ScenarioResult(scenario_id="TC-99", status=ScenarioStatus.PASS, points=2, summary="new scenario"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "2026-04-19T10-00-00Z_abc123")

            output = console.file.getvalue()
            assert "new scenario" in output

    def test_summary_shows_total_delta(self) -> None:
        """Summary line should show total points delta."""
        prev_runs = [
            {
                "run_id": "2026-04-19T10-00-00Z_abc123",
                "model": "test-model",
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "points": 0, "status": "fail"},
                    ]
                },
            },
        ]
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = _make_repo_mock(prev_runs)
            MockRepo.return_value = repo


            current_results = [
                ScenarioResult(scenario_id="TC-01", status=ScenarioStatus.PASS, points=2, summary="pass"),
            ]

            console = Console(file=StringIO(), width=200, no_color=True)
            print_diff(console, current_results, "2026-04-19T10-00-00Z_abc123")

            output = console.file.getvalue()
            assert "Points:" in output
            assert "0 → 2" in output
            assert "+2" in output


# ===========================================================================
# compare_runs
# ===========================================================================


class TestCompareRuns:
    """Tests for the compare_runs CLI sub-command."""

    def test_both_runs_found_and_compared(self) -> None:
        """compare_runs should resolve both run IDs and show comparison."""
        run_a = {
            "run_id": "run_a_id",
            "config": {"model": "model-a"},
            "scores": {
                "final_score": 70,
                "scenario_results": [
                    {"scenario_id": "TC-01", "points": 2, "status": "pass"},
                    {"scenario_id": "TC-02", "points": 0, "status": "fail"},
                ],
            },
        }
        run_b = {
            "run_id": "run_b_id",
            "config": {"model": "model-b"},
            "scores": {
                "final_score": 85,
                "scenario_results": [
                    {"scenario_id": "TC-01", "points": 2, "status": "pass"},
                    {"scenario_id": "TC-02", "points": 2, "status": "pass"},
                ],
            },
        }

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get.side_effect = lambda rid: run_a if rid == "run_a_id" else run_b
            repo.get_latest.return_value = None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            compare_runs(console, "run_a_id", "run_b_id")

            repo.get.assert_any_call("run_a_id")
            repo.get.assert_any_call("run_b_id")
            output = console.file.getvalue()
            assert "model-a" in output
            assert "model-b" in output
            assert "70" in output
            assert "85" in output

    def test_run_a_not_found_exits(self) -> None:
        """compare_runs should exit with error if run A is not found."""
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get.return_value = None
            repo.get_latest.return_value = None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            # Should call sys.exit(1)
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                with pytest.raises(SystemExit):
                    compare_runs(console, "missing_a", "run_b_id")
                mock_exit.assert_called_once_with(1)

    def test_run_b_not_found_exits(self) -> None:
        """compare_runs should exit with error if run B is not found."""
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get.side_effect = [
                {"run_id": "run_a_id", "config": {}, "scores": {}},
                None,
            ]
            repo.get_latest.return_value = None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                with pytest.raises(SystemExit):
                    compare_runs(console, "run_a_id", "missing_b")
                mock_exit.assert_called_once_with(1)

    def test_latest_shorthand_for_run_a(self) -> None:
        """compare_runs should resolve 'latest' to actual run ID for run A."""
        latest_run = {
            "run_id": "latest_run_id",
            "config": {"model": "model-latest"},
            "scores": {
                "final_score": 90,
                "scenario_results": [{"scenario_id": "TC-01", "points": 2, "status": "pass"}],
            },
        }
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get_latest.return_value = latest_run
            repo.get.side_effect = lambda rid: {"run_id": rid, "config": {}, "scores": {}} if rid == "run_b_id" else None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            with patch("sys.exit"):
                compare_runs(console, "latest", "run_b_id")

            repo.get_latest.assert_called_once()

    def test_latest_shorthand_for_run_b(self) -> None:
        """compare_runs should resolve 'latest' to actual run ID for run B."""
        latest_run = {
            "run_id": "latest_run_id",
            "config": {"model": "model-latest"},
            "scores": {
                "final_score": 90,
                "scenario_results": [{"scenario_id": "TC-01", "points": 2, "status": "pass"}],
            },
        }
        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get_latest.return_value = latest_run
            repo.get.side_effect = lambda rid: {"run_id": rid, "config": {}, "scores": {}} if rid == "run_a_id" else None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            with patch("sys.exit"):
                compare_runs(console, "run_a_id", "latest")

            repo.get_latest.assert_called()

    def test_no_scenario_results_shows_error(self) -> None:
        """compare_runs should show error if one or both runs have no scenario results."""
        run_a = {"run_id": "a", "config": {}, "scores": {"scenario_results": []}}
        run_b = {"run_id": "b", "config": {}, "scores": {"final_score": 50}}

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get.side_effect = lambda rid: run_a if rid == "a" else run_b
            repo.get_latest.return_value = None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            compare_runs(console, "a", "b")

            output = console.file.getvalue()
            assert "no scenario results" in output

    def test_scenario_removed_in_b_shows_removed(self) -> None:
        """A scenario in A but not in B should show as removed."""
        run_a = {
            "run_id": "a",
            "config": {},
            "scores": {
                "scenario_results": [
                    {"scenario_id": "TC-01", "points": 2, "status": "pass"},
                ],
            },
        }
        run_b = {
            "run_id": "b",
            "config": {},
            "scores": {
                "scenario_results": [
                    {"scenario_id": "TC-99", "points": 0, "status": "fail"},
                ],
            },
        }

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get.side_effect = lambda rid: run_a if rid == "a" else run_b
            repo.get_latest.return_value = None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            compare_runs(console, "a", "b")

            output = console.file.getvalue()
            assert "removed in B" in output

    def test_scenario_new_in_b_shows_new(self) -> None:
        """A scenario in B but not in A should show as new."""
        run_a = {
            "run_id": "a",
            "config": {},
            "scores": {
                "scenario_results": [
                    {"scenario_id": "TC-01", "points": 0, "status": "fail"},
                ],
            },
        }
        run_b = {
            "run_id": "b",
            "config": {},
            "scores": {
                "scenario_results": [
                    {"scenario_id": "TC-99", "points": 2, "status": "pass"},
                ],
            },
        }

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get.side_effect = lambda rid: run_a if rid == "a" else run_b
            repo.get_latest.return_value = None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            compare_runs(console, "a", "b")

            output = console.file.getvalue()
            assert "new in B" in output

    def test_summary_shows_score_and_points_delta(self) -> None:
        """Summary should show both points and final score delta."""
        run_a = {
            "run_id": "a",
            "config": {},
            "scores": {
                "final_score": 60,
                "scenario_results": [
                    {"scenario_id": "TC-01", "points": 0, "status": "fail"},
                ],
            },
        }
        run_b = {
            "run_id": "b",
            "config": {},
            "scores": {
                "final_score": 90,
                "scenario_results": [
                    {"scenario_id": "TC-01", "points": 2, "status": "pass"},
                ],
            },
        }

        with patch("tool_eval_bench.storage.db.RunRepository") as MockRepo:
            repo = MagicMock()
            repo.get.side_effect = lambda rid: run_a if rid == "a" else run_b
            repo.get_latest.return_value = None
            MockRepo.return_value = repo


            console = Console(file=StringIO(), width=200, no_color=True)
            compare_runs(console, "a", "b")

            output = console.file.getvalue()
            assert "Score:" in output
            assert "60 → 90" in output
            assert "Points:" in output
            assert "0 → 2" in output
