"""``--perf-only`` stores its measurements, and the stored row reads back.

The goldens pin the exact row handed to ``persist_run``. This goes one step
further, through real SQLite, to the queries ``history`` and ``export`` are
built on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.test_mode_run_golden import _args, _context, _target
from tool_eval_bench.runner.throughput import ThroughputSample


def _cell(**overrides: Any) -> ThroughputSample:
    values: dict[str, Any] = {
        "pp_tokens": 2048,
        "tg_tokens": 128,
        "depth": 4096,
        "concurrency": 1,
        "ttft_ms": 480.25,
        "total_ms": 1830.5,
        "pp_tps": 4264.4,
        "tg_tps": 94.81,
        "requested_pp": 2048,
        "requested_depth": 4096,
        "calibration_confidence": "llama-benchy",
        "pp_estimated": True,
    }
    values.update(overrides)
    return ThroughputSample(**values)


def test_stored_perf_only_results_read_back_through_history_and_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tool_eval_bench.application import run_queries
    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.cli.history import print_history
    from tool_eval_bench.cli.leaderboard import export_runs

    # RunRepository puts the database under ./data, so this isolates it.
    monkeypatch.chdir(tmp_path)
    ok = [_cell(), _cell(concurrency=4, pp_tps=9100.0, tg_tps=310.2, pp_estimated=False)]
    failed = ThroughputSample(
        concurrency=8,
        requested_pp=2048,
        error="Server error '503' for url 'http://user:secret@gpu-box.internal:8000/v1'",
    )
    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: [ok[0], failed, ok[1]])
    target = _target(_args(tmp_path / "runs", "--perf-only"), _context())

    # A failed cell fails the run, after the row is stored.
    with pytest.raises(SystemExit):
        dispatch._run_throughput_mode(target)

    [listed] = run_queries.recent_runs()
    stored = run_queries.get_run(listed["run_id"])
    assert stored is not None
    scores = stored["scores"]
    assert scores == listed["scores"]
    assert (scores["samples"], scores["successful"], scores["failed"]) == (3, 2, 1)
    assert scores["results"] == [sample.to_result() for sample in ok]
    # Raw, not the report's rounding.
    assert scores["results"][0]["tg_tps"] == 94.81
    assert scores["results"][0]["pp_estimated"] is True
    # The failed cell is a count only, so its error text never reaches the row.
    assert "secret" not in json.dumps(stored)
    assert all(result["concurrency"] != 8 for result in scores["results"])

    console = Console(record=True, width=200)
    print_history(console)
    assert listed["run_id"] in console.export_text()

    capsys.readouterr()
    # Export ranks tool-eval runs only; a perf row is skipped, not an error.
    export_runs(Console(record=True), fmt="json")
    assert json.loads(capsys.readouterr().out) == []


def test_to_result_holds_what_the_report_prints_and_nothing_unfilled() -> None:
    result = _cell(token_timestamps=[0.0, 0.5], draft_n=3).to_result()

    assert result == {
        "requested_pp": 2048,
        "requested_depth": 4096,
        "pp_tokens": 2048,
        "tg_tokens": 128,
        "depth": 4096,
        "concurrency": 1,
        "ttft_ms": 480.25,
        "total_ms": 1830.5,
        "pp_tps": 4264.4,
        "tg_tps": 94.81,
        "pp_estimated": True,
        "calibration_confidence": "llama-benchy",
    }
