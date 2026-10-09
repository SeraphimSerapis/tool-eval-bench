"""Spec-bench stores its measurements, and the stored row reads back.

The goldens pin the exact row handed to ``persist_run``. These tests go one
step further, through real SQLite, to the queries ``history`` and ``export``
are built on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.test_mode_run_golden import _args, _context, _spec_samples, _target
from tool_eval_bench.runner.speculative import SpecDecodeSample


def _run_spec_bench(monkeypatch: pytest.MonkeyPatch, out: Path, samples: list[Any]) -> None:
    from tool_eval_bench.cli.dispatch import _run_spec_bench_mode
    from tool_eval_bench.runner import speculative

    async def fake_run(*args: Any, on_sample: Callable[..., Any], **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    monkeypatch.setattr(speculative, "run_spec_bench", fake_run)
    args = _args(out, "--spec-bench", "--depth", "0,4096", "--spec-prompts", "filler,code")
    assert _run_spec_bench_mode(_target(args, _context()))


def test_stored_spec_bench_results_read_back_through_history_and_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tool_eval_bench.application import run_queries
    from tool_eval_bench.cli.history import print_history
    from tool_eval_bench.cli.leaderboard import export_runs

    # RunRepository puts the database under ./data, so this isolates it.
    monkeypatch.chdir(tmp_path)
    ok = _spec_samples("response", 1)
    failed = SpecDecodeSample(
        depth=4096,
        prompt_type="structured",
        error="Server error '503' for url 'http://user:secret@gpu-box.internal:8000/v1'",
    )
    _run_spec_bench(monkeypatch, tmp_path / "runs", [ok[0], failed, ok[1]])

    [listed] = run_queries.recent_runs()
    stored = run_queries.get_run(listed["run_id"])
    assert stored is not None
    scores = stored["scores"]
    assert scores == listed["scores"]
    assert scores["samples"] == 2
    assert scores["failed"] == 1
    assert scores["results"] == [sample.to_result() for sample in ok]
    assert scores["results"][0]["effective_tg_tps"] == ok[0].effective_tg_tps
    # Failed samples are a count only, so their error text never reaches the row.
    assert "secret" not in json.dumps(stored)
    assert "structured" not in json.dumps(scores)

    console = Console(record=True, width=200)
    print_history(console)
    assert listed["run_id"] in console.export_text()

    capsys.readouterr()
    # Export ranks tool-eval runs only; a spec-bench row is skipped, not an error.
    export_runs(Console(record=True), fmt="json")
    assert json.loads(capsys.readouterr().out) == []


def test_to_result_keeps_derived_metrics_and_drops_per_step_arrays() -> None:
    sample = SpecDecodeSample(
        tg_tokens=100,
        ttft_ms=100.0,
        total_ms=1100.0,
        acceptance_rate=0.5,
        baseline_tg_tps=50.0,
        per_step_accepted=[2, 0],
        per_step_drafted=[2, 2],
        runs=2,
        acceptance_rate_range=(0.4, 0.6),
    )

    result = sample.to_result()

    assert result["effective_tg_tps"] == 100.0
    assert result["speedup_ratio"] == 2.0
    assert result["waste_ratio"] == 0.5
    assert result["per_position_acceptance"] == [0.5, 0.5]
    # A list, because the row is JSON; a tuple would not compare equal on read-back.
    assert result["acceptance_rate_range"] == [0.4, 0.6]
    assert "per_step_accepted" not in result
    assert "per_step_drafted" not in result
    assert "error" not in result


def test_to_result_leaves_unmeasured_metrics_empty() -> None:
    result = SpecDecodeSample(tg_tokens=10, total_ms=1000.0).to_result()

    assert result["effective_tg_tps"] == 10.0
    for key in (
        "acceptance_rate",
        "acceptance_rate_range",
        "speedup_ratio",
        "draft_window",
        "draft_tps",
        "waste_ratio",
        "verify_steps_per_s",
        "per_position_acceptance",
    ):
        assert result[key] is None, key
