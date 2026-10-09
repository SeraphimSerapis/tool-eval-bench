"""A scored ``--perf`` run stores its throughput beside the score.

The measurements go under one additive ``scores["throughput"]`` key. The second
test proves that key is invisible to every reader of the scored row: history,
leaderboard, export, and compare render the same output with or without it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests.conftest import open_repository
from tests.test_e2e import SmokeMockAdapter
from tool_eval_bench.runner.service import BenchmarkService
from tool_eval_bench.runner.throughput import ThroughputSample
from tool_eval_bench.storage.reports import MarkdownReporter

OK_CELL = ThroughputSample(
    pp_tokens=2048,
    tg_tokens=128,
    depth=0,
    concurrency=1,
    ttft_ms=210.5,
    total_ms=1560.25,
    pp_tps=9729.2,
    tg_tps=94.81,
    requested_pp=2048,
    calibration_confidence="llama-benchy",
)
FAILED_CELL = ThroughputSample(
    concurrency=4,
    requested_pp=2048,
    error="Server error '503' for url 'http://user:secret@gpu-box.internal:8000/v1'",
)


async def _scored_run(tmp_path: Path, throughput: list[ThroughputSample] | None) -> dict[str, Any]:
    from tool_eval_bench.evals.scenarios import SCENARIOS

    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")
    service = BenchmarkService(
        repo=open_repository(db_path=str(tmp_path / "bench.sqlite")),
        reporter=MarkdownReporter(root=str(tmp_path / "runs")),
    )
    service._adapter_for = lambda *_args, **_kwargs: SmokeMockAdapter()  # type: ignore[method-assign]
    result: dict[str, Any] = await service.run_benchmark(
        model="perf-model",
        backend="vllm",
        base_url="http://localhost:9999",
        scenarios=[tc01],
        timeout_seconds=10.0,
        throughput_samples=throughput,
    )
    return result


@pytest.mark.asyncio
async def test_scored_perf_run_stores_throughput_through_sqlite(tmp_path: Path) -> None:
    with_perf = await _scored_run(tmp_path, [OK_CELL, FAILED_CELL])
    without_perf = await _scored_run(tmp_path, None)

    repo = open_repository(db_path=str(tmp_path / "bench.sqlite"))
    stored = repo.get(with_perf["run_id"])
    assert stored["scores"]["throughput"] == {
        "samples": 2,
        "failed": 1,
        "results": [OK_CELL.to_result()],
    }
    # Raw, not the report's rounding.
    assert stored["scores"]["throughput"]["results"][0]["tg_tps"] == 94.81
    # The failed cell is a count only, so its error text never reaches the row.
    assert "secret" not in json.dumps(stored)
    # The score itself is untouched by the extra key.
    assert stored["scores"]["final_score"] == 100

    # Present only when perf ran.
    assert "throughput" not in repo.get(without_perf["run_id"])["scores"]
    # The run config, and with it the fingerprint, does not depend on perf.
    fingerprint = with_perf["config"]["config_fingerprint"]
    assert fingerprint == without_perf["config"]["config_fingerprint"]


def _serve(monkeypatch: pytest.MonkeyPatch, runs: list[dict[str, Any]]) -> None:
    from tool_eval_bench.application import run_queries

    by_id = {run["run_id"]: run for run in runs}

    def recent_runs(limit: int = 15) -> list[dict[str, Any]]:
        return copy.deepcopy(runs[:limit])

    def resolve_run(run_id: str) -> tuple[str, dict[str, Any]] | None:
        run = by_id.get(run_id)
        return (run_id, copy.deepcopy(run)) if run is not None else None

    monkeypatch.setattr(run_queries, "recent_runs", recent_runs)
    monkeypatch.setattr(run_queries, "resolve_run", resolve_run)


def _render_every_reader(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runs: list[dict[str, Any]],
) -> dict[str, str]:
    from tool_eval_bench.cli.history import compare_runs, print_history
    from tool_eval_bench.cli.leaderboard import export_runs, print_leaderboard

    _serve(monkeypatch, runs)
    rendered: dict[str, str] = {}
    for name, render in {
        "history": print_history,
        "leaderboard": print_leaderboard,
        "compare": lambda c: compare_runs(c, runs[1]["run_id"], runs[0]["run_id"]),
        "export-json": lambda c: export_runs(c, fmt="json"),
        "export-csv": lambda c: export_runs(c, fmt="csv"),
    }.items():
        console = Console(record=True, width=200)
        capsys.readouterr()
        render(console)
        rendered[name] = console.export_text() + capsys.readouterr().out
    return rendered


@pytest.mark.asyncio
async def test_readers_of_the_scored_row_ignore_the_throughput_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _scored_run(tmp_path, [OK_CELL])
    await _scored_run(tmp_path, [OK_CELL, FAILED_CELL])
    repo = open_repository(db_path=str(tmp_path / "bench.sqlite"))
    stored = repo.list()
    assert all("throughput" in run["scores"] for run in stored)
    stripped = copy.deepcopy(stored)
    for run in stripped:
        del run["scores"]["throughput"]

    with_key = _render_every_reader(monkeypatch, capsys, stored)
    without_key = _render_every_reader(monkeypatch, capsys, stripped)

    assert with_key == without_key
    # Each reader rendered the runs, so equal output is not two empty screens.
    assert stored[0]["run_id"] in with_key["history"]
    assert "perf-model" in with_key["leaderboard"]
    assert json.loads(with_key["export-json"])[0]["model"] == "perf-model"
    assert stored[0]["run_id"] in with_key["export-csv"]
    assert "TC-01" in with_key["compare"]
