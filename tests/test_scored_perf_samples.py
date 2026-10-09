"""A scored ``--perf`` run hands every throughput cell to the service, in every output mode.

The service stores ``scores["throughput"]`` with a ``failed`` count, so the CLI
must pass failed cells along rather than dropping them first. The reports and
the live display filter failed cells themselves, which the last test pins.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import open_repository
from tests.test_e2e import SmokeMockAdapter
from tests.test_scored_perf_scores import FAILED_CELL, OK_CELL
from tool_eval_bench.application import service as service_module
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.cli import dispatch
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
from tool_eval_bench.runner.orchestrator import score_results
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.storage.reports import scenario as scenario_report


async def _no_metadata(*_args: Any) -> dict[str, Any]:
    return {}


@pytest.mark.parametrize(
    "mode_flags", [[], ["--no-live"], ["--json"]], ids=["live", "plain", "json"]
)
def test_scored_perf_row_counts_failed_cells_in_every_mode(
    mode_flags: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key in list(os.environ):
        if key.startswith("TOOL_EVAL_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: [OK_CELL, FAILED_CELL])
    monkeypatch.setattr(dispatch, "_build_run_context", lambda *a, **k: None)
    monkeypatch.setattr(service_module, "_collect_metadata_safe", _no_metadata)
    monkeypatch.setattr(
        BenchmarkService, "_adapter_for", lambda *_args, **_kwargs: SmokeMockAdapter()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--base-url",
            "http://localhost:9999",
            "--model",
            "perf-model",
            "--backend",
            "vllm",
            "--scenarios",
            "TC-01",
            "--perf",
            "--no-preflight",
            "--no-warmup",
            "--no-probe-engine",
            "--output-dir",
            str(tmp_path / "runs"),
            *mode_flags,
        ],
    )

    dispatch.main()

    capsys.readouterr()
    (stored,) = open_repository(db_path=str(tmp_path / "data" / "benchmarks.sqlite")).list()
    assert stored["scores"]["throughput"] == {
        "samples": 2,
        "failed": 1,
        "results": [OK_CELL.to_result()],
    }


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> _FrozenDatetime:
        return cls(2026, 10, 9, 4, 0, tzinfo=timezone.utc)


def test_scored_report_is_identical_with_or_without_failed_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tool_eval_bench.evals.scenarios import SCENARIOS

    monkeypatch.setattr(scenario_report, "datetime", _FrozenDatetime)
    tc01 = next(s for s in SCENARIOS if s.id == "TC-01")
    summary = score_results([ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok")], [tc01])

    def report(samples: list[Any], root: Path) -> bytes:
        path = MarkdownReporter(root=str(root)).write_scenario_report(
            "run-1", "perf-model", summary, throughput_samples=samples
        )
        return path.read_bytes()

    with_failed = report([OK_CELL, FAILED_CELL], tmp_path / "a")
    successful_only = report([OK_CELL], tmp_path / "b")

    assert with_failed == successful_only
    assert b"## Throughput Metrics" in with_failed
