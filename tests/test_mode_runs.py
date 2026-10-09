"""Ordering and persistence contract of the shared mode-run finalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench.application import mode_runs, run_queries
from tool_eval_bench.application.mode_runs import ModeRun, finalize_mode_run
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports.mode import ModeReport, ReportContext


def _run(*, metadata_label: str | None = None) -> ModeRun:
    return ModeRun(
        run_type="spec-bench",
        config={"model": "m", "base_url": "http://user:secret@host:8000/v1", "mode": "spec-bench"},
        scores={"samples": 1},
        status="completed",
        metadata_label=metadata_label,
    )


def _report() -> ModeReport:
    return ModeReport(
        title="Speculative Decoding Benchmark",
        display_name="Display",
        mode="spec-bench",
        label=None,
        version_line=True,
        context=ReportContext.ENGINE,
        header=[],
        body=["## Results", ""],
    )


def _context() -> RunContext:
    return RunContext(
        tool_version="1.2.3",
        git_sha=None,
        hostname="host",
        platform_info="linux",
        python_version="3.13",
        model="m",
        backend="vllm",
        base_url="http://***:8000",
    )


@pytest.fixture
def persisted(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(run_queries, "persist_run", rows.append)
    return rows


def test_report_failure_prevents_completed_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    def fail_report(*_args: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(mode_runs, "write_mode_report", fail_report)
    with pytest.raises(OSError, match="disk full"):
        finalize_mode_run(
            _run(),
            _report(),
            run_context=None,
            output_dir=str(tmp_path),
        )

    assert persisted == []


def test_persisted_run_points_at_the_written_report(
    tmp_path: Path, persisted: list[dict[str, Any]]
) -> None:
    context = _context()

    finalized = finalize_mode_run(
        _run(metadata_label="nightly"),
        _report(),
        run_context=context,
        output_dir=str(tmp_path),
    )

    [run] = persisted
    assert list(run) == [
        "run_id",
        "run_type",
        "status",
        "config",
        "scores",
        "metadata",
        "report_path",
    ]
    assert run["run_id"] == finalized.run_id
    assert Path(run["report_path"]) == finalized.report_path
    assert finalized.report_path.is_file()
    assert finalized.report_path.is_relative_to(tmp_path)
    assert "secret" not in str(run["config"])
    assert run["config"]["config_fingerprint"]
    assert run["metadata"] == {**context.to_dict(), "label": "nightly"}


def test_persister_is_resolved_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A patch applied after import still intercepts persistence."""
    first: list[dict[str, Any]] = []
    second: list[dict[str, Any]] = []

    monkeypatch.setattr(run_queries, "persist_run", first.append)
    finalize_mode_run(_run(), _report(), run_context=None, output_dir=str(tmp_path))
    monkeypatch.setattr(run_queries, "persist_run", second.append)
    finalized = finalize_mode_run(_run(), _report(), run_context=None, output_dir=str(tmp_path))

    assert len(first) == 1
    assert [run["run_id"] for run in second] == [finalized.run_id]
    assert second[0]["metadata"] == {}
