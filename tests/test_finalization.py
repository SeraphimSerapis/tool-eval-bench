from __future__ import annotations

from pathlib import Path

import pytest

from tool_eval_bench.application.finalization import finalize_completed_run


def test_finalization_writes_artifact_before_persisting() -> None:
    events: list[str] = []
    run_data = {"run_id": "r1", "status": "completed"}
    # str(Path(...)) uses the platform separator, so name the expectation the
    # same way rather than hardcoding a forward slash.
    report = Path("runs") / "r1.md"

    result = finalize_completed_run(
        run_data,
        write_report=lambda: events.append("report") or report,
        persist=lambda data: events.append(f"persist:{data['report_path']}"),
    )

    assert result["report_path"] == str(report)
    assert events == ["report", f"persist:{report}"]


def test_finalization_does_not_persist_when_report_fails() -> None:
    persisted = False

    def fail_report() -> Path:
        raise OSError("disk full")

    def persist(_data: dict) -> None:
        nonlocal persisted
        persisted = True

    with pytest.raises(OSError, match="disk full"):
        finalize_completed_run({}, write_report=fail_report, persist=persist)
    assert persisted is False
