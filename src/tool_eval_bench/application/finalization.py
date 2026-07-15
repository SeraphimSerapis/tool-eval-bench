"""Shared completed-run finalization invariant.

All benchmark modes follow the same ordering: create the human-readable
artifact first, then persist the completed run. A failed artifact write must
never leave a run claiming completion without its report.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

ReportWriter = Callable[[], Path | str | None]
RunPersister = Callable[[dict[str, Any]], None]


def finalize_completed_run(
    run_data: dict[str, Any],
    *,
    write_report: ReportWriter | None,
    persist: RunPersister | None,
) -> dict[str, Any]:
    """Write a report and persist the run in that order.

    The input mapping is updated with ``report_path`` when a report writer
    returns a path, then passed to the persistence callback.
    """
    if write_report is not None:
        report_path = write_report()
        if report_path is not None:
            run_data["report_path"] = str(report_path)
    if persist is not None:
        persist(run_data)
    return run_data
