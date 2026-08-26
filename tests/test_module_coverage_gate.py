from __future__ import annotations

import json
from pathlib import Path

from scripts.check_module_coverage import MODULE_FLOORS, check_module_coverage


def _write_report(path: Path, percentages: dict[str, float]) -> None:
    path.write_text(
        json.dumps(
            {
                "files": {
                    module: {"summary": {"percent_covered": percent}}
                    for module, percent in percentages.items()
                }
            }
        ),
        encoding="utf-8",
    )


def test_critical_module_coverage_accepts_floors(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    _write_report(report, MODULE_FLOORS)
    assert check_module_coverage(report) == []


def test_critical_module_coverage_reports_missing_and_low_modules(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    modules = dict(MODULE_FLOORS)
    missing = next(iter(modules))
    modules.pop(missing)
    low = next(iter(modules))
    modules[low] -= 0.01
    _write_report(report, modules)

    failures = check_module_coverage(report)

    assert f"{missing}: absent from coverage report" in failures
    assert any(failure.startswith(f"{low}:") for failure in failures)


def test_the_gate_reads_a_report_written_on_windows(tmp_path: Path) -> None:
    """coverage.py records the separator of the platform it ran on.

    Matching keys literally made every floored module look absent on Windows,
    so the gate failed for a reason that had nothing to do with coverage.
    """
    report = tmp_path / "coverage.json"
    _write_report(
        report, {module.replace("/", "\\"): floor for module, floor in MODULE_FLOORS.items()}
    )

    assert check_module_coverage(report) == []
