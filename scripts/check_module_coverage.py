#!/usr/bin/env python3
"""Fail when coverage of a critical user-facing module drops below its floor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODULE_FLOORS = {
    "src/tool_eval_bench/cli/compare_report.py": 80.0,
    "src/tool_eval_bench/cli/dispatch.py": 70.0,
    "src/tool_eval_bench/cli/parser.py": 75.0,
    "src/tool_eval_bench/cli/plugin_runners.py": 80.0,
    "src/tool_eval_bench/cli/pressure.py": 75.0,
    "src/tool_eval_bench/cli/server.py": 80.0,
    "src/tool_eval_bench/compare_reports/summary.py": 75.0,
    "src/tool_eval_bench/compare_reports/tool_eval.py": 80.0,
    "src/tool_eval_bench/runner/speculative.py": 90.0,
    "src/tool_eval_bench/runner/throughput.py": 80.0,
}


def check_module_coverage(report_path: Path) -> list[str]:
    """Return human-readable failures from a coverage.py JSON report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    files = report.get("files", {})
    failures: list[str] = []
    for module, floor in MODULE_FLOORS.items():
        details = files.get(module)
        if details is None:
            failures.append(f"{module}: absent from coverage report")
            continue
        actual = float(details["summary"]["percent_covered"])
        if actual < floor:
            failures.append(f"{module}: {actual:.2f}% < {floor:.2f}%")
        else:
            print(f"PASS {module}: {actual:.2f}% >= {floor:.2f}%")
    return failures


def main() -> int:
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "coverage.json")
    failures = check_module_coverage(report_path)
    if not failures:
        return 0
    print("Critical module coverage gate failed:", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
