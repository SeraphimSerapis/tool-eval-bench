"""Regression coverage for validation performed before server discovery."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from tool_eval_bench.cli.bench import _make_parser
from tool_eval_bench.cli.local_commands import _render_dry_run
from tool_eval_bench.cli.parser import parse_cli_args


@pytest.mark.parametrize(
    "argv",
    [
        ["--dry-run", "--categories", "Z"],
        ["--dry-run", "--scenarios", "NOPE"],
        ["run", "--dry-run", "--scenarios", "TC-01", "NOPE"],
    ],
)
def test_dry_run_rejects_unknown_scenario_selectors(argv: list[str]) -> None:
    _, args = parse_cli_args(_make_parser, argv)
    console = Console(file=StringIO(), no_color=True)

    from tool_eval_bench.cli.commands import resolve_scenarios

    with pytest.raises(SystemExit) as exc_info:
        _render_dry_run(args, console, resolve_scenarios)

    assert exc_info.value.code == 2
    assert "Error:" in console.file.getvalue()


@pytest.mark.parametrize(
    "argv",
    [
        ["--parallel", "0"],
        ["--parallel", "-1"],
        ["plugin", "gsm8k", "--parallel", "0"],
    ],
)
def test_parallel_must_be_positive_at_parse_time(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_args(_make_parser, argv)

    assert exc_info.value.code == 2


def test_spec_bench_finalization_persists_report_path() -> None:
    from tool_eval_bench.cli.spec_bench import _report_then_persist_spec_bench

    persisted: list[dict] = []
    run_data = {"run_id": "spec-run", "status": "completed"}

    _report_then_persist_spec_bench(
        run_data=run_data,
        write_report=lambda: "runs/2026/08/spec-run.md",
        persist_plugin_run=persisted.append,
    )

    assert persisted == [{**run_data, "report_path": "runs/2026/08/spec-run.md"}]
