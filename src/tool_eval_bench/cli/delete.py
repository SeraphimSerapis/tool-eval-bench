"""The ``delete`` command: remove stored runs and their Markdown reports."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm

from tool_eval_bench.application import run_queries
from tool_eval_bench.application.run_deletion import delete_runs, find_runs


def _error(console: Console, message: str, code: int) -> SystemExit:
    console.print(f"[bold red]Error:[/] {escape(message)}")
    return SystemExit(code)


def _confirm(console: Console, prompt: str) -> bool:
    return Confirm.ask(prompt, console=console, default=False)


def _selected_runs(args: argparse.Namespace, console: Console) -> Sequence[Mapping[str, Any]]:
    """The runs the arguments name, or the runs the filters match, newest first."""
    run_ids: list[str] = list(dict.fromkeys(args.delete or ()))
    statuses = args.delete_status
    filters = statuses or args.delete_before or args.model
    if run_ids and filters:
        raise _error(
            console, "Name run IDs or select runs with --status, --before, and --model, not both", 2
        )
    if not run_ids and not filters:
        raise _error(
            console,
            "Name the runs to delete, or select them with --status, --before, or --model",
            2,
        )
    if not run_ids:
        return find_runs(statuses=statuses, before=args.delete_before, model=args.model)

    if not run_queries.database_exists():
        raise _error(
            console, "No benchmark runs are stored here: there is no data/benchmarks.sqlite", 1
        )
    runs: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for run_id in run_ids:
        run = run_queries.get_run(run_id, include_traces=False)
        if run is None:
            missing.append(run_id)
        else:
            runs.append(run)
    if missing:
        raise _error(console, f"No stored run with id {', '.join(missing)}; nothing was deleted", 1)
    return runs


def _label(run: Mapping[str, Any]) -> str:
    label = run.get("label")
    if label is None:
        metadata = run.get("metadata")
        label = metadata.get("label") if isinstance(metadata, Mapping) else None
    return str(label) if label else ""


def _print_runs(console: Console, runs: Sequence[Mapping[str, Any]]) -> None:
    """One line per run, id first, so the id can be copied whole at any width."""
    for run in runs:
        stored = str(run.get("created_at", ""))[:16].replace("T", " ")
        kind = str(run.get("run_type") or "tool_eval")
        details = f"{run.get('status', '')} {kind}  {run.get('model', '')}"
        label = _label(run)
        if label:
            details += f"  [{label}]"
        console.print(
            f"  [bold]{escape(str(run.get('run_id', '')))}[/]  [dim]{escape(stored)} UTC[/]  "
            f"{escape(details)}",
            soft_wrap=True,
        )


def run_delete_command(args: argparse.Namespace, console: Console) -> None:
    """Delete the selected runs after showing them and asking for confirmation."""
    runs = _selected_runs(args, console)
    if not runs:
        console.print("No stored runs match; nothing was deleted.")
        return

    count = len(runs)
    noun = "run" if count == 1 else "runs"
    _print_runs(console, runs)
    if any(run.get("status") == "running" for run in runs):
        console.print(
            "[yellow]Note:[/] a run still executing in another process writes its row "
            "back when it finishes."
        )
    if args.dry_run:
        console.print(f"Dry run: {count} {noun} would be deleted.")
        return
    if not args.yes:
        if not sys.stdin.isatty():
            raise _error(console, "Pass --yes to delete without a confirmation prompt", 2)
        if not _confirm(console, f"Delete {count} {noun} and their Markdown reports?"):
            console.print("Nothing was deleted.")
            return

    result = delete_runs([str(run["run_id"]) for run in runs])
    deleted = len(result.deleted)
    reports = len(result.reports_removed)
    console.print(
        f"[bold green]✓[/] Deleted {deleted} {'run' if deleted == 1 else 'runs'} and "
        f"{reports} {'report' if reports == 1 else 'reports'}."
    )
    for kept in result.reports_kept:
        console.print(f"[yellow]Kept[/] {escape(kept.path)}: {escape(kept.reason)}")
    if result.missing:
        # Another process deleted them between the listing and the delete.
        console.print(f"[dim]Already gone: {escape(', '.join(result.missing))}[/]")
