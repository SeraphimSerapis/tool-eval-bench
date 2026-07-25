"""Server-independent legacy CLI command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from rich.console import Console

from tool_eval_bench.domain.scenarios import CATEGORY_LABELS


def _render_dry_run(
    args: argparse.Namespace,
    console: Console,
    resolve_scenarios: Callable[[argparse.Namespace], list[Any]],
) -> None:
    try:
        scenarios = resolve_scenarios(args)
    except ValueError as exc:
        console.print(f"\n[bold red]Error:[/] {exc}\n")
        raise SystemExit(2) from None
    if args.json:
        category_counts: dict[str, int] = {}
        for scenario in scenarios:
            category = scenario.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        output = {
            "event": "dry_run",
            "total_scenarios": len(scenarios),
            "estimated_minutes": round(len(scenarios) * 0.3, 1),
            "categories": {
                category: {
                    "label": CATEGORY_LABELS.get(
                        next(
                            scenario.category
                            for scenario in scenarios
                            if scenario.category.value == category
                        ),
                        category,
                    ),
                    "count": count,
                }
                for category, count in sorted(category_counts.items())
            },
            "scenarios": [
                {
                    "id": scenario.id,
                    "title": scenario.title,
                    "category": scenario.category.value,
                    "difficulty": scenario.difficulty,
                }
                for scenario in scenarios
            ],
        }
        sys.stdout.write(json.dumps(output, indent=2) + "\n")
    else:
        console.print(f"\n[bold]Dry run:[/] {len(scenarios)} scenarios would execute\n")
        console.print(f"  Estimated time: ~{len(scenarios) * 0.3:.0f} minutes (at ~18s/scenario)\n")
        category_counts = {}
        for scenario in scenarios:
            label = CATEGORY_LABELS.get(scenario.category, scenario.category.value)
            category_counts[label] = category_counts.get(label, 0) + 1
        for label, count in sorted(category_counts.items()):
            console.print(f"  {label}: {count} scenarios")
        console.print()
        difficulty_stars = {1: "★", 2: "★★", 3: "★★★", 4: "★★★★", 5: "★★★★★"}
        for scenario in scenarios:
            difficulty = (
                difficulty_stars.get(scenario.difficulty, "?") if scenario.difficulty else "?"
            )
            console.print(f"  [dim]{scenario.id}[/]  {difficulty:>5s}  {scenario.title}")
        console.print()
    raise SystemExit(0)


def handle_local_command(
    args: argparse.Namespace,
    console: Console,
    *,
    resolve_scenarios: Callable[[argparse.Namespace], list[Any]],
    print_history: Callable[..., Any],
    print_leaderboard: Callable[..., Any],
    export_runs: Callable[..., Any],
    compare_runs: Callable[..., Any],
) -> bool:
    """Handle a command that does not need inference-server discovery."""
    if args.history:
        print_history(console)
    elif args.leaderboard:
        print_leaderboard(console)
    elif args.export:
        export_runs(console, fmt=args.export, output=args.export_output)
    elif args.compare:
        compare_runs(console, args.compare[0], args.compare[1])
    elif args.dry_run:
        _render_dry_run(args, console, resolve_scenarios)
    else:
        return False
    return True
