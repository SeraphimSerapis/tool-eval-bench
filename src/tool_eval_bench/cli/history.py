"""CLI sub-commands: run history listing, diff, and comparison.

Extracted from bench.py (CODE-02) to keep each module focused.
"""

from __future__ import annotations

import sys

from rich.console import Console


def print_history(console: Console) -> None:
    """List recent benchmark runs from SQLite."""
    from rich.table import Table

    from tool_eval_bench.storage.db import RunRepository

    repo = RunRepository()
    runs = repo.list(limit=15)

    if not runs:
        console.print("\n  [dim]No previous runs found.[/]\n")
        return

    table = Table(
        title="[bold]Recent Benchmark Runs[/]",
        show_header=True,
        header_style="bold",
        border_style="bright_blue",
    )
    table.add_column("Run ID", min_width=30, no_wrap=True)
    table.add_column("Model", min_width=20)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Rating", min_width=16)
    table.add_column("Date", width=20)

    for run in runs:
        scores = run.get("scores") or {}
        score = scores.get("final_score", "?")
        rating = scores.get("rating", "")
        created = run.get("created_at", "?")[:19]
        table.add_row(
            f"[dim]{run['run_id']}[/]",
            run.get("model", "?"),
            f"[bold]{score}[/]",
            rating,
            f"[dim]{created}[/]",
        )

    console.print()
    console.print(table)
    console.print()


def print_diff(
    console: Console,
    current_results: list,  # list of ScenarioResult
    diff_run_id: str,
) -> None:
    """Compare current results against a previous run and print a diff table."""
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.storage.db import RunRepository

    repo = RunRepository()

    # Resolve 'latest' to actual run ID
    if diff_run_id.lower() == "latest":
        latest = repo.get_latest()
        if not latest:
            console.print("\n  [yellow]No previous runs found for comparison.[/]\n")
            return
        diff_run_id = latest["run_id"]

    prev_results = repo.get_scenario_results(diff_run_id)
    if prev_results is None:
        console.print(f"\n  [yellow]Run '{diff_run_id}' not found in database.[/]\n")
        return

    # Build lookup: scenario_id → previous result dict
    prev_map = {r["scenario_id"]: r for r in prev_results}

    # Stats
    improved = 0
    regressed = 0
    unchanged = 0
    new_scenarios = 0

    table = Table(
        title=f"[bold]Diff vs {diff_run_id[:30]}…[/]",
        show_header=True,
        header_style="bold",
        border_style="bright_cyan",
    )
    table.add_column("ID", width=6, no_wrap=True)
    table.add_column("Scenario", min_width=20, no_wrap=True)
    table.add_column("Prev", justify="center", width=6)
    table.add_column("→", justify="center", width=3)
    table.add_column("Now", justify="center", width=6)
    table.add_column("Δ", justify="center", width=6)
    table.add_column("Time Δ", justify="right", width=8)
    table.add_column("Note", ratio=1)

    status_symbols = {"pass": "✅", "partial": "⚠️", "fail": "❌"}

    for cr in current_results:
        sc_id = cr.scenario_id
        prev = prev_map.get(sc_id)

        cur_pts = cr.points
        cur_status = cr.status.value
        cur_dur = cr.duration_seconds

        if prev is None:
            new_scenarios += 1
            table.add_row(
                sc_id, cr.summary[:30],
                "[dim]—[/]", "→", f"[bold]{cur_pts}[/]/2",
                "[dim]new[/]", "", "[dim]new scenario[/]",
            )
            continue

        prev_pts = prev.get("points", 0)
        prev_status = prev.get("status", "fail")
        prev_dur = prev.get("duration_seconds", 0.0)

        delta = cur_pts - prev_pts
        dur_delta = cur_dur - prev_dur

        if delta > 0:
            improved += 1
            delta_str = f"[bold green]+{delta}[/]"
            note = "[green]✅ improved[/]"
        elif delta < 0:
            regressed += 1
            delta_str = f"[bold red]{delta}[/]"
            note = "[red]❌ regressed[/]"
        else:
            unchanged += 1
            delta_str = "[dim]=[/]"
            note = ""

        dur_sign = "+" if dur_delta >= 0 else ""
        dur_str = f"[dim]{dur_sign}{dur_delta:.1f}s[/]" if prev_dur > 0 else ""

        prev_sym = status_symbols.get(prev_status, "?")
        cur_sym = status_symbols.get(cur_status, "?")

        table.add_row(
            sc_id,
            cr.summary[:30] if delta != 0 else f"[dim]{cr.summary[:30]}[/]",
            f"[dim]{prev_sym} {prev_pts}[/]",
            "→",
            f"{cur_sym} [bold]{cur_pts}[/]",
            delta_str,
            dur_str,
            note,
        )

    console.print()
    console.print(table)

    # Summary line
    cur_total = sum(r.points for r in current_results)
    prev_total = sum(r.get("points", 0) for r in prev_results if r["scenario_id"] in {cr.scenario_id for cr in current_results})
    total_delta = cur_total - prev_total
    delta_color = "green" if total_delta > 0 else ("red" if total_delta < 0 else "dim")
    delta_sign = "+" if total_delta > 0 else ""

    summary = (
        f"  [green]↑ {improved} improved[/]  "
        f"[red]↓ {regressed} regressed[/]  "
        f"[dim]= {unchanged} unchanged[/]"
    )
    if new_scenarios:
        summary += f"  [cyan]+ {new_scenarios} new[/]"
    summary += f"\n  [bold]Points: {prev_total} → {cur_total} ([{delta_color}]{delta_sign}{total_delta}[/])[/]"

    console.print(Panel(summary, border_style="bright_cyan", padding=(0, 2)))
    console.print()


def compare_runs(console: Console, run_id_a: str, run_id_b: str) -> None:
    """Compare two stored runs from SQLite and print a diff table.

    run_id_a is treated as the baseline, run_id_b as the new run.
    Supports 'latest' as a shorthand for the most recent run.
    """
    from rich.panel import Panel
    from rich.table import Table

    from tool_eval_bench.storage.db import RunRepository

    repo = RunRepository()

    def _resolve(rid: str) -> tuple[str, dict]:
        if rid.lower() == "latest":
            run = repo.get_latest()
            if not run:
                console.print("\n  [red]No runs found in database.[/]\n")
                sys.exit(1)
            return run["run_id"], run
        run = repo.get(rid)
        if not run:
            console.print(f"\n  [red]Run '{rid}' not found in database.[/]\n")
            console.print("  [dim]Use --history to list available runs.[/]\n")
            sys.exit(1)
        return rid, run

    id_a, run_a = _resolve(run_id_a)
    id_b, run_b = _resolve(run_id_b)

    results_a = run_a.get("scores", {}).get("scenario_results", [])
    results_b = run_b.get("scores", {}).get("scenario_results", [])

    if not results_a or not results_b:
        console.print("\n  [red]One or both runs have no scenario results.[/]\n")
        return

    model_a = run_a.get("config", {}).get("model", "?")
    model_b = run_b.get("config", {}).get("model", "?")

    # Header
    console.print()
    console.print(Panel(
        f"  [bold]A (baseline):[/] {id_a[:40]}  [dim]model={model_a}[/]\n"
        f"  [bold]B (current):[/]  {id_b[:40]}  [dim]model={model_b}[/]",
        title="[bold]📊 Run Comparison[/]",
        border_style="bright_cyan",
    ))

    # Build lookup: scenario_id → result dict
    map_a = {r["scenario_id"]: r for r in results_a}
    map_b = {r["scenario_id"]: r for r in results_b}
    all_ids = list(dict.fromkeys(
        [r["scenario_id"] for r in results_a] + [r["scenario_id"] for r in results_b]
    ))

    status_symbols = {"pass": "✅", "partial": "⚠️", "fail": "❌"}
    improved = regressed = unchanged = 0

    table = Table(
        show_header=True,
        header_style="bold",
        border_style="bright_cyan",
    )
    table.add_column("ID", width=6, no_wrap=True)
    table.add_column("A", justify="center", width=8)
    table.add_column("→", justify="center", width=3)
    table.add_column("B", justify="center", width=8)
    table.add_column("Δ", justify="center", width=6)
    table.add_column("Time Δ", justify="right", width=8)
    table.add_column("Note", ratio=1)

    for sc_id in all_ids:
        ra = map_a.get(sc_id)
        rb = map_b.get(sc_id)

        if ra and not rb:
            table.add_row(sc_id, f"[dim]{ra.get('points', 0)}/2[/]", "→", "[dim]—[/]", "", "", "[dim]removed in B[/]")
            continue
        if rb and not ra:
            table.add_row(sc_id, "[dim]—[/]", "→", f"[bold]{rb.get('points', 0)}/2[/]", "[dim]new[/]", "", "[dim]new in B[/]")
            continue

        pts_a, pts_b = ra.get("points", 0), rb.get("points", 0)
        st_a, st_b = ra.get("status", "fail"), rb.get("status", "fail")
        dur_a, dur_b = ra.get("duration_seconds", 0.0), rb.get("duration_seconds", 0.0)

        delta = pts_b - pts_a
        if delta > 0:
            improved += 1
            delta_str = f"[bold green]+{delta}[/]"
            note = "[green]improved[/]"
        elif delta < 0:
            regressed += 1
            delta_str = f"[bold red]{delta}[/]"
            note = "[red]regressed[/]"
        else:
            unchanged += 1
            delta_str = "[dim]=[/]"
            note = ""

        dur_delta = dur_b - dur_a
        dur_sign = "+" if dur_delta >= 0 else ""
        dur_str = f"[dim]{dur_sign}{dur_delta:.1f}s[/]" if dur_a > 0 else ""

        sym_a, sym_b = status_symbols.get(st_a, "?"), status_symbols.get(st_b, "?")

        table.add_row(
            sc_id,
            f"[dim]{sym_a} {pts_a}[/]", "→", f"{sym_b} [bold]{pts_b}[/]",
            delta_str, dur_str, note,
        )

    console.print()
    console.print(table)

    # Summary
    total_a = sum(r.get("points", 0) for r in results_a)
    total_b = sum(r.get("points", 0) for r in results_b)
    total_delta = total_b - total_a
    delta_color = "green" if total_delta > 0 else ("red" if total_delta < 0 else "dim")
    delta_sign = "+" if total_delta > 0 else ""

    score_a = run_a.get("scores", {}).get("final_score", "?")
    score_b = run_b.get("scores", {}).get("final_score", "?")

    summary = (
        f"  [green]↑ {improved} improved[/]  "
        f"[red]↓ {regressed} regressed[/]  "
        f"[dim]= {unchanged} unchanged[/]\n"
        f"  [bold]Points: {total_a} → {total_b} ([{delta_color}]{delta_sign}{total_delta}[/])[/]\n"
        f"  [bold]Score:  {score_a} → {score_b}[/]"
    )

    console.print(Panel(summary, border_style="bright_cyan", padding=(0, 2)))
    console.print()
    repo.close()
