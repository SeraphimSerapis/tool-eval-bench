"""Leaderboard and export CLI sub-commands.

Provides:
  tool-eval-bench leaderboard   — beautiful Rich table ranking all models
  tool-eval-bench export        — CSV/JSON export of all stored runs
"""

from __future__ import annotations

import csv
import io
import json
import sys
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from tool_eval_bench.application.leaderboard import extract_leaderboard_rows, shorten_model_name

# ---------------------------------------------------------------------------
# Score-to-color helpers
# ---------------------------------------------------------------------------


def _score_color(pct: float) -> str:
    """Map a percentage score to a Rich color for heatmap display."""
    if pct >= 90:
        return "bold green"
    if pct >= 75:
        return "green"
    if pct >= 60:
        return "yellow"
    if pct >= 40:
        return "red"
    return "bold red"


def _score_bg(pct: float) -> str:
    """Map percentage to background-style heatmap cell content."""
    if pct >= 90:
        return f"[bold green]{pct:>3.0f}[/]"
    if pct >= 75:
        return f"[green]{pct:>3.0f}[/]"
    if pct >= 60:
        return f"[yellow]{pct:>3.0f}[/]"
    if pct >= 40:
        return f"[red]{pct:>3.0f}[/]"
    return f"[bold red]{pct:>3.0f}[/]"


def _rating_short(rating: str) -> str:
    """Shorten a rating string for compact display."""
    if "Excellent" in rating:
        return "[bold green]★★★★★[/]"
    if "Good" in rating:
        return "[green]★★★★[/]"
    if "safety-capped" in rating:
        return "[bold red]★★★ⓢ[/]"
    if "Adequate" in rating:
        return "[yellow]★★★[/]"
    if "Weak" in rating:
        return "[red]★★[/]"
    return "[bold red]★[/]"


# ---------------------------------------------------------------------------
# Category label mapping
# ---------------------------------------------------------------------------

_CAT_LABELS = {
    "A": "Sel",  # Tool Selection
    "B": "Prm",  # Parameter Precision
    "C": "Chn",  # Multi-Step Chains
    "D": "Rst",  # Restraint & Refusal
    "E": "Err",  # Error Recovery
    "F": "Loc",  # Localization
    "G": "Rsn",  # Structured Reasoning
    "H": "Ins",  # Instruction Following
    "I": "Ctx",  # Context & State
    "J": "Cod",  # Code Patterns
    "K": "Saf",  # Safety & Boundaries
    "L": "Scl",  # Toolset Scale
    "M": "Pln",  # Autonomous Planning
    "N": "Crt",  # Creative Composition
    "O": "Out",  # Structured Output
}

_CAT_FULL = {
    "A": "Tool Selection",
    "B": "Param Precision",
    "C": "Multi-Step Chains",
    "D": "Restraint",
    "E": "Error Recovery",
    "F": "Localization",
    "G": "Reasoning",
    "H": "Instruction",
    "I": "Context & State",
    "J": "Code Patterns",
    "K": "Safety",
    "L": "Toolset Scale",
    "M": "Planning",
    "N": "Creative",
    "O": "Structured Output",
}


# ---------------------------------------------------------------------------
# Leaderboard display
# ---------------------------------------------------------------------------


def print_leaderboard(console: Console, limit: int = 50) -> None:
    """Print a beautiful, screenshottable leaderboard table."""
    from tool_eval_bench.application.run_queries import recent_runs

    runs = recent_runs(limit=500)  # Fetch many to deduplicate

    if not runs:
        console.print("\n  [dim]No benchmark runs found. Run some benchmarks first![/]\n")
        return

    rows = extract_leaderboard_rows(runs)
    if not rows:
        console.print("\n  [dim]No valid results found.[/]\n")
        return

    rows = rows[:limit]

    # Determine which categories are present across all runs
    all_cats = set()
    for r in rows:
        all_cats.update(r["cat_scores"].keys())
    cat_order = sorted(all_cats)

    # Build the leaderboard table
    table = Table(
        title="[bold]🏆 Model Leaderboard[/]",
        show_header=True,
        header_style="bold",
        border_style="bright_blue",
        title_style="bold bright_white",
        show_lines=True,
        padding=(0, 1),
        expand=True,
    )

    # Rank column
    table.add_column("#", justify="center", width=3, style="bold")
    table.add_column("Model", min_width=20, no_wrap=True)
    table.add_column("Cohort", justify="center", width=32, no_wrap=True)
    table.add_column("Score", justify="center", width=7, style="bold")
    table.add_column("Rating", justify="center", width=7)
    table.add_column("P/F/C", justify="center", width=14, no_wrap=True)

    # Per-category heatmap columns
    for cat in cat_order:
        label = _CAT_LABELS.get(cat, cat)
        table.add_column(
            label,
            justify="center",
            width=4,
            no_wrap=True,
        )

    # Efficiency and metadata
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Runs", justify="center", width=4)

    single_cohort = len({row["cohort_fingerprint"] for row in rows}) == 1
    for idx, row in enumerate(rows, 1):
        if idx > 1 and row["cohort_rank"] == 1:
            table.add_section()

        # Rank medal
        if not single_cohort:
            rank = f"[dim]{row['cohort_rank']}[/]"
        elif idx == 1:
            rank = "[bold bright_yellow]🥇[/]"
        elif idx == 2:
            rank = "[bold white]🥈[/]"
        elif idx == 3:
            rank = "[bold #cd7f32]🥉[/]"
        else:
            rank = f"[dim]{idx}[/]"

        # Model name — shorten paths to owner/model for readability (#15)
        model_name = shorten_model_name(row["model"])

        # Score with color
        score = row["final_score"]
        score_str = f"[{_score_color(score)}]{score}[/]"

        # Rating stars
        rating_str = _rating_short(row["rating"])

        # Cohort label explains the comparison conditions for this section.
        cohort_str = f"[dim]{escape(row['cohort_label'])}[/]"

        # Pass/Fail summary
        p, pt, f = row["passes"], row["partials"], row["fails"]
        complete = row["completion_count"]
        pf_str = (
            f"[green]{p}[/]/[yellow]{pt}[/]/[red]{f}[/] [dim]{complete}/{row['scenario_count']}[/]"
        )

        # Category heatmap cells
        cat_cells = []
        for cat in cat_order:
            pct = row["cat_scores"].get(cat, -1)
            if pct < 0:
                cat_cells.append("[dim]—[/]")
            else:
                cat_cells.append(_score_bg(pct))

        # Token usage
        tokens = row.get("total_tokens", 0)
        tok_str = f"[dim]{tokens // 1000}K[/]" if tokens > 0 else "[dim]—[/]"

        # Runs count
        runs_str = f"[dim]{row['num_runs']}[/]"

        table.add_row(
            rank,
            f"[bold]{escape(model_name)}[/]",
            cohort_str,
            score_str,
            rating_str,
            pf_str,
            *cat_cells,
            tok_str,
            runs_str,
        )

    console.print()
    console.print(table)

    # Legend
    legend_parts = []
    for cat in cat_order:
        full = _CAT_FULL.get(cat, cat)
        short = _CAT_LABELS.get(cat, cat)
        legend_parts.append(f"[bold]{short}[/]={full}")

    # Check if any partial runs exist
    has_partial = any(r.get("scenario_count", 69) < 69 for r in rows)
    partial_note = (
        "\n  [dim]Runs with different backends or scenario counts are ranked separately.[/]"
        if has_partial
        else ""
    )

    console.print(
        Panel(
            "  ".join(legend_parts)
            + "\n\n"
            + "  [dim]P/F/C = ✅pass / ⚠️partial / ❌fail / completed scenarios   │   "
            + "Cohort = comparable benchmark conditions   │   "
            + "Scores: [bold green]90+[/] [green]75+[/] [yellow]60+[/] [red]40+[/] [bold red]<40[/]   │   "
            + "★★★ⓢ = safety-capped[/]"
            + partial_note
            + (
                "\n  [dim]Different benchmark cohorts are grouped and ranked separately.[/]"
                if not single_cohort
                else ""
            ),
            border_style="dim",
            padding=(0, 1),
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_runs(
    console: Console,
    fmt: str = "csv",
    output: str | None = None,
    limit: int = 500,
) -> None:
    """Export all stored runs in CSV or JSON format."""
    from tool_eval_bench.application.run_queries import recent_runs

    runs = recent_runs(limit=limit)

    if not runs:
        console.print("\n  [dim]No benchmark runs found.[/]\n")
        return

    rows = extract_leaderboard_rows(runs)

    # All categories present
    all_cats = set()
    for r in rows:
        all_cats.update(r["cat_scores"].keys())
    cat_order = sorted(all_cats)

    if fmt == "json":
        export_data = []
        for row in rows:
            entry: dict[str, Any] = {
                "model": row["model"],
                "run_id": row["run_id"],
                "date": row["date"],
                "final_score": row["final_score"],
                "rating": row["rating"],
                "total_points": row["total_points"],
                "max_points": row["max_points"],
                "passes": row["passes"],
                "partials": row["partials"],
                "fails": row["fails"],
                "scenario_count": row["scenario_count"],
                "backend": row["backend"],
                "total_tokens": row["total_tokens"],
                "num_runs": row["num_runs"],
                "cohort_label": row["cohort_label"],
                "cohort_fingerprint": row["cohort_fingerprint"],
                "categories": {cat: row["cat_scores"].get(cat) for cat in cat_order},
            }
            if row.get("token_efficiency") is not None:
                entry["token_efficiency"] = row["token_efficiency"]
            if row.get("deployability") is not None:
                entry["deployability"] = row["deployability"]
            if row.get("median_turn_ms") is not None:
                entry["median_turn_ms"] = row["median_turn_ms"]
            # Issue #6 metadata fields
            for meta_key in [
                "tool_version",
                "engine_name",
                "engine_version",
                "quantization",
                "max_model_len",
                "temperature",
                "server_model_root",
            ]:
                if row.get(meta_key) is not None:
                    entry[meta_key] = row[meta_key]
            export_data.append(entry)

        result = json.dumps(export_data, indent=2, default=str)
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result)
            console.print(f"\n  [green]✓[/] Exported {len(rows)} models to [bold]{output}[/]\n")
        else:
            print(result)

    elif fmt == "csv":
        headers = [
            "rank",
            "model",
            "run_id",
            "date",
            "final_score",
            "rating",
            "total_points",
            "max_points",
            "passes",
            "partials",
            "fails",
            "scenario_count",
            "backend",
            "total_tokens",
            "num_runs",
            "tool_version",
            "engine_name",
            "engine_version",
            "quantization",
            "max_model_len",
            "temperature",
            "server_model_root",
            "cohort",
            "cohort_fingerprint",
        ]
        headers.extend(f"cat_{cat}" for cat in cat_order)

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            # Ranks restart per cohort, as on screen: a global rank across
            # non-comparable cohorts would put whichever label sorts first on top.
            csv_row: dict[str, Any] = {
                "rank": row["cohort_rank"],
                "model": row["model"],
                "run_id": row["run_id"],
                "date": row["date"],
                "final_score": row["final_score"],
                "rating": row["rating"],
                "total_points": row["total_points"],
                "max_points": row["max_points"],
                "passes": row["passes"],
                "partials": row["partials"],
                "fails": row["fails"],
                "scenario_count": row["scenario_count"],
                "backend": row["backend"],
                "total_tokens": row["total_tokens"],
                "num_runs": row["num_runs"],
                "tool_version": row.get("tool_version", ""),
                "engine_name": row.get("engine_name", ""),
                "engine_version": row.get("engine_version", ""),
                "quantization": row.get("quantization", ""),
                "max_model_len": row.get("max_model_len", ""),
                "temperature": row.get("temperature", ""),
                "server_model_root": row.get("server_model_root", ""),
                "cohort": row["cohort_label"],
                "cohort_fingerprint": row["cohort_fingerprint"],
            }
            for cat in cat_order:
                csv_row[f"cat_{cat}"] = row["cat_scores"].get(cat, "")
            writer.writerow(csv_row)

        result = buf.getvalue()
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result)
            console.print(f"\n  [green]✓[/] Exported {len(rows)} models to [bold]{output}[/]\n")
        else:
            print(result)
    else:
        console.print(f"\n  [red]Unknown format: {fmt}. Use 'csv' or 'json'.[/]\n")
        sys.exit(1)
