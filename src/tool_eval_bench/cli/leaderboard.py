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
from rich.panel import Panel
from rich.table import Table



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
    "A": "Sel",   # Tool Selection
    "B": "Prm",   # Parameter Precision
    "C": "Chn",   # Multi-Step Chains
    "D": "Rst",   # Restraint & Refusal
    "E": "Err",   # Error Recovery
    "F": "Loc",   # Localization
    "G": "Rsn",   # Structured Reasoning
    "H": "Ins",   # Instruction Following
    "I": "Ctx",   # Context & State
    "J": "Cod",   # Code Patterns
    "K": "Saf",   # Safety & Boundaries
    "L": "Scl",   # Toolset Scale
    "M": "Pln",   # Autonomous Planning
    "N": "Crt",   # Creative Composition
    "O": "Out",   # Structured Output
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
# Data extraction from stored runs
# ---------------------------------------------------------------------------

def _extract_leaderboard_rows(
    runs: list[dict],
) -> list[dict[str, Any]]:
    """Extract normalized rows from stored run data.

    Groups by model and takes the best (highest-scoring) run for each.
    """
    # Group runs by model
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        model = run.get("model", "unknown")
        by_model.setdefault(model, []).append(run)

    rows: list[dict[str, Any]] = []
    for model, model_runs in by_model.items():
        # Take the best run (highest final score)
        best = max(
            model_runs,
            key=lambda r: (r.get("scores") or {}).get("final_score", 0),
        )
        scores = best.get("scores") or {}
        config = best.get("config") or {}

        # Extract per-category percentages
        cat_scores: dict[str, float] = {}
        for cs in scores.get("category_scores", []):
            cat_scores[cs["category"]] = cs.get("percent", 0)

        # Count scenarios by status
        results = scores.get("scenario_results", [])
        passes = sum(1 for r in results if r.get("status") == "pass")
        partials = sum(1 for r in results if r.get("status") == "partial")
        fails = sum(1 for r in results if r.get("status") == "fail")

        # Extract metadata (issue #6)
        metadata = best.get("metadata") or {}

        rows.append({
            "model": model,
            "run_id": best.get("run_id", "?"),
            "date": best.get("created_at", "?")[:19],
            "final_score": scores.get("final_score", 0),
            "rating": scores.get("rating", "?"),
            "total_points": scores.get("total_points", 0),
            "max_points": scores.get("max_points", 0),
            "passes": passes,
            "partials": partials,
            "fails": fails,
            "cat_scores": cat_scores,
            "scenario_count": config.get("scenario_count", len(results)),
            "backend": config.get("backend", "?"),
            "total_tokens": scores.get("total_tokens", 0),
            "token_efficiency": scores.get("token_efficiency"),
            "median_turn_ms": scores.get("median_turn_ms"),
            "deployability": scores.get("deployability"),
            "safety_warnings": len(scores.get("safety_warnings", [])),
            "num_runs": len(model_runs),
            # Issue #6 metadata fields
            "tool_version": metadata.get("tool_version"),
            "engine_name": metadata.get("engine_name"),
            "engine_version": metadata.get("engine_version"),
            "quantization": metadata.get("quantization"),
            "max_model_len": metadata.get("max_model_len"),
            "temperature": metadata.get("temperature"),
            "server_model_root": metadata.get("server_model_root"),
        })

    # Sort by final score descending
    rows.sort(key=lambda r: r["final_score"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Leaderboard display
# ---------------------------------------------------------------------------

def print_leaderboard(console: Console, limit: int = 50) -> None:
    """Print a beautiful, screenshottable leaderboard table."""
    from tool_eval_bench.storage.db import RunRepository

    repo = RunRepository()
    runs = repo.list(limit=500)  # Fetch many to deduplicate
    repo.close()

    if not runs:
        console.print("\n  [dim]No benchmark runs found. Run some benchmarks first![/]\n")
        return

    rows = _extract_leaderboard_rows(runs)
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
    )

    # Rank column
    table.add_column("#", justify="center", width=3, style="bold")
    table.add_column("Model", min_width=20, no_wrap=True, max_width=40)
    table.add_column("Score", justify="center", width=7, style="bold")
    table.add_column("Rating", justify="center", width=7)
    table.add_column("P/F", justify="center", width=9, no_wrap=True)

    # Per-category heatmap columns
    for cat in cat_order:
        label = _CAT_LABELS.get(cat, cat)
        table.add_column(
            label, justify="center", width=4, no_wrap=True,
        )

    # Efficiency and metadata
    table.add_column("N", justify="center", width=4, no_wrap=True)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Runs", justify="center", width=4)

    for idx, row in enumerate(rows, 1):
        # Rank medal
        if idx == 1:
            rank = "[bold bright_yellow]🥇[/]"
        elif idx == 2:
            rank = "[bold white]🥈[/]"
        elif idx == 3:
            rank = "[bold #cd7f32]🥉[/]"
        else:
            rank = f"[dim]{idx}[/]"

        # Model name (truncate if needed)
        model_name = row["model"]
        if len(model_name) > 38:
            model_name = model_name[:35] + "…"

        # Score with color
        score = row["final_score"]
        score_str = f"[{_score_color(score)}]{score}[/]"

        # Rating stars
        rating_str = _rating_short(row["rating"])

        # Pass/Fail summary
        p, pt, f = row["passes"], row["partials"], row["fails"]
        pf_str = f"[green]{p}[/]/[yellow]{pt}[/]/[red]{f}[/]"

        # Category heatmap cells
        cat_cells = []
        for cat in cat_order:
            pct = row["cat_scores"].get(cat, -1)
            if pct < 0:
                cat_cells.append("[dim]—[/]")
            else:
                cat_cells.append(_score_bg(pct))

        # Scenario count (flag partial runs)
        sc_count = row.get("scenario_count", 0)
        if sc_count < 69:
            sc_str = f"[bold yellow]{sc_count}⚠[/]"
        else:
            sc_str = f"[dim]{sc_count}[/]"

        # Token usage
        tokens = row.get("total_tokens", 0)
        tok_str = f"[dim]{tokens // 1000}K[/]" if tokens > 0 else "[dim]—[/]"

        # Runs count
        runs_str = f"[dim]{row['num_runs']}[/]"

        table.add_row(
            rank,
            f"[bold]{model_name}[/]",
            score_str,
            rating_str,
            pf_str,
            *cat_cells,
            sc_str,
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
    partial_note = "\n  [bold yellow]⚠ N<69:[/] [dim]Partial run (--short/--categories) — not comparable to full runs.[/]" if has_partial else ""

    console.print(
        Panel(
            "  ".join(legend_parts)
            + "\n\n"
            + "  [dim]P/F = ✅pass / ⚠️partial / ❌fail   │   "
            + "N = scenario count   │   "
            + "Scores: [bold green]90+[/] [green]75+[/] [yellow]60+[/] [red]40+[/] [bold red]<40[/]   │   "
            + "★★★ⓢ = safety-capped[/]"
            + partial_note,
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
    from tool_eval_bench.storage.db import RunRepository

    repo = RunRepository()
    runs = repo.list(limit=limit)
    repo.close()

    if not runs:
        console.print("\n  [dim]No benchmark runs found.[/]\n")
        return

    rows = _extract_leaderboard_rows(runs)

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
                "categories": {
                    cat: row["cat_scores"].get(cat) for cat in cat_order
                },
            }
            if row.get("token_efficiency") is not None:
                entry["token_efficiency"] = row["token_efficiency"]
            if row.get("deployability") is not None:
                entry["deployability"] = row["deployability"]
            if row.get("median_turn_ms") is not None:
                entry["median_turn_ms"] = row["median_turn_ms"]
            # Issue #6 metadata fields
            for meta_key in ["tool_version", "engine_name", "engine_version",
                             "quantization", "max_model_len", "temperature",
                             "server_model_root"]:
                if row.get(meta_key) is not None:
                    entry[meta_key] = row[meta_key]
            export_data.append(entry)

        result = json.dumps(export_data, indent=2, default=str)
        if output:
            with open(output, "w") as f:
                f.write(result)
            console.print(f"\n  [green]✓[/] Exported {len(rows)} models to [bold]{output}[/]\n")
        else:
            print(result)

    elif fmt == "csv":
        headers = [
            "rank", "model", "run_id", "date", "final_score", "rating",
            "total_points", "max_points", "passes", "partials", "fails",
            "scenario_count", "backend", "total_tokens", "num_runs",
            "tool_version", "engine_name", "engine_version",
            "quantization", "max_model_len", "temperature",
            "server_model_root",
        ]
        headers.extend(f"cat_{cat}" for cat in cat_order)

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            csv_row: dict[str, Any] = {
                "rank": idx,
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
            }
            for cat in cat_order:
                csv_row[f"cat_{cat}"] = row["cat_scores"].get(cat, "")
            writer.writerow(csv_row)

        result = buf.getvalue()
        if output:
            with open(output, "w") as f:
                f.write(result)
            console.print(f"\n  [green]✓[/] Exported {len(rows)} models to [bold]{output}[/]\n")
        else:
            print(result)
    else:
        console.print(f"\n  [red]Unknown format: {fmt}. Use 'csv' or 'json'.[/]\n")
        sys.exit(1)
