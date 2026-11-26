"""CLI runner for the Toolscale tool-count x similarity sweep."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from tool_eval_bench.cli.headless import report_run_failed
from tool_eval_bench.cli.plugin_lifecycle import (
    ProgressCallback,
    finalize_plugin_run,
    items_per_minute,
    run_plugin_session,
)
from tool_eval_bench.cli.plugin_progress import (
    PluginProgressDisplay,
    final_tally_line,
    status_icon,
    tally_line,
)
from tool_eval_bench.domain.models import RunContext

# OpenAI's Chat Completions API rejects requests with more tools than this.
OPENAI_TOOL_CAP = 128

# A catalog tool definition costs about this many prompt tokens. Used only to
# stretch the request timeout for long tool lists, as the needle runner does
# for long haystacks.
_TOKENS_PER_TOOL_ESTIMATE = 135

# Console wording for each condition-axis effect: (details key, name, contrast).
_TOOLSCALE_EFFECTS = (
    ("cue_effects", "Cue effect", "literal minus implied"),
    ("density_effects", "Density effect", "sparse minus crowded"),
    ("description_effects", "Description effect", "clean minus {versus}"),
)


def toolscale_timeout(requested: float, largest_count: int) -> float:
    """The per-request timeout, stretched so a long tool list can prefill."""
    return max(requested, 120.0 + largest_count * _TOKENS_PER_TOOL_ESTIMATE / 50_000 * 60.0)


def _toolscale_list(value: str, flag: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError(f"{flag} needs at least one value")
    return items


def print_toolscale_effects(console: Console, details: Mapping[str, Any]) -> None:
    """One line per condition contrast: near tier, at the largest N it was measured."""
    present = {
        "cue": details.get("cues", ["literal"]),
        "density": details.get("densities", ["sparse"]),
        "descriptions": details.get("descriptions", ["clean"]),
    }
    baseline = {"cue": "literal", "density": "sparse", "descriptions": "clean"}
    for key, name, contrast in _TOOLSCALE_EFFECTS:
        effects = [e for e in details.get(key, []) if e["similarity"] == "near"]
        # The largest N per slice: a sparse slice stops at the sparse maximum
        # while a crowded one can run further, and both deserve a line.
        largest: dict[tuple[Any, ...], int] = {}
        for e in effects:
            slice_key = tuple(e.get(a) for a in ("cue", "density", "descriptions", "versus"))
            largest[slice_key] = max(largest.get(slice_key, 0), e["count"])
        for e in effects:
            slice_key = tuple(e.get(a) for a in ("cue", "density", "descriptions", "versus"))
            if e["count"] != largest[slice_key]:
                continue
            context = ", ".join(
                e.get(axis, baseline[axis])
                for axis in ("cue", "density", "descriptions")
                if axis in e and len(present[axis]) > 1
            )
            where = f"{context}, near" if context else "near"
            console.print(
                f"  [bold]{name}[/] ({where}, N={e['count']}): {e['delta']:+.1f} pts "
                f"[dim]({contrast.format(versus=e['versus'])}; 95% CI "
                f"{e['ci_low']:+.1f} to {e['ci_high']:+.1f})[/]"
            )


def print_toolscale_grid(console: Console, result: Any) -> None:
    """Print accuracy-when-called and call rate per cell, one row per tier and condition."""
    details = result.details
    counts: list[int] = details.get("counts", [])
    levels: list[str] = details.get("similarities", [])
    cues: list[str] = details.get("cues", ["literal"])
    densities: list[str] = details.get("densities", ["sparse"])
    described: list[str] = details.get("descriptions", ["clean"])
    if not counts or not levels:
        return
    cells = {
        (
            c["count"],
            c["similarity"],
            c["cue"],
            c.get("density", "sparse"),
            c.get("descriptions", "clean"),
        ): c
        for c in details.get("cells", [])
    }

    # Two numbers per cell in an 80-column terminal: keep them bare and say
    # what they are once, in the title.
    table = Table(
        title="Accuracy when called / call rate (%)",
        title_style="bold",
        border_style="bright_blue",
    )
    table.add_column("Cue", style="dim")
    if len(densities) > 1:
        table.add_column("Fill", style="dim")
    if len(described) > 1:
        table.add_column("Desc", style="dim")
    table.add_column("Tier", style="dim")
    for count in counts:
        table.add_column(f"N={count}", justify="right")
    for cue in cues:
        for density in densities:
            for level_of_description in described:
                for level in levels:
                    labels = [cue]
                    if len(densities) > 1:
                        labels.append(density)
                    if len(described) > 1:
                        labels.append(level_of_description)
                    labels.append(level)
                    row = []
                    for count in counts:
                        cell = cells.get((count, level, cue, density, level_of_description))
                        if cell is None:
                            row.append("")
                            continue
                        acc = cell["accuracy_when_called"]
                        colour = "green" if acc >= 90 else "yellow" if acc >= 70 else "red"
                        row.append(f"[{colour}]{acc:.0f}[/][dim]/{cell['call_rate']:.0f}[/]")
                    table.add_row(*labels, *row)
    console.print(table)


def run_toolscale_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: RunContext | None = None,
    openai_tool_cap: int = OPENAI_TOOL_CAP,
) -> None:
    """Run the tool-count x similarity selection sweep and display the grid."""
    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.plugins.toolscale.catalog import (
        CatalogError,
        lint_catalog,
        load_builtin_catalog,
        load_catalog_file,
    )
    from tool_eval_bench.plugins.toolscale.grid import GridError, build_items
    from tool_eval_bench.plugins.toolscale.plugin import ToolscalePlugin

    seed = getattr(args, "seed", None)
    trials = max(1, getattr(args, "trials", 1) or 1)
    catalog_path = getattr(args, "toolscale_catalog", None)
    limit = args.toolscale_limit or None
    try:
        counts = [int(c) for c in _toolscale_list(args.toolscale_counts, "--toolscale-counts")]
        levels = _toolscale_list(args.toolscale_similarity, "--toolscale-similarity")
        cues = _toolscale_list(args.toolscale_cues, "--toolscale-cues")
        densities = _toolscale_list(args.toolscale_fillers, "--toolscale-fillers")
        described = _toolscale_list(args.toolscale_descriptions, "--toolscale-descriptions")
        catalog = load_catalog_file(catalog_path) if catalog_path else load_builtin_catalog()
        items = build_items(
            catalog,
            counts=counts,
            similarities=levels,
            limit=limit,
            trials=trials,
            seed=seed if seed is not None else 0,
            cues=cues,
            densities=densities,
            descriptions=described,
        )
    except (CatalogError, GridError, ValueError) as exc:
        report_run_failed(console, f"\n[bold red]Toolscale error:[/] {escape(str(exc))}")
        sys.exit(1)

    if catalog.is_custom:
        # A private catalog may break the authoring conventions on purpose, so
        # problems are warnings. Names stay out of the console for the same
        # reason the report withholds them.
        problems = lint_catalog(catalog)
        if problems:
            console.print(
                f"  [yellow]⚠ {len(problems)} catalog lint finding(s): items may be "
                "ambiguous. Run lint_catalog() on the file to see them.[/]"
            )

    counts = sorted({item.count for item in items})
    if counts[-1] > openai_tool_cap:
        console.print(
            f"  [yellow]⚠ N up to {counts[-1]}: some providers cap tools per request "
            f"(OpenAI: {openai_tool_cap}). A rejected request is recorded as an error, "
            "not a wrong pick. Local OpenAI-compatible servers have no tool cap.[/]"
        )
    levels = [
        lvl for lvl in ("near", "related", "distinct") if any(i.similarity == lvl for i in items)
    ]
    # Conditions come from the grid items: the N=1 control is always sparse.
    grid_items = [i for i in items if i.count > 1] or items
    cues = [cue for cue in ("literal", "implied") if any(i.cue == cue for i in grid_items)]
    densities = [d for d in ("sparse", "crowded") if any(i.density == d for i in grid_items)]
    described = [
        level
        for level in ("clean", "terse", "bare")
        if any(i.descriptions == level for i in grid_items)
    ]
    if len(described) > 1 and "clean" not in described:
        console.print(
            "  [yellow]⚠ Description effects compare against clean descriptions; add clean "
            "to --toolscale-descriptions to measure them.[/]"
        )
    targets = len({item.target_rank for item in items})
    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""
    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]N = {', '.join(map(str, counts))} × {', '.join(levels)} × "
            f"{', '.join(cues)} cue · {', '.join(densities)} fillers · "
            f"{', '.join(described)} descriptions · {targets} targets · "
            f"{trials} trial(s) = "
            f"{len(items)} requests · "
            f"catalog v{catalog.version} ({catalog.source}){parallel_label}[/]",
            title="[bold]🧰 Toolscale — Tool Count × Similarity[/]",
            border_style="bright_blue",
        )
    )

    plugin = ToolscalePlugin()
    adapter = build_adapter(base_url, **adapter_options(args))

    def _format_progress(
        display: PluginProgressDisplay, current: int, item_info: dict[str, Any]
    ) -> tuple[str, str]:
        label = (
            f"N{item_info.get('count')}/{item_info.get('similarity')}/"
            f"{item_info.get('cue')}/{item_info.get('density')}/"
            f"{item_info.get('descriptions')}"
            if catalog.is_custom
            else item_info.get("item_id")
        )
        return (
            tally_line(
                display.tally,
                rate=display.rate_per_minute(current),
                unit="req/min",
                accent="blue",
            ),
            f"  {status_icon(item_info)} [bold]{escape(str(label))}[/]  "
            f"[dim]{item_info.get('outcome')}[/]",
        )

    def _format_finish(res: Any) -> tuple[int, str]:
        d = res.details
        # The tally describes the scored items (the grid, or the control when it
        # ran alone), so its counts agree with the score.
        return (
            d["total"],
            final_tally_line(
                correct=d["grid_correct"],
                wrong=d["grid_items"] - d["grid_correct"] - d["grid_errors"],
                errors=d["grid_errors"],
                score=d["accuracy"],
                rate=items_per_minute(d["total"], res.duration_seconds),
                unit="req/min",
                accent="blue",
            ),
        )

    async def _run_plugin(on_progress: ProgressCallback) -> Any:
        return await plugin.run(
            adapter,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout_seconds=toolscale_timeout(args.timeout, counts[-1]),
            seed=seed,
            extra_params=extra_params,
            on_progress=on_progress,
            items=items,
            catalog=catalog,
            concurrency=parallel,
        )

    result = run_plugin_session(
        console,
        benchmark_name="Toolscale",
        adapter=adapter,
        total=len(items),
        run_plugin=_run_plugin,
        format_progress=_format_progress,
        format_finish=_format_finish,
    )
    if result is None:
        return
    details = result.details

    console.print()
    print_toolscale_grid(console, result)
    console.print()
    console.print(
        f"  [bold]Accuracy when called:[/] [bold green]{details['accuracy_when_called']:.1f}%[/] "
        f"({details['grid_correct']}/{details['grid_called']})  ·  "
        f"[bold]Call rate:[/] [bold cyan]{details['call_rate']:.1f}%[/] "
        f"({details['grid_called']}/{details['grid_items']})"
    )
    for axis in ("by_cue", "by_density", "by_descriptions"):
        breakdown = details.get(axis, {})
        if len(breakdown) > 1:
            for value, h in breakdown.items():
                console.print(
                    f"    [dim]{value}:[/] {h['accuracy_when_called']:.1f}% when called · "
                    f"{h['call_rate']:.1f}% call rate"
                )
    print_toolscale_effects(console, details)
    for cue, control in (details.get("control") or {}).items():
        console.print(
            f"  [bold]Control (N=1, {cue}):[/] {control['accuracy']:.1f}% "
            f"({control['correct']}/{control['items']})"
        )
    console.print(f"  [dim]Combined accuracy (stored score): {details['accuracy']:.1f}%[/]")
    errs = details.get("errors", 0)
    if errs > 0:
        console.print(f"  [bold yellow]⚠ {errs} errors[/] (counted as misses)")
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    finalize_plugin_run(
        console,
        mode="toolscale",
        title="Toolscale",
        display_name=display_name,
        result=result,
        config={
            "model": model,
            "base_url": base_url,
            "mode": "toolscale",
            # A catalog edit or a different grid must never share a fingerprint
            # with older runs.
            "catalog_version": catalog.version,
            "catalog_source": catalog.source,
            "catalog_sha256": catalog.sha256,
            "counts": counts,
            "similarities": levels,
            "cues": cues,
            "densities": densities,
            "descriptions": described,
            "targets": targets,
            "trials": trials,
            "temperature": args.temperature,
            "seed": seed,
            "extra_params": extra_params or None,
        },
        report_metrics=[
            f"- **Accuracy when called**: **{details['accuracy_when_called']:.1f}%**",
            f"- **Call rate**: **{details['call_rate']:.1f}%**",
            f"- **Combined accuracy**: {details['accuracy']:.1f}%",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=plugin.render_report_section(result),
        output_dir=output_dir,
        label=getattr(args, "label", None),
        run_context=run_context,
    )
