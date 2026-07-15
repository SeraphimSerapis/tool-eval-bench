"""Context-pressure sweep runner for the CLI.

Extracted from the monolithic ``cli/bench.py``. Provides ``run_pressure_sweep``,
which runs a set of scenarios at increasing context-fill ratios and reports
the breaking point where all scenarios start to fail.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

logger = logging.getLogger(__name__)


def _trace_block(raw_log: str) -> list[str]:
    """Return a Markdown fence long enough to preserve the complete raw trace."""
    longest = max((len(match) for match in re.findall(r"`+", raw_log)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", raw_log or "(empty trace)", fence]


def _write_pressure_sweep_report(
    *,
    run_id: str,
    model: str,
    backend: str,
    display_url: str,
    context_size: int,
    level_results: list[dict[str, Any]],
    breaking_point: float | None,
    first_degradation: float | None,
    output_dir: str | None,
) -> Path:
    """Write a trace-complete Markdown artifact for a context-pressure sweep."""
    from tool_eval_bench.storage.reports import MarkdownReporter

    reporter = MarkdownReporter(root=output_dir)
    now = datetime.now(timezone.utc)
    folder = reporter.root / f"{now.year:04d}" / f"{now.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.md"
    markdown = [
        f"# Context Pressure Sweep — {model}",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Date**: `{now.isoformat()}`",
        "- **Mode**: context-pressure-sweep",
        f"- **Backend**: {backend}",
        f"- **Server**: {display_url}",
        f"- **Context Window**: {context_size:,} tokens",
        f"- **Executed Levels**: {len(level_results)}",
        f"- **Breaking Point**: {breaking_point:.0%}"
        if breaking_point is not None
        else "- **Breaking Point**: none",
        f"- **First Degradation**: {first_degradation:.0%}"
        if first_degradation is not None
        else "- **First Degradation**: none",
        "",
    ]
    for index, level in enumerate(level_results, start=1):
        markdown.extend(
            [
                f"## Level {index} — {level['ratio']:.0%}",
                "",
                f"- **Fill Tokens**: {level['fill_tokens']:,}",
                f"- **Pass Rate**: {level['score_pct']:.1f}%",
            ]
        )
        if level.get("error"):
            markdown.append(f"- **Level Error**: {level['error']}")
        markdown.append("")
        for scenario in level["scenario_results"]:
            markdown.extend(
                [
                    f"### {scenario['scenario_id']}",
                    "",
                    f"- **Status**: {scenario['status']}",
                    f"- **Points**: {scenario['points']} / 2",
                    f"- **Summary**: {scenario.get('summary') or ''}",
                    f"- **Expected**: {scenario.get('expected_behavior') or ''}",
                    f"- **Tool Calls**: {', '.join(scenario.get('tool_calls_made') or []) or 'none'}",
                    "",
                    "#### Full trace",
                    "",
                    *_trace_block(scenario.get("raw_log", "")),
                    "",
                ]
            )
    path.write_text("\n".join(markdown), encoding="utf-8")
    return path


def _report_then_persist_pressure_sweep(
    *,
    report_kwargs: dict[str, Any],
    run_data: dict[str, Any],
    persist_plugin_run: Any,
) -> Path:
    """Enforce artifact-first completed-run finalization for pressure sweeps."""
    report_path = _write_pressure_sweep_report(**report_kwargs)
    persist_plugin_run(run_data)
    return report_path


def run_pressure_sweep(
    console: Console,
    model: str,
    display_name: str,
    backend: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    display_url: str | None = None,
    extra_params: dict[str, Any] | None = None,
    parse_sweep_range: Any = None,
    resolve_scenarios: Any = None,
    with_config_fingerprint: Any = None,
    persist_plugin_run: Any = None,
    metadata_for_storage: Any = None,
) -> None:
    """Run scenarios at increasing context pressure and report breaking point."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
    from tool_eval_bench.runner.context_pressure import (
        ContextPressureConfig,
        build_pressure_messages,
        calibrate_pressure_messages,
        compute_fill_budget,
        detect_context_size,
        detect_kv_capacity,
    )
    from tool_eval_bench.runner.orchestrator import run_all_scenarios

    # Parse range
    try:
        start, end = parse_sweep_range(args.context_pressure_sweep)
    except ValueError as exc:
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    steps = max(2, args.sweep_steps)
    levels = [start + i * (end - start) / (steps - 1) for i in range(steps)]
    levels = [round(lv, 4) for lv in levels]

    scenarios = resolve_scenarios(args)
    if not scenarios:
        console.print("[bold red]Error:[/] No scenarios matched.")
        sys.exit(1)

    scenario_ids = [s.id for s in scenarios]

    console.print(f"\n[bold]⚡ Context Pressure Sweep[/] — {display_name}")
    console.print(f"[dim]  Backend: {backend}  |  Server: {display_url or base_url}[/]")
    console.print(
        f"[dim]  Range: {start:.0%} → {end:.0%}  |  "
        f"{len(levels)} levels  |  "
        f"{len(scenarios)} scenario{'s' if len(scenarios) != 1 else ''}[/]\n"
    )

    # Detect context size once
    try:
        context_size: int | None = args.context_size
        if context_size is None:
            context_size = asyncio.run(detect_context_size(base_url, model, api_key))
        if context_size is None:
            console.print(
                "[bold red]Error:[/] Could not auto-detect context size. "
                "Use --context-size to specify it."
            )
            sys.exit(1)

        if args.context_size is None:
            kv_info = asyncio.run(
                detect_kv_capacity(
                    base_url, api_key, metrics_url=getattr(args, "metrics_url", None)
                )
            )
            if kv_info is not None and kv_info.is_hybrid:
                console.print(
                    f"  [dim]ℹ Hybrid model detected — trusting "
                    f"max_model_len ({context_size:,} tokens)[/]"
                )
            elif kv_info is not None and kv_info.capacity < context_size:
                console.print(
                    f"  [dim]⚠ KV cache capacity ({kv_info.capacity:,} tokens) < "
                    f"max_model_len ({context_size:,}) — capping[/]"
                )
                context_size = kv_info.capacity

        console.print(f"  [dim]Context window: {context_size:,} tokens[/]\n")
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    _STATUS_EMOJI = {
        "pass": "✅",
        "partial": "⚠️ ",
        "fail": "❌",
    }

    level_results: list[dict[str, Any]] = []
    consecutive_all_fail = 0

    try:
        for level_idx, ratio in enumerate(levels):
            fill_tokens = compute_fill_budget(context_size, ratio)
            cfg = ContextPressureConfig(
                ratio=ratio,
                fill_tokens=fill_tokens,
                detected_context=context_size,
            )

            level_seed: int | None = None
            if args.seed is not None:
                level_seed = args.seed + level_idx
            pressure_messages = build_pressure_messages(
                cfg,
                seed=level_seed,
            )

            import asyncio as _aio

            _loop = _aio.new_event_loop()
            try:
                pressure_messages, actual_fill = _loop.run_until_complete(
                    calibrate_pressure_messages(
                        pressure_messages,
                        fill_tokens,
                        base_url,
                        model,
                        api_key,
                        seed=level_seed,
                    )
                )
            finally:
                _loop.close()

            n_msg_pairs = len(pressure_messages) // 2
            cal_delta = actual_fill - fill_tokens
            logger.info(
                "Sweep %d/%d: ratio=%.4f fill_target=%d actual=%d delta=%+d msg_pairs=%d",
                level_idx + 1,
                len(levels),
                ratio,
                fill_tokens,
                actual_fill,
                cal_delta,
                n_msg_pairs,
            )

            base_timeout = getattr(args, "timeout", 60.0)
            fill_scaling = max(0, fill_tokens / 50_000) * 60.0
            effective_timeout = max(base_timeout, 120.0 + fill_scaling)

            pct_done = (level_idx + 1) / len(levels)
            bar_filled = int(pct_done * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            console.print(
                f"  [bold cyan]⚡[/] Sweep {level_idx + 1}/{len(levels)}: "
                f"[bold]{ratio:>4.0%}[/] pressure  {bar}  ",
                end="",
            )

            adapter = OpenAICompatibleAdapter()
            try:
                summary = asyncio.run(
                    run_all_scenarios(
                        adapter,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        scenarios=scenarios,
                        temperature=0.0,
                        timeout_seconds=effective_timeout,
                        extra_params=extra_params,
                        context_pressure_messages=pressure_messages,
                    )
                )

                results_map: dict[str, str] = {}
                for sr in summary.scenario_results:
                    results_map[sr.scenario_id] = sr.status.value

                pass_count = sum(1 for s in results_map.values() if s == "pass")
                total = len(scenarios)
                score_pct = (pass_count / total * 100) if total else 0

                emoji_str = "  ".join(
                    _STATUS_EMOJI.get(results_map.get(sid, "fail"), "❌") for sid in scenario_ids
                )
                console.print(f"{emoji_str}  [bold]{score_pct:.0f}%[/]")

                level_results.append(
                    {
                        "ratio": ratio,
                        "results": results_map,
                        "scenario_results": [
                            result.to_dict() for result in summary.scenario_results
                        ],
                        "score_pct": score_pct,
                        "pass_count": pass_count,
                        "fill_tokens": fill_tokens,
                    }
                )

                if pass_count == 0:
                    consecutive_all_fail += 1
                else:
                    consecutive_all_fail = 0

                if consecutive_all_fail >= 2:
                    console.print("  [dim]··· stopped (2 consecutive all-fail levels)[/]")
                    break

            except Exception as exc:
                console.print(f"[red]error: {exc}[/]")
                level_results.append(
                    {
                        "ratio": ratio,
                        "results": {sid: "fail" for sid in scenario_ids},
                        "scenario_results": [
                            {
                                "scenario_id": sid,
                                "status": "fail",
                                "points": 0,
                                "summary": f"Sweep level failed: {exc}",
                                "note": None,
                                "tool_calls_made": [],
                                "expected_behavior": "",
                                "duration_seconds": 0.0,
                                "turn_count": 0,
                                "raw_log": str(exc),
                            }
                            for sid in scenario_ids
                        ],
                        "score_pct": 0,
                        "pass_count": 0,
                        "fill_tokens": fill_tokens,
                        "error": str(exc),
                    }
                )
                consecutive_all_fail += 1
                if consecutive_all_fail >= 2:
                    console.print("  [dim]··· stopped (2 consecutive all-fail levels)[/]")
                    break

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted.[/]")

    if not level_results:
        console.print("\n[bold red]No results collected.[/]")
        return

    console.print()

    lines: list[str] = []
    breaking_point: float | None = None
    first_degradation: float | None = None

    for lr in level_results:
        ratio = lr["ratio"]
        score = lr["score_pct"]
        emoji_str = "  ".join(
            _STATUS_EMOJI.get(lr["results"].get(sid, "fail"), "❌") for sid in scenario_ids
        )
        bar_len = int(score / 100 * 20)
        if score >= 100:
            bar_color = "green"
        elif score >= 50:
            bar_color = "yellow"
        else:
            bar_color = "red"
        bar = f"[{bar_color}]{'█' * bar_len}[/]{'░' * (20 - bar_len)}"

        lines.append(f"  [bold]{ratio:>4.0%}[/]  {emoji_str}   {score:>3.0f}%  {bar}")

        all_pass = all(v == "pass" for v in lr["results"].values())
        if all_pass:
            breaking_point = ratio
        if first_degradation is None and not all_pass:
            first_degradation = ratio

    lines.append("")
    if breaking_point is not None:
        lines.append(f"  [bold green]Breaking point:[/] {breaking_point:.0%} (all scenarios pass)")
    else:
        lines.append("  [bold red]Breaking point:[/] none (no level had all scenarios pass)")
    if first_degradation is not None:
        lines.append(
            f"  [bold yellow]Degradation:[/]    {first_degradation:.0%} (first partial/fail)"
        )

    header = "  ".join(f"[dim]{sid}[/]" for sid in scenario_ids)
    lines.insert(0, f"  [dim]      {header}[/]")

    panel_content = "\n".join(lines)
    console.print(
        Panel(
            panel_content,
            title="[bold]⚡ Context Pressure Sweep Results[/]",
            border_style="bright_cyan",
            padding=(1, 1),
        )
    )
    console.print()

    from tool_eval_bench.utils.ids import build_run_id

    sweep_config = with_config_fingerprint(
        {
            "model": model,
            "base_url": base_url,
            "mode": "context-pressure-sweep",
            "start": start,
            "end": end,
            "steps": steps,
            "scenarios": scenario_ids,
        }
    )
    sweep_run_id = build_run_id(sweep_config)
    run_data = {
        "run_id": sweep_run_id,
        "run_type": "context-pressure",
        "status": "completed",
        "config": sweep_config,
        "scores": {
            "levels": len(level_results),
            "breaking_point": breaking_point,
            "first_degradation": first_degradation,
            "level_results": level_results,
        },
        "metadata": metadata_for_storage(None),
    }
    report_path = _report_then_persist_pressure_sweep(
        report_kwargs={
            "run_id": sweep_run_id,
            "model": display_name,
            "backend": backend,
            "display_url": display_url or base_url,
            "context_size": context_size,
            "level_results": level_results,
            "breaking_point": breaking_point,
            "first_degradation": first_degradation,
            "output_dir": getattr(args, "output_dir", None),
        },
        run_data=run_data,
        persist_plugin_run=persist_plugin_run,
    )
    console.print(f"  [dim]Report saved to {report_path}[/]\n")
