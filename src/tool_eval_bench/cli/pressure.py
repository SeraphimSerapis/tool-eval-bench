"""Context-pressure sweep runner for the CLI.

Extracted from the monolithic ``cli/bench.py``. Provides ``run_pressure_sweep``,
which runs a set of scenarios at increasing context-fill ratios and reports
the breaking point where all scenarios start to fail.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import sys
from typing import Any

from rich.console import Console

from tool_eval_bench.application.mode_runs import ModeRun, finalize_mode_run
from tool_eval_bench.cli.headless import report_run_failed, report_run_saved
from tool_eval_bench.cli.resolve import parse_sweep_range, redact_url, resolve_scenarios
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.storage.reports.pressure import pressure_sweep_report
from tool_eval_bench.utils.urls import redact_urls

logger = logging.getLogger(__name__)


async def _run_pressure_level(
    adapter: Any,
    run_scenarios: Any,
    **scenario_kwargs: Any,
) -> Any:
    """Run one sweep level and always release its adapter."""
    try:
        return await run_scenarios(adapter, **scenario_kwargs)
    finally:
        aclose = getattr(adapter, "aclose", None)
        if callable(aclose):
            await aclose()


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
    label: str | None = None,
    run_context: RunContext | None = None,
) -> None:
    """Run scenarios at increasing context pressure and report breaking point."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.runner.context_pressure import (
        _RESERVED_FOR_OUTPUT,
        _RESERVED_FOR_SCENARIO,
        _TOKENS_PER_FILLER_CHUNK,
        ContextPressureConfig,
        build_pressure_messages,
        calibrate_pressure_messages,
        compute_fill_budget,
        detect_context_size,
        detect_kv_capacity,
        reported_context_window,
    )
    from tool_eval_bench.runner.orchestrator import run_all_scenarios

    # Parse range
    try:
        start, end = parse_sweep_range(args.context_pressure_sweep)
    except ValueError as exc:
        report_run_failed(console, f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    steps = max(2, args.sweep_steps)
    levels = [start + i * (end - start) / (steps - 1) for i in range(steps)]
    levels = [round(lv, 4) for lv in levels]

    scenarios = resolve_scenarios(args)
    if not scenarios:
        report_run_failed(console, "[bold red]Error:[/] No scenarios matched.")
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
            context_size = asyncio.run(
                detect_context_size(
                    base_url,
                    model,
                    api_key,
                    client_factory=HTTPMeasurementClient,
                    reported_context=reported_context_window(run_context),
                )
            )
        if context_size is None:
            report_run_failed(
                console,
                "[bold red]Error:[/] Could not auto-detect context size. "
                "Use --context-size to specify it.",
            )
            sys.exit(1)

        if args.context_size is None:
            kv_info = asyncio.run(
                detect_kv_capacity(
                    base_url,
                    api_key,
                    metrics_url=getattr(args, "metrics_url", None),
                    client_factory=HTTPMeasurementClient,
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
        report_run_failed(console, f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    top_fill = compute_fill_budget(context_size, end)
    if top_fill < _TOKENS_PER_FILLER_CHUNK:
        report_run_failed(
            console,
            f"[bold red]Error:[/] Context window {context_size:,} tokens is too small for a "
            f"pressure sweep: {_RESERVED_FOR_OUTPUT + _RESERVED_FOR_SCENARIO:,} tokens are "
            f"reserved for output and the scenario, leaving {top_fill:,} filler tokens at "
            f"{end:.0%}, less than one {_TOKENS_PER_FILLER_CHUNK:,}-token filler chunk. "
            "Check --context-size; on llama.cpp the window is per slot (-c divided by "
            "--parallel).",
        )
        sys.exit(1)

    _STATUS_EMOJI = {
        "pass": "✅",
        "partial": "⚠️ ",
        "fail": "❌",
    }
    _EXCLUDED_EMOJI = "➖"

    level_results: list[dict[str, Any]] = []
    consecutive_all_fail = 0
    consecutive_unscored = 0
    interrupted = False
    stop_reason: str | None = None

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

            estimated: list[bool] = []
            _loop = _aio.new_event_loop()
            try:
                pressure_messages, actual_fill = _loop.run_until_complete(
                    calibrate_pressure_messages(
                        pressure_messages,
                        fill_tokens,
                        base_url,
                        model,
                        api_key,
                        client_factory=HTTPMeasurementClient,
                        seed=level_seed,
                        on_estimated=functools.partial(estimated.append, True),
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

            adapter = build_adapter(base_url, **adapter_options(args))

            try:
                summary = asyncio.run(
                    _run_pressure_level(
                        adapter,
                        run_all_scenarios,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        scenarios=scenarios,
                        temperature=0.0,
                        timeout_seconds=effective_timeout,
                        extra_params=extra_params,
                        context_pressure_messages=pressure_messages,
                        system_prompt=getattr(args, "system_prompt", None),
                    )
                )

                results_map: dict[str, str] = {}
                for sr in summary.scenario_results:
                    results_map[sr.scenario_id] = sr.status.value
                # Infrastructure failures (timeouts, connection errors, 5xx, and
                # TC-45's exclusion on an endpoint that ignores tool_choice) say
                # nothing about the model at this fill. As in scored runs they
                # leave the pass rate, the breaking point, and the early stop.
                excluded = {
                    sr.scenario_id
                    for sr in summary.scenario_results
                    if sr.is_infrastructure_failure
                }

                pass_count = sum(1 for s in results_map.values() if s == "pass")
                scored_total = len(scenarios) - len(excluded)
                score_pct: float | None = (
                    pass_count / scored_total * 100 if scored_total > 0 else None
                )
                level: dict[str, Any] = {
                    "ratio": ratio,
                    "results": results_map,
                    "scenario_results": [result.to_dict() for result in summary.scenario_results],
                    "score_pct": score_pct,
                    "pass_count": pass_count,
                    "fill_tokens": fill_tokens,
                    "excluded_count": len(excluded),
                    "excluded_scenarios": [sid for sid in scenario_ids if sid in excluded],
                }
            except Exception as exc:
                # The whole level failed before any scenario could be scored:
                # an infrastructure failure, not evidence about the model.
                error = redact_urls(str(exc))
                excluded = set(scenario_ids)
                pass_count = 0
                scored_total = 0
                score_pct = None
                level = {
                    "ratio": ratio,
                    "results": {sid: "fail" for sid in scenario_ids},
                    "scenario_results": [
                        {
                            "scenario_id": sid,
                            "status": "fail",
                            "points": 0,
                            "summary": f"Sweep level failed: {error}",
                            "note": None,
                            "tool_calls_made": [],
                            "expected_behavior": "",
                            "duration_seconds": 0.0,
                            "turn_count": 0,
                            "raw_log": error,
                        }
                        for sid in scenario_ids
                    ],
                    "score_pct": score_pct,
                    "pass_count": pass_count,
                    "fill_tokens": fill_tokens,
                    "excluded_count": len(excluded),
                    "excluded_scenarios": list(scenario_ids),
                    "error": error,
                }
                console.print(f"[red]error: {exc}[/]")
            else:
                emoji_str = "  ".join(
                    _EXCLUDED_EMOJI
                    if sid in excluded
                    else _STATUS_EMOJI.get(results_map.get(sid, "fail"), "❌")
                    for sid in scenario_ids
                )
                score_text = f"{score_pct:.0f}%" if score_pct is not None else "n/a"
                excluded_text = f"  [dim]({len(excluded)} excluded)[/]" if excluded else ""
                console.print(f"{emoji_str}  [bold]{score_text}[/]{excluded_text}")

            if estimated:
                # No usable /tokenize: the level ran on the 4 chars/token
                # estimate, so its real fill is unmeasured.
                level["fill_tokens_estimated"] = True
            level_results.append(level)

            # A level with nothing scored is neither a pass nor a fail, so it
            # leaves the all-fail run alone; two in a row mean the endpoint,
            # not the model, has stopped answering.
            if scored_total == 0:
                consecutive_unscored += 1
            else:
                consecutive_unscored = 0
                consecutive_all_fail = consecutive_all_fail + 1 if pass_count == 0 else 0

            if consecutive_all_fail >= 2:
                stop_reason = "2 consecutive all-fail levels"
            elif consecutive_unscored >= 2:
                stop_reason = "2 consecutive levels with nothing scored"
            if stop_reason is not None:
                console.print(f"  [dim]··· stopped ({stop_reason})[/]")
                break

    except KeyboardInterrupt:
        interrupted = True
        if level_results:
            console.print("\n[bold red]Interrupted.[/]")

    if not level_results:
        # Nothing to save or report, so the run failed; under --json this is
        # the run_failed event a script waits for, not a log line and exit 0.
        reason = "Interrupted. No results collected." if interrupted else "No results collected."
        report_run_failed(console, f"\n[bold red]{reason}[/]")
        sys.exit(1)

    console.print()

    lines: list[str] = []
    breaking_point: float | None = None
    first_degradation: float | None = None

    for lr in level_results:
        ratio = lr["ratio"]
        score = lr["score_pct"]
        excluded = set(lr["excluded_scenarios"])
        emoji_str = "  ".join(
            _EXCLUDED_EMOJI
            if sid in excluded
            else _STATUS_EMOJI.get(lr["results"].get(sid, "fail"), "❌")
            for sid in scenario_ids
        )
        if score is None:
            lines.append(f"  [bold]{ratio:>4.0%}[/]  {emoji_str}   n/a  {'░' * 20}")
            continue
        bar_len = int(score / 100 * 20)
        if score >= 100:
            bar_color = "green"
        elif score >= 50:
            bar_color = "yellow"
        else:
            bar_color = "red"
        bar = f"[{bar_color}]{'█' * bar_len}[/]{'░' * (20 - bar_len)}"

        lines.append(f"  [bold]{ratio:>4.0%}[/]  {emoji_str}   {score:>3.0f}%  {bar}")

        scored = [v for sid, v in lr["results"].items() if sid not in excluded]
        all_pass = all(v == "pass" for v in scored)
        if all_pass:
            breaking_point = ratio
        if first_degradation is None and not all_pass:
            first_degradation = ratio

    # Levels never reached could still pass, so the highest passing level so
    # far is only a lower bound. An observed degradation stays real.
    if interrupted:
        breaking_point = None

    lines.append("")
    if interrupted:
        lines.append(
            f"  [bold red]Breaking point:[/] withheld (interrupted after "
            f"{len(level_results)} of {len(levels)} levels)"
        )
    elif breaking_point is not None:
        lines.append(f"  [bold green]Breaking point:[/] {breaking_point:.0%} (all scenarios pass)")
    elif all(lr["score_pct"] is None for lr in level_results):
        lines.append("  [bold red]Breaking point:[/] n/a (no level was scored)")
    else:
        lines.append("  [bold red]Breaking point:[/] none (no level had all scenarios pass)")
    if first_degradation is not None:
        lines.append(
            f"  [bold yellow]Degradation:[/]    {first_degradation:.0%} (first partial/fail)"
        )
    if stop_reason is not None:
        lines.append(f"  [dim]Stopped early: {stop_reason}[/]")
    if any(lr["excluded_scenarios"] for lr in level_results):
        lines.append(
            f"  [dim]{_EXCLUDED_EMOJI} excluded from scoring (timeout, server or connection "
            "error, or not hostable on this endpoint)[/]"
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

    sweep_fields: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "mode": "context-pressure-sweep",
        "start": start,
        "end": end,
        "steps": steps,
        # The effective window, after --context-size and the KV-capacity cap:
        # it sets every level's fill, as the seed sets its filler text.
        "context_size": context_size,
        "seed": args.seed,
        "scenarios": scenario_ids,
    }
    # Present only for an override, as in a scored run's config.
    system_prompt = getattr(args, "system_prompt", None)
    if system_prompt is not None:
        sweep_fields["system_prompt"] = system_prompt
    # Status stays "completed" even when interrupted: history and resume treat
    # any other status as a resumable scored run.
    run = ModeRun(
        run_type="context-pressure",
        config=sweep_fields,
        scores={
            "levels": len(level_results),
            "planned_levels": len(levels),
            "interrupted": interrupted,
            "stop_reason": stop_reason,
            "breaking_point": breaking_point,
            "first_degradation": first_degradation,
            "level_results": level_results,
        },
        status="completed",
    )
    finalized = finalize_mode_run(
        run,
        pressure_sweep_report(
            model=display_name,
            backend=backend,
            # The report is a shareable artifact: redacted whatever --redact-url says.
            server=redact_url(base_url),
            context_size=context_size,
            level_results=level_results,
            breaking_point=breaking_point,
            first_degradation=first_degradation,
            label=label,
            planned_levels=len(levels),
            interrupted=interrupted,
            stop_reason=stop_reason,
        ),
        run_context=run_context,
        output_dir=getattr(args, "output_dir", None),
    )
    report_run_saved(console, run, finalized)
    report_path = finalized.report_path
    console.print(f"  [dim]Report saved to {report_path}[/]\n")
