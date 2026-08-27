"""Shared lifecycle dispatch for external benchmark plugins."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rich.console import Console

from tool_eval_bench.cli.helpers import metadata_for_storage as _metadata_for_storage
from tool_eval_bench.cli.helpers import persist_plugin_run as _persist_plugin_run
from tool_eval_bench.cli.plugin_datasets import load_dataset_with_progress
from tool_eval_bench.cli.plugin_lifecycle import (
    execute_plugin as _execute_plugin_impl,
)
from tool_eval_bench.cli.plugin_lifecycle import (
    finalize_plugin_run as _finalize_plugin_run_impl,
)
from tool_eval_bench.cli.plugin_progress import (
    PluginProgressDisplay,
    final_tally_line,
    status_icon,
    tally_line,
    truncate,
)
from tool_eval_bench.cli.resolve import with_config_fingerprint as _with_config_fingerprint


def _execute_plugin(
    console: Console, benchmark_name: str, run: Callable[[], Any], result_holder: list[Any]
) -> Any | None:
    return _execute_plugin_impl(console, benchmark_name, run, result_holder)


def _finalize_plugin_run(
    *,
    mode: str,
    title: str,
    display_name: str,
    result: Any,
    config: dict[str, Any],
    report_metrics: list[str],
    report_lines: list[str],
    output_dir: str | None,
    run_context: Any | None,
) -> str:
    return _finalize_plugin_run_impl(
        mode=mode,
        title=title,
        display_name=display_name,
        result=result,
        config=config,
        report_metrics=report_metrics,
        report_lines=report_lines,
        output_dir=output_dir,
        run_context=run_context,
        with_config_fingerprint=_with_config_fingerprint,
        persist_plugin_run=_persist_plugin_run,
        metadata_for_storage=_metadata_for_storage,
    )


def _run_gsm8k_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: Any | None = None,
) -> None:
    """Run the GSM8K grade-school math benchmark and display results."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin

    n_shots = args.gsm8k_shots
    limit = args.gsm8k_limit
    shuffle = args.gsm8k_shuffle
    seed = getattr(args, "seed", None)
    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""
    limit_label = "all 1319" if limit == 0 else f"{limit}"

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]{n_shots}-shot CoT · {limit_label} questions"
            f"{' · shuffled' if shuffle else ''}{parallel_label}[/]",
            title="[bold]📐 GSM8K — Grade School Math[/]",
            border_style="bright_magenta",
        )
    )

    plugin = GSM8KPlugin()
    adapter = build_adapter(base_url, wire_format=getattr(args, "format", None))
    result_holder: list = []

    # -- Phase 1: Load dataset (with visible progress) --
    from tool_eval_bench.plugins.gsm8k.dataset import _find_cache_file, load_dataset

    dataset_items = load_dataset_with_progress(
        console,
        name="GSM8K",
        noun="questions",
        cache_path=_find_cache_file(),
        load=load_dataset,
        cache_note="data/gsm8k/test.jsonl",
    )
    if dataset_items is None:
        return

    # -- Phase 2: Evaluate with model --
    async def run() -> None:
        eval_total = limit if limit > 0 else len(dataset_items)

        with PluginProgressDisplay(console, total=eval_total) as display:

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                display.tally.record(item_info)
                got = item_info.get("extracted_answer", "?")
                expected = item_info.get("ground_truth", "?")
                display.advance(
                    current,
                    total,
                    stats=tally_line(display.tally, rate=display.rate_per_minute(current)),
                    detail=(
                        f"  {status_icon(item_info)} [bold]{got}[/]/{expected} "
                        f"[dim italic]{truncate(item_info.get('question'))}[/]"
                    ),
                )

            try:
                result = await plugin.run(
                    adapter,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout,
                    seed=seed,
                    extra_params=extra_params,
                    on_progress=on_progress,
                    n_shots=n_shots,
                    limit=limit,
                    shuffle=shuffle,
                    concurrency=args.parallel,
                    _preloaded_items=dataset_items,
                )
                result_holder.append(result)

                # Final state
                total_items = result.details["total"]
                final_speed = (
                    total_items / result.duration_seconds * 60 if result.duration_seconds > 0 else 0
                )
                errs = result.details.get("errors", 0)
                display.finish(
                    completed=total_items,
                    stats=final_tally_line(
                        correct=result.details["correct"],
                        wrong=total_items - result.details["correct"] - errs,
                        errors=errs,
                        score=result.score,
                        rate=final_speed,
                    ),
                )
            finally:
                if hasattr(adapter, "aclose"):
                    await adapter.aclose()

    result = _execute_plugin(console, "GSM8K", run, result_holder)
    if result is None:
        return
    details = result.details

    # Display summary
    console.print()
    errs = details.get("errors", 0)
    total = details["total"]
    answered = details.get("answered", total - errs)
    console.print(
        f"  [bold]GSM8K Accuracy:[/] [bold magenta]{result.score:.1f}%[/] "
        f"({details['correct']}/{total})"
    )
    if errs > 0:
        console.print(
            f"  [bold yellow]⚠ {errs} errors[/] (counted in accuracy; {answered}/{total} answered)"
        )
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    report_lines = plugin.render_report_section(result)
    _finalize_plugin_run(
        mode="gsm8k",
        title="GSM8K",
        display_name=display_name,
        result=result,
        config={
            "model": model,
            "base_url": base_url,
            "mode": "gsm8k",
            "n_shots": n_shots,
            "limit": limit,
            "temperature": args.temperature,
            "seed": seed,
            "shuffle": shuffle,
        },
        report_metrics=[
            f"- **Accuracy**: **{result.score:.1f}%**",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=report_lines,
        output_dir=output_dir,
        run_context=run_context,
    )

    console.print("\n  [dim]Report saved to runs/[/]\n")


# ---------------------------------------------------------------------------
# MMLU benchmark (--mmlu / --mmlu-only)
# ---------------------------------------------------------------------------


def _run_mmlu_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: Any | None = None,
) -> None:
    """Run the MMLU benchmark and display results."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.plugins.mmlu.plugin import MMLUPlugin

    n_shots = args.mmlu_shots
    limit = args.mmlu_limit
    subjects_str = args.mmlu_subjects
    seed = getattr(args, "seed", None)
    limit_label = "all 14042" if limit == 0 else f"{limit}"
    subjects_list = [s.strip() for s in subjects_str.split(",")] if subjects_str else None
    subjects_label = f" · subjects: {subjects_str}" if subjects_str else ""

    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]{n_shots}-shot · {limit_label} questions{subjects_label}{parallel_label}[/]",
            title="[bold]🧠 MMLU — Massive Multitask Language Understanding[/]",
            border_style="bright_blue",
        )
    )

    plugin = MMLUPlugin()
    adapter = build_adapter(base_url, wire_format=getattr(args, "format", None))
    result_holder: list = []

    # -- Phase 1: Load dataset (with visible progress) --
    from tool_eval_bench.plugins.mmlu.dataset import _find_cache_file, load_dataset

    test_items = load_dataset_with_progress(
        console,
        name="MMLU",
        noun="questions",
        cache_path=_find_cache_file("test"),
        load=lambda **kw: load_dataset("test", **kw),
        cache_note="data/mmlu/test.jsonl",
        partial_path=Path("data") / "mmlu" / "test.partial.jsonl",
    )
    if test_items is None:
        return

    # Load dev split for few-shot
    dev_items = []
    if n_shots > 0:
        dev_cache = _find_cache_file("dev")
        if dev_cache.exists():
            dev_items = load_dataset("dev")
        else:
            with console.status("[dim]Downloading MMLU dev split…[/]", spinner="dots"):
                dev_items = load_dataset("dev")
            console.print(f"  [dim]Loaded {len(dev_items)} dev examples for few-shot[/]")

    preloaded = {"test": test_items, "dev": dev_items}

    # -- Phase 2: Evaluate with model --
    async def run() -> None:

        eval_total = limit if limit > 0 else len(test_items)
        if subjects_list:
            # Adjust for filtering
            from tool_eval_bench.plugins.mmlu.dataset import CATEGORIES, SUBJECT_CATEGORIES

            expanded: set[str] = set()
            for s in subjects_list:
                if s in CATEGORIES:
                    expanded.update(subj for subj, cat in SUBJECT_CATEGORIES.items() if cat == s)
                else:
                    expanded.add(s)
            filtered = [it for it in test_items if it.subject in expanded]
            eval_total = min(eval_total, len(filtered)) if limit > 0 else len(filtered)

        with PluginProgressDisplay(console, total=eval_total) as display:

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                display.tally.record(item_info)
                got = item_info.get("extracted_answer", "?")
                expected = item_info.get("ground_truth", "?")
                subj = item_info.get("subject", "?")
                display.advance(
                    current,
                    total,
                    stats=tally_line(
                        display.tally, rate=display.rate_per_minute(current), accent="blue"
                    ),
                    detail=(
                        f"  {status_icon(item_info)} [bold]{got}[/]/{expected} "
                        f"[dim]{subj}[/]  [dim italic]{truncate(item_info.get('question'))}[/]"
                    ),
                )

            try:
                result = await plugin.run(
                    adapter,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout,
                    seed=seed,
                    extra_params=extra_params,
                    on_progress=on_progress,
                    n_shots=n_shots,
                    limit=limit,
                    subjects=subjects_list,
                    concurrency=args.parallel,
                    _preloaded_items=preloaded,
                )
                result_holder.append(result)

                total_items = result.details["total"]
                final_speed = (
                    total_items / result.duration_seconds * 60 if result.duration_seconds > 0 else 0
                )
                errs = result.details.get("errors", 0)
                display.finish(
                    completed=total_items,
                    stats=final_tally_line(
                        correct=result.details["correct"],
                        wrong=total_items - result.details["correct"] - errs,
                        errors=errs,
                        score=result.score,
                        rate=final_speed,
                        accent="blue",
                    ),
                )
            finally:
                if hasattr(adapter, "aclose"):
                    await adapter.aclose()

    result = _execute_plugin(console, "MMLU", run, result_holder)
    if result is None:
        return
    details = result.details

    console.print()
    errs = details.get("errors", 0)
    total = details["total"]
    answered = details.get("answered", total - errs)
    console.print(
        f"  [bold]MMLU Accuracy:[/] [bold blue]{result.score:.1f}%[/] "
        f"({details['correct']}/{total})"
    )
    if errs > 0:
        console.print(
            f"  [bold yellow]⚠ {errs} errors[/] (counted in accuracy; {answered}/{total} answered)"
        )
    console.print(f"  [bold]Rating:[/] {result.rating}")
    # Show category breakdown
    cats = details.get("categories", {})
    if cats:
        parts = [f"{cat}: {c['accuracy']:.1f}%" for cat, c in sorted(cats.items())]
        console.print(f"  [dim]{' · '.join(parts)}[/]")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    report_lines = plugin.render_report_section(result)
    _finalize_plugin_run(
        mode="mmlu",
        title="MMLU",
        display_name=display_name,
        result=result,
        config={
            "model": model,
            "base_url": base_url,
            "mode": "mmlu",
            "n_shots": n_shots,
            "limit": limit,
            "temperature": args.temperature,
            "seed": seed,
            "subjects": subjects_str,
        },
        report_metrics=[
            f"- **Accuracy**: **{result.score:.1f}%**",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=report_lines,
        output_dir=output_dir,
        run_context=run_context,
    )

    console.print("\n  [dim]Report saved to runs/[/]\n")


# ---------------------------------------------------------------------------
# IFEval benchmark (--ifeval / --ifeval-only)
# ---------------------------------------------------------------------------


def _run_ifeval_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: Any | None = None,
) -> None:
    """Run the IFEval instruction-following benchmark and display results."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.plugins.ifeval.plugin import IFEvalPlugin

    limit = args.ifeval_limit
    seed = getattr(args, "seed", None)
    limit_label = "all 541" if limit == 0 else f"{limit}"

    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n[dim]{limit_label} prompts · 25 constraint types{parallel_label}[/]",
            title="[bold]📋 IFEval — Instruction Following Evaluation[/]",
            border_style="bright_green",
        )
    )

    plugin = IFEvalPlugin()
    adapter = build_adapter(base_url, wire_format=getattr(args, "format", None))
    result_holder: list = []

    # -- Phase 1: Load dataset --
    from tool_eval_bench.plugins.ifeval.dataset import _find_cache_file, load_dataset

    dataset_items = load_dataset_with_progress(
        console,
        name="IFEval",
        noun="prompts",
        cache_path=_find_cache_file(),
        load=load_dataset,
        cache_note="data/ifeval/prompts.jsonl",
        partial_path=Path("data") / "ifeval" / "prompts.partial.jsonl",
    )
    if dataset_items is None:
        return

    # -- Phase 2: Evaluate with model --
    async def run() -> None:

        eval_total = limit if limit > 0 else len(dataset_items)

        instructions_passed = 0
        instructions_total = 0

        with PluginProgressDisplay(console, total=eval_total) as display:

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                # IFEval scores whole prompts, and also reports how many
                # individual constraints within each prompt were satisfied, so
                # its tally carries two percentages rather than one accuracy.
                nonlocal instructions_passed, instructions_total
                display.tally.record({**item_info, "correct": item_info.get("prompt_pass")})
                instructions_passed += item_info.get("instructions_passed", 0)
                instructions_total += item_info.get("instructions_total", 0)

                inst_pct = (
                    (instructions_passed / instructions_total * 100)
                    if instructions_total > 0
                    else 0
                )
                status_parts = [
                    f"  [bold green]✓ {display.tally.correct}[/]",
                    f"[bold red]✗ {display.tally.wrong}[/]",
                ]
                if display.tally.errors > 0:
                    status_parts.append(f"[bold yellow]⚠ {display.tally.errors}[/]")
                status_parts += [
                    "[dim]│[/]",
                    f"[bold green]{display.tally.accuracy:.1f}%[/] prompt",
                    f"[bold cyan]{inst_pct:.1f}%[/] instr",
                    "[dim]│[/]",
                    f"[dim]{display.rate_per_minute(current):.1f} p/min[/]",
                ]
                ip = item_info.get("instructions_passed", 0)
                it = item_info.get("instructions_total", 0)
                display.advance(
                    current,
                    total,
                    stats="  ".join(status_parts),
                    detail=(
                        f"  {status_icon(item_info, key='prompt_pass')} [bold]{ip}[/]/{it} "
                        f"constraints  [dim italic]{truncate(item_info.get('prompt'))}[/]"
                    ),
                )

            try:
                result = await plugin.run(
                    adapter,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout,
                    seed=seed,
                    extra_params=extra_params,
                    on_progress=on_progress,
                    limit=limit,
                    concurrency=args.parallel,
                    _preloaded_items=dataset_items,
                )
                result_holder.append(result)

                d = result.details
                final_speed = (
                    d["total"] / result.duration_seconds * 60 if result.duration_seconds > 0 else 0
                )
                errs = d.get("errors", 0)
                wrong = d["total"] - d["prompts_passed"] - errs
                parts = f"  [bold green]✓ {d['prompts_passed']}[/]  [bold red]✗ {wrong}[/]  "
                if errs > 0:
                    parts += f"[bold yellow]⚠ {errs} errors[/]  "
                parts += (
                    f"[dim]│[/]  "
                    f"[bold green]{d['prompt_accuracy']:.1f}%[/] prompt  "
                    f"[bold cyan]{d.get('instruction_accuracy', 0):.1f}%[/] instr  "
                    f"[dim]│[/]  "
                    f"[dim]{final_speed:.1f} p/min[/]"
                )
                display.finish(completed=d["total"], stats=parts)
            finally:
                if hasattr(adapter, "aclose"):
                    await adapter.aclose()

    result = _execute_plugin(console, "IFEval", run, result_holder)
    if result is None:
        return
    details = result.details

    console.print()
    errs = details.get("errors", 0)
    total = details["total"]
    answered = details.get("answered", total - errs)
    console.print(
        f"  [bold]IFEval Prompt Accuracy:[/] [bold green]{details.get('prompt_accuracy', 0):.1f}%[/] "
        f"({details['prompts_passed']}/{total})"
    )
    if errs > 0:
        console.print(
            f"  [bold yellow]⚠ {errs} errors[/] (counted in accuracy; {answered}/{total} answered)"
        )
    console.print(
        f"  [bold]IFEval Instruction Accuracy:[/] [bold cyan]"
        f"{details.get('instruction_accuracy', 0):.1f}%[/] "
        f"({details.get('instructions_passed', 0)}/{details.get('instructions_total', 0)})"
    )
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    report_lines = plugin.render_report_section(result)
    _finalize_plugin_run(
        mode="ifeval",
        title="IFEval",
        display_name=display_name,
        result=result,
        config={
            "model": model,
            "base_url": base_url,
            "mode": "ifeval",
            "limit": limit,
            "temperature": args.temperature,
            "seed": seed,
        },
        report_metrics=[
            f"- **Prompt Accuracy**: **{details.get('prompt_accuracy', 0):.1f}%**",
            f"- **Instruction Accuracy**: **{details.get('instruction_accuracy', 0):.1f}%**",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=report_lines,
        output_dir=output_dir,
        run_context=run_context,
    )

    console.print("\n  [dim]Report saved to runs/[/]\n")


# ---------------------------------------------------------------------------
# Needle in a haystack (--needle / --needle-only)
# ---------------------------------------------------------------------------

# The prompt scaffolding, the question, and the answer budget all live inside
# the context window alongside the haystack.  Sizing haystacks against the raw
# window would push the largest cells past it and score a context overflow as a
# retrieval failure.
_NEEDLE_PROMPT_OVERHEAD = 2048

# The shallowest haystack worth probing.  Below this a miss says nothing about
# long-context retrieval.
_NEEDLE_MIN_TOKENS = 1024


def _needle_context_lengths(context_size: int, steps: int) -> list[int]:
    """Evenly spaced haystack sizes from ``_NEEDLE_MIN_TOKENS`` to the window."""
    usable = context_size - _NEEDLE_PROMPT_OVERHEAD
    if usable <= _NEEDLE_MIN_TOKENS:
        return [max(1, usable)]
    if steps < 2:
        return [usable]
    span = usable - _NEEDLE_MIN_TOKENS
    return [int(_NEEDLE_MIN_TOKENS + span * i / (steps - 1)) for i in range(steps)]


def _needle_depths(steps: int) -> list[float]:
    """Evenly spaced depths across the haystack, inclusive of both ends."""
    if steps < 2:
        return [0.5]
    return [round(i / (steps - 1), 4) for i in range(steps)]


def _resolve_needle_context_size(
    console: Console,
    base_url: str,
    model: str,
    api_key: str | None,
    args: argparse.Namespace,
) -> int | None:
    """Return the effective context window, or ``None`` when it cannot be found."""
    import asyncio

    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.context_pressure import detect_context_size, detect_kv_capacity

    if args.context_size:
        return int(args.context_size)

    context_size = asyncio.run(
        detect_context_size(base_url, model, api_key, client_factory=HTTPMeasurementClient)
    )
    if context_size is None:
        console.print(
            "\n[bold red]Error:[/] Could not auto-detect the context window. "
            "Use --context-size to specify it."
        )
        return None

    # Same cap the pressure sweep applies: a server may have allocated far less
    # KV cache than the model architecture allows, and a haystack it cannot hold
    # measures the deployment rather than the model.
    kv_info = asyncio.run(
        detect_kv_capacity(
            base_url,
            api_key,
            metrics_url=getattr(args, "metrics_url", None),
            client_factory=HTTPMeasurementClient,
        )
    )
    if kv_info is not None and not kv_info.is_hybrid and kv_info.capacity < context_size:
        console.print(
            f"  [dim]⚠ KV cache capacity ({kv_info.capacity:,} tokens) < "
            f"max_model_len ({context_size:,}) — capping[/]"
        )
        context_size = kv_info.capacity
    return context_size


def _run_needle_benchmark(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    extra_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_context: Any | None = None,
) -> None:
    """Run needle-in-a-haystack retrieval and display the grid."""
    from rich.panel import Panel

    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.plugins.needle.haystack import build_cases
    from tool_eval_bench.plugins.needle.plugin import NeedlePlugin

    seed = getattr(args, "seed", None)

    context_size = _resolve_needle_context_size(console, base_url, model, api_key, args)
    if context_size is None:
        return

    lengths = _needle_context_lengths(context_size, max(1, args.needle_lengths))
    depths = _needle_depths(max(1, args.needle_depths))
    cases = build_cases(lengths, depths, seed=seed)

    parallel = args.parallel
    parallel_label = f" · parallel {parallel}" if parallel > 1 else ""

    console.print()
    console.print(
        Panel(
            f"[bold]{display_name}[/]\n"
            f"[dim]{len(lengths)} haystack sizes × {len(depths)} depths = "
            f"{len(cases)} needles · up to {lengths[-1]:,} tokens"
            f"{parallel_label}[/]",
            title="[bold]🪡 Needle in a Haystack — Retrieval[/]",
            border_style="bright_yellow",
        )
    )

    plugin = NeedlePlugin()
    adapter = build_adapter(base_url, wire_format=getattr(args, "format", None))
    result_holder: list = []

    # A 100K-token prompt takes far longer to prefill than a scenario turn, so
    # the per-request timeout scales with the largest haystack in the grid.
    effective_timeout = max(args.timeout, 120.0 + lengths[-1] / 50_000 * 60.0)

    async def run() -> None:
        with PluginProgressDisplay(console, total=len(cases)) as display:

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                display.tally.record({**item_info, "correct": item_info.get("found")})
                display.advance(
                    current,
                    total,
                    stats=tally_line(
                        display.tally,
                        rate=display.rate_per_minute(current),
                        unit="n/min",
                        accent="yellow",
                    ),
                    detail=(
                        f"  {status_icon(item_info, key='found')} "
                        f"[bold]{item_info.get('cell_id')}[/]  "
                        f"[dim italic]{truncate(item_info.get('model_response'))}[/]"
                    ),
                )

            try:
                result = await plugin.run(
                    adapter,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=effective_timeout,
                    seed=seed,
                    extra_params=extra_params,
                    on_progress=on_progress,
                    cases=cases,
                    concurrency=parallel,
                    context_size=context_size,
                )
                result_holder.append(result)
                d = result.details
                display.finish(
                    completed=d["total"],
                    stats=final_tally_line(
                        correct=d["retrieved"],
                        wrong=d["total"] - d["retrieved"] - d.get("errors", 0),
                        errors=d.get("errors", 0),
                        score=d["accuracy"],
                        rate=(
                            d["total"] / result.duration_seconds * 60
                            if result.duration_seconds > 0
                            else 0
                        ),
                        unit="n/min",
                        accent="yellow",
                    ),
                )
            finally:
                if hasattr(adapter, "aclose"):
                    await adapter.aclose()

    result = _execute_plugin(console, "Needle", run, result_holder)
    if result is None:
        return
    details = result.details

    console.print()
    _print_needle_grid(console, result)
    console.print()
    console.print(
        f"  [bold]Retrieval Accuracy:[/] [bold green]{details['accuracy']:.1f}%[/] "
        f"({details['retrieved']}/{details['total']})"
    )
    effective = details.get("effective_context")
    if effective:
        console.print(
            f"  [bold]Effective context:[/] [bold cyan]{effective:,}[/] tokens "
            f"[dim](largest haystack retrieved at every depth)[/]"
        )
    else:
        console.print(
            "  [bold yellow]Effective context:[/] none — every haystack size missed a needle"
        )
    errs = details.get("errors", 0)
    if errs > 0:
        console.print(f"  [bold yellow]⚠ {errs} errors[/] (counted as misses)")
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    _finalize_plugin_run(
        mode="needle",
        title="Needle in a Haystack",
        display_name=display_name,
        result=result,
        config={
            "model": model,
            "base_url": base_url,
            "mode": "needle",
            "context_size": context_size,
            "lengths": lengths,
            "depths": depths,
            "temperature": args.temperature,
            "seed": seed,
        },
        report_metrics=[
            f"- **Retrieval Accuracy**: **{details['accuracy']:.1f}%**",
            f"- **Effective Context**: {f'{effective:,} tokens' if effective else 'none'}",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=plugin.render_report_section(result),
        output_dir=output_dir,
        run_context=run_context,
    )

    console.print("\n  [dim]Report saved to runs/[/]\n")


def _print_needle_grid(console: Console, result: Any) -> None:
    """Print the retrieval grid as depth rows against haystack-size columns."""
    from rich.table import Table

    details = result.details
    lengths: list[int] = details.get("context_lengths", [])
    depths: list[float] = details.get("depths", [])
    if not lengths or not depths:
        return

    found = {(r["context_tokens"], r["depth_percent"]): r["found"] for r in result.item_results}

    table = Table(title="Retrieval grid", title_style="bold", border_style="bright_yellow")
    table.add_column("Depth", justify="right", style="dim")
    for length in lengths:
        table.add_column(f"{length // 1024}K", justify="center")
    for depth in depths:
        cells = ["[green]●[/]" if found.get((length, depth)) else "[red]○[/]" for length in lengths]
        table.add_row(f"{depth:.0%}", *cells)
    console.print(table)


PluginRunner = Callable[..., None]


def run_selected_plugins(
    console: Console,
    model: str,
    display_name: str,
    base_url: str,
    api_key: str | None,
    args: argparse.Namespace,
    *,
    runners: Mapping[str, PluginRunner],
    extra_params: dict[str, Any] | None,
    output_dir: str | None,
    run_context: Any | None,
) -> bool:
    """Run requested plugins in stable order and report an ``only`` stop.

    Plugin implementations retain their benchmark-specific loading, scoring,
    and rendering.  This function owns the common selection/invocation/stop
    lifecycle so combined and plugin-only modes cannot drift apart.
    """
    for name in ("gsm8k", "mmlu", "ifeval", "needle"):
        selected = getattr(args, name) or getattr(args, f"{name}_only")
        if not selected:
            continue
        runners[name](
            console,
            model,
            display_name,
            base_url,
            api_key,
            args,
            extra_params=extra_params,
            output_dir=output_dir,
            run_context=run_context,
        )
        if getattr(args, f"{name}_only"):
            return True
    return False
