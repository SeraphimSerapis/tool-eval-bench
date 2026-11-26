"""CLI runners for HuggingFace-backed accuracy benchmarks (GSM8K, MMLU, IFEval)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from tool_eval_bench.cli.headless import report_run_failed
from tool_eval_bench.cli.plugin_datasets import load_dataset_with_progress
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
    truncate,
)
from tool_eval_bench.domain.models import RunContext


def dataset_identity(details: Mapping[str, Any]) -> dict[str, Any]:
    """Config keys that pin which dataset revision and items a run graded."""
    return {
        "dataset_revision": details.get("dataset_revision"),
        "items_sha256": details.get("items_sha256"),
    }


def print_truncated(console: Console, details: Mapping[str, Any]) -> None:
    """Say how many answers the token budget cut off, when any were."""
    truncated = details.get("truncated", 0)
    if truncated:
        console.print(
            f"  [bold yellow]⚠ {truncated} truncated[/] (token budget ran out before an "
            "answer; counted as wrong)"
        )


def run_gsm8k_benchmark(
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
    load_dataset_fn: Callable[..., Any] = load_dataset_with_progress,
) -> None:
    """Run the GSM8K grade-school math benchmark and display results."""
    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.plugins.gsm8k.dataset import _find_cache_file, load_dataset
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
    dataset_items = load_dataset_fn(
        console,
        name="GSM8K",
        noun="questions",
        cache_path=_find_cache_file(),
        load=load_dataset,
        cache_note="data/gsm8k/test.jsonl",
    )
    if dataset_items is None:
        sys.exit(1)

    adapter = build_adapter(base_url, **adapter_options(args))
    eval_total = min(limit, len(dataset_items)) if limit > 0 else len(dataset_items)

    def _format_progress(
        display: PluginProgressDisplay, current: int, item_info: dict[str, Any]
    ) -> tuple[str, str]:
        got = item_info.get("extracted_answer", "?")
        expected = item_info.get("ground_truth", "?")
        return (
            tally_line(display.tally, rate=display.rate_per_minute(current)),
            f"  {status_icon(item_info)} [bold]{got}[/]/{expected} "
            f"[dim italic]{truncate(item_info.get('question'))}[/]",
        )

    def _format_finish(res: Any) -> tuple[int, str]:
        total_items = res.details["total"]
        errs = res.details.get("errors", 0)
        return (
            total_items,
            final_tally_line(
                correct=res.details["correct"],
                wrong=total_items - res.details["correct"] - errs,
                errors=errs,
                score=res.score,
                rate=items_per_minute(total_items, res.duration_seconds),
            ),
        )

    async def _run_plugin(on_progress: ProgressCallback) -> Any:
        return await plugin.run(
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

    result = run_plugin_session(
        console,
        benchmark_name="GSM8K",
        adapter=adapter,
        total=eval_total,
        run_plugin=_run_plugin,
        format_progress=_format_progress,
        format_finish=_format_finish,
    )
    if result is None:
        return
    details = result.details

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
    print_truncated(console, details)
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    finalize_plugin_run(
        console,
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
            # An unseeded shuffle draws its own seed; recording it keeps two
            # such runs from sharing a fingerprint and makes the sample
            # reproducible.
            "shuffle_seed": details.get("shuffle_seed"),
            "extra_params": extra_params or None,
            **dataset_identity(details),
        },
        report_metrics=[
            f"- **Accuracy**: **{result.score:.1f}%**",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=plugin.render_report_section(result),
        output_dir=output_dir,
        label=getattr(args, "label", None),
        run_context=run_context,
    )


def run_mmlu_benchmark(
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
    load_dataset_fn: Callable[..., Any] = load_dataset_with_progress,
) -> None:
    """Run the MMLU benchmark and display results."""
    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.plugins.mmlu.dataset import _find_cache_file, load_dataset
    from tool_eval_bench.plugins.mmlu.plugin import (
        SAMPLING,
        MMLUPlugin,
        resolve_subjects,
        select_items,
    )

    n_shots = args.mmlu_shots
    limit = args.mmlu_limit
    subjects_str = args.mmlu_subjects
    seed = getattr(args, "seed", None)
    limit_label = "all 14042" if limit == 0 else f"{limit} stratified"
    subjects_list: list[str] | None = None
    if subjects_str is not None:
        # Normalised so "STEM" and "stem,STEM" store, and fingerprint, alike.
        subjects_list = sorted({s.strip().lower() for s in subjects_str.split(",") if s.strip()})
        subjects_str = ",".join(subjects_list)
        # A typo or an empty list would otherwise download 14K questions first.
        try:
            if not subjects_list:
                raise ValueError("--mmlu-subjects needs at least one subject or category")
            resolve_subjects(subjects_list)
        except ValueError as exc:
            report_run_failed(console, f"\n[bold red]MMLU error:[/] {escape(str(exc))}")
            sys.exit(1)
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
    test_items = load_dataset_fn(
        console,
        name="MMLU",
        noun="questions",
        cache_path=_find_cache_file("test"),
        load=lambda **kw: load_dataset("test", **kw),
        cache_note="data/mmlu/test.jsonl",
        partial_path=Path("data") / "mmlu" / "test.partial.jsonl",
    )
    if test_items is None:
        sys.exit(1)

    # Load dev split for few-shot.  Same loader as the test split, so a failed
    # download is reported and resumable instead of escaping as a traceback.
    dev_items: Any = []
    if n_shots > 0:
        dev_items = load_dataset_fn(
            console,
            name="MMLU dev split",
            noun="few-shot examples",
            cache_path=_find_cache_file("dev"),
            load=lambda **kw: load_dataset("dev", **kw),
            cache_note="data/mmlu/dev.jsonl",
            partial_path=Path("data") / "mmlu" / "dev.partial.jsonl",
        )
        if dev_items is None:
            sys.exit(1)

    adapter = build_adapter(base_url, **adapter_options(args))
    preloaded = {"test": test_items, "dev": dev_items}
    # The plugin makes the same selection; this only sizes the progress bar.
    eval_total = len(select_items(test_items, subjects_list, limit))

    def _format_progress(
        display: PluginProgressDisplay, current: int, item_info: dict[str, Any]
    ) -> tuple[str, str]:
        got = item_info.get("extracted_answer", "?")
        expected = item_info.get("ground_truth", "?")
        subj = item_info.get("subject", "?")
        return (
            tally_line(display.tally, rate=display.rate_per_minute(current), accent="blue"),
            f"  {status_icon(item_info)} [bold]{got}[/]/{expected} "
            f"[dim]{subj}[/]  [dim italic]{truncate(item_info.get('question'))}[/]",
        )

    def _format_finish(res: Any) -> tuple[int, str]:
        total_items = res.details["total"]
        errs = res.details.get("errors", 0)
        return (
            total_items,
            final_tally_line(
                correct=res.details["correct"],
                wrong=total_items - res.details["correct"] - errs,
                errors=errs,
                score=res.score,
                rate=items_per_minute(total_items, res.duration_seconds),
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

    result = run_plugin_session(
        console,
        benchmark_name="MMLU",
        adapter=adapter,
        total=eval_total,
        run_plugin=_run_plugin,
        format_progress=_format_progress,
        format_finish=_format_finish,
    )
    if result is None:
        return
    details = result.details

    console.print()
    errs = details.get("errors", 0)
    total = details["total"]
    answered = details.get("answered", total - errs)
    console.print(
        f"  [bold]MMLU Accuracy:[/] [bold blue]{result.score:.1f}%[/] ({details['correct']}/{total})"
    )
    if errs > 0:
        console.print(
            f"  [bold yellow]⚠ {errs} errors[/] (counted in accuracy; {answered}/{total} answered)"
        )
    print_truncated(console, details)
    console.print(f"  [bold]Rating:[/] {result.rating}")
    cats = details.get("categories", {})
    if cats:
        parts = [f"{cat}: {c['accuracy']:.1f}%" for cat, c in sorted(cats.items())]
        console.print(f"  [dim]{' · '.join(parts)}[/]")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    finalize_plugin_run(
        console,
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
            "sampling": SAMPLING,
            "temperature": args.temperature,
            "seed": seed,
            "subjects": subjects_str,
            "extra_params": extra_params or None,
            **dataset_identity(details),
        },
        report_metrics=[
            f"- **Accuracy**: **{result.score:.1f}%**",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=plugin.render_report_section(result),
        output_dir=output_dir,
        label=getattr(args, "label", None),
        run_context=run_context,
    )


def run_ifeval_benchmark(
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
    load_dataset_fn: Callable[..., Any] = load_dataset_with_progress,
) -> None:
    """Run the IFEval instruction-following benchmark and display results."""
    from tool_eval_bench.adapters.factory import build_adapter
    from tool_eval_bench.cli.helpers import adapter_options
    from tool_eval_bench.plugins.ifeval.dataset import _find_cache_file, load_dataset
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
    dataset_items = load_dataset_fn(
        console,
        name="IFEval",
        noun="prompts",
        cache_path=_find_cache_file(),
        load=load_dataset,
        cache_note="data/ifeval/prompts.jsonl",
        partial_path=Path("data") / "ifeval" / "prompts.partial.jsonl",
    )
    if dataset_items is None:
        sys.exit(1)

    adapter = build_adapter(base_url, **adapter_options(args))
    eval_total = min(limit, len(dataset_items)) if limit > 0 else len(dataset_items)
    instructions_passed = 0
    instructions_total = 0

    def _record_item(display: PluginProgressDisplay, item_info: dict[str, Any]) -> None:
        nonlocal instructions_passed, instructions_total
        # IFEval scores whole prompts, and also reports how many individual
        # constraints within each prompt were satisfied.
        display.tally.record({**item_info, "correct": item_info.get("prompt_pass")})
        instructions_passed += item_info.get("instructions_passed", 0)
        instructions_total += item_info.get("instructions_total", 0)

    def _format_progress(
        display: PluginProgressDisplay, current: int, item_info: dict[str, Any]
    ) -> tuple[str, str]:
        inst_pct = (
            (instructions_passed / instructions_total * 100) if instructions_total > 0 else 0.0
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
        return (
            "  ".join(status_parts),
            f"  {status_icon(item_info, key='prompt_pass')} [bold]{ip}[/]/{it} "
            f"constraints  [dim italic]{truncate(item_info.get('prompt'))}[/]",
        )

    def _format_finish(res: Any) -> tuple[int, str]:
        d = res.details
        final_speed = items_per_minute(d["total"], res.duration_seconds)
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
        return (d["total"], parts)

    async def _run_plugin(on_progress: ProgressCallback) -> Any:
        return await plugin.run(
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

    result = run_plugin_session(
        console,
        benchmark_name="IFEval",
        adapter=adapter,
        total=eval_total,
        run_plugin=_run_plugin,
        format_progress=_format_progress,
        format_finish=_format_finish,
        record_item=_record_item,
    )
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
    print_truncated(console, details)
    console.print(
        f"  [bold]IFEval Instruction Accuracy:[/] [bold cyan]"
        f"{details.get('instruction_accuracy', 0):.1f}%[/] "
        f"({details.get('instructions_passed', 0)}/{details.get('instructions_total', 0)})"
    )
    console.print(f"  [bold]Rating:[/] {result.rating}")
    console.print(
        f"  [dim]Duration: {result.duration_seconds:.1f}s · Tokens: {result.total_tokens:,}[/]"
    )

    finalize_plugin_run(
        console,
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
            "extra_params": extra_params or None,
            **dataset_identity(details),
        },
        report_metrics=[
            f"- **Prompt Accuracy**: **{details.get('prompt_accuracy', 0):.1f}%**",
            f"- **Instruction Accuracy**: **{details.get('instruction_accuracy', 0):.1f}%**",
            f"- **Completion**: {details.get('completion_rate', 100.0):.1f}%",
        ],
        report_lines=plugin.render_report_section(result),
        output_dir=output_dir,
        label=getattr(args, "label", None),
        run_context=run_context,
    )
