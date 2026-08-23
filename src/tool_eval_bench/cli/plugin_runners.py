"""Shared lifecycle dispatch for external benchmark plugins."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from typing import Any

from rich.console import Console

from tool_eval_bench.cli.helpers import metadata_for_storage as _metadata_for_storage
from tool_eval_bench.cli.helpers import persist_plugin_run as _persist_plugin_run
from tool_eval_bench.cli.plugin_lifecycle import (
    execute_plugin as _execute_plugin_impl,
)
from tool_eval_bench.cli.plugin_lifecycle import (
    finalize_plugin_run as _finalize_plugin_run_impl,
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

    cache_path = _find_cache_file()
    if cache_path.exists():
        console.print("  [dim]Loading GSM8K from cache…[/]", end=" ")
        dataset_items = load_dataset()
        console.print(f"[bold green]✓[/] [dim]{len(dataset_items)} questions[/]")
    else:
        # First use — download with visible progress
        try:
            import datasets as _ds  # noqa: F401

            method_hint = "via datasets lib"
        except ImportError:
            method_hint = "via REST API"
        console.print()
        with console.status(
            f"[bold]Downloading GSM8K dataset from HuggingFace…[/] [dim]({method_hint})[/]",
            spinner="dots",
        ) as status:

            def on_download(downloaded: int, total: int) -> None:
                pct = downloaded / total * 100 if total else 0
                status.update(
                    f"[bold]Downloading GSM8K dataset…[/] "
                    f"[dim]{downloaded:,}/{total:,} questions ({pct:.0f}%)[/]"
                )

            try:
                dataset_items = load_dataset(on_progress=on_download)
            except Exception as exc:
                console.print(
                    f"\n  [bold red]✗[/] Failed to download GSM8K dataset: {exc}\n"
                    "  [dim]This is usually caused by HuggingFace rate limiting.\n"
                    "  Tip: pip install tool-eval-bench[hf] for rate-limit-free downloads.[/]"
                )
                return

        console.print(
            f"  [bold green]✓[/] Downloaded [bold]{len(dataset_items)}[/] questions "
            f"[dim](cached to data/gsm8k/test.jsonl)[/]"
        )

    # -- Phase 2: Evaluate with model --
    async def run() -> None:
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        eval_total = limit if limit > 0 else len(dataset_items)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%[/]"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            console=console,
        )

        stats_text = TextColumn("")
        stats_progress = Progress(stats_text, console=console)
        last_q_text = TextColumn("")
        last_q_progress = Progress(last_q_text, console=console)

        from rich.console import Group

        group = Group(progress, stats_progress, last_q_progress)

        correct_so_far = 0
        wrong_so_far = 0
        errors_so_far = 0
        t_start = time.monotonic()
        stats_progress.add_task("", total=None)
        last_q_progress.add_task("", total=None)

        with Live(group, console=console, refresh_per_second=4):
            task = progress.add_task("Evaluating…", total=eval_total)

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                nonlocal correct_so_far, wrong_so_far, errors_so_far
                if item_info.get("is_error"):
                    errors_so_far += 1
                elif item_info.get("correct"):
                    correct_so_far += 1
                else:
                    wrong_so_far += 1

                processed = correct_so_far + wrong_so_far + errors_so_far
                pct = (correct_so_far / processed * 100) if processed > 0 else 0
                elapsed = time.monotonic() - t_start
                speed = current / elapsed * 60 if elapsed > 0 else 0  # questions/min

                # Build a compact status line
                status_parts = [
                    f"  [bold green]✓ {correct_so_far}[/]",
                    f"[bold red]✗ {wrong_so_far}[/]",
                ]
                if errors_so_far > 0:
                    status_parts.append(f"[bold yellow]⚠ {errors_so_far}[/]")
                status_parts += [
                    "[dim]│[/]",
                    f"[bold magenta]{pct:.1f}%[/] accuracy",
                    "[dim]│[/]",
                    f"[dim]{speed:.1f} q/min[/]",
                ]
                stats_text.text_format = "  ".join(status_parts)

                progress.update(task, completed=current, total=total)

                # Show last completed question
                if item_info.get("is_error"):
                    icon = "[yellow]⚠[/]"
                elif item_info.get("correct", False):
                    icon = "[green]✓[/]"
                else:
                    icon = "[red]✗[/]"
                got = item_info.get("extracted_answer", "?")
                expected = item_info.get("ground_truth", "?")
                question = (item_info.get("question") or "").replace("\n", " ").strip()
                if len(question) > 90:
                    question = question[:87] + "…"
                last_q_text.text_format = (
                    f"  {icon} [bold]{got}[/]/{expected} [dim italic]{question}[/]"
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
                progress.update(
                    task, completed=result.details["total"], description="[green]✓ Complete"
                )
                final_speed = (
                    result.details["total"] / result.duration_seconds * 60
                    if result.duration_seconds > 0
                    else 0
                )
                errs = result.details.get("errors", 0)
                wrong = result.details["total"] - result.details["correct"] - errs
                parts = f"  [bold green]✓ {result.details['correct']}[/]  [bold red]✗ {wrong}[/]  "
                if errs > 0:
                    parts += f"[bold yellow]⚠ {errs} errors[/]  "
                parts += (
                    f"[dim]│[/]  "
                    f"[bold magenta]{result.score:.1f}%[/] accuracy  "
                    f"[dim]│[/]  "
                    f"[dim]{final_speed:.1f} q/min[/]"
                )
                stats_text.text_format = parts
                last_q_text.text_format = ""
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

    cache_path = _find_cache_file("test")
    if cache_path.exists():
        console.print("  [dim]Loading MMLU from cache…[/]", end=" ")
        test_items = load_dataset("test")
        console.print(f"[bold green]✓[/] [dim]{len(test_items)} questions[/]")
    else:
        from pathlib import Path as _Path

        partial_path = _Path("data") / "mmlu" / "test.partial.jsonl"
        resuming = partial_path.exists()
        # Check which download method will be used
        try:
            import datasets as _ds  # noqa: F401

            method_hint = "via datasets lib"
        except ImportError:
            method_hint = "via REST API"
        label = "Resuming MMLU download" if resuming else "Downloading MMLU dataset"
        console.print()
        with console.status(
            f"[bold]{label} from HuggingFace…[/] [dim]({method_hint})[/]",
            spinner="dots",
        ) as status:

            def on_download(downloaded: int, total: int) -> None:
                pct = downloaded / total * 100 if total else 0
                status.update(
                    f"[bold]{label}…[/] [dim]{downloaded:,}/{total:,} questions ({pct:.0f}%)[/]"
                )

            try:
                test_items = load_dataset("test", on_progress=on_download)
            except Exception as exc:
                console.print(
                    f"\n  [bold red]✗[/] Failed to download MMLU dataset: {exc}\n"
                    "  [dim]This is usually caused by HuggingFace rate limiting.\n"
                    "  Progress is saved — re-run to resume from where it stopped.\n"
                    "  Tip: pip install tool-eval-bench[hf] for rate-limit-free downloads.[/]"
                )
                return
        console.print(
            f"  [bold green]✓[/] Downloaded [bold]{len(test_items)}[/] questions "
            f"[dim](cached to data/mmlu/test.jsonl)[/]"
        )

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
        from rich.console import Group
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

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

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%[/]"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            console=console,
        )

        stats_text = TextColumn("")
        stats_progress = Progress(stats_text, console=console)
        last_q_text = TextColumn("")
        last_q_progress = Progress(last_q_text, console=console)
        group = Group(progress, stats_progress, last_q_progress)

        correct_so_far = 0
        wrong_so_far = 0
        errors_so_far = 0
        t_start = time.monotonic()
        stats_progress.add_task("", total=None)
        last_q_progress.add_task("", total=None)

        with Live(group, console=console, refresh_per_second=4):
            task = progress.add_task("Evaluating…", total=eval_total)

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                nonlocal correct_so_far, wrong_so_far, errors_so_far
                if item_info.get("is_error"):
                    errors_so_far += 1
                elif item_info.get("correct"):
                    correct_so_far += 1
                else:
                    wrong_so_far += 1

                processed = correct_so_far + wrong_so_far + errors_so_far
                pct = (correct_so_far / processed * 100) if processed > 0 else 0
                elapsed = time.monotonic() - t_start
                speed = current / elapsed * 60 if elapsed > 0 else 0

                status_parts = [
                    f"  [bold green]✓ {correct_so_far}[/]",
                    f"[bold red]✗ {wrong_so_far}[/]",
                ]
                if errors_so_far > 0:
                    status_parts.append(f"[bold yellow]⚠ {errors_so_far}[/]")
                status_parts += [
                    "[dim]│[/]",
                    f"[bold blue]{pct:.1f}%[/] accuracy",
                    "[dim]│[/]",
                    f"[dim]{speed:.1f} q/min[/]",
                ]
                stats_text.text_format = "  ".join(status_parts)
                progress.update(task, completed=current, total=total)

                # Show last completed question details
                subj = item_info.get("subject", "?")
                if item_info.get("is_error"):
                    icon = "[yellow]⚠[/]"
                elif item_info.get("correct", False):
                    icon = "[green]✓[/]"
                else:
                    icon = "[red]✗[/]"
                got = item_info.get("extracted_answer", "?")
                expected = item_info.get("ground_truth", "?")
                question = (item_info.get("question") or "").replace("\n", " ").strip()
                if len(question) > 90:
                    question = question[:87] + "…"
                last_q_text.text_format = (
                    f"  {icon} [bold]{got}[/]/{expected} [dim]{subj}[/]  [dim italic]{question}[/]"
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

                progress.update(
                    task, completed=result.details["total"], description="[green]✓ Complete"
                )
                final_speed = (
                    result.details["total"] / result.duration_seconds * 60
                    if result.duration_seconds > 0
                    else 0
                )
                errs = result.details.get("errors", 0)
                wrong = result.details["total"] - result.details["correct"] - errs
                parts = f"  [bold green]✓ {result.details['correct']}[/]  [bold red]✗ {wrong}[/]  "
                if errs > 0:
                    parts += f"[bold yellow]⚠ {errs} errors[/]  "
                parts += (
                    f"[dim]│[/]  "
                    f"[bold blue]{result.score:.1f}%[/] accuracy  "
                    f"[dim]│[/]  "
                    f"[dim]{final_speed:.1f} q/min[/]"
                )
                stats_text.text_format = parts
                last_q_text.text_format = ""
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

    cache_path = _find_cache_file()
    if cache_path.exists():
        console.print("  [dim]Loading IFEval from cache…[/]", end=" ")
        dataset_items = load_dataset()
        console.print(f"[bold green]✓[/] [dim]{len(dataset_items)} prompts[/]")
    else:
        from pathlib import Path as _Path

        partial_path = _Path("data") / "ifeval" / "prompts.partial.jsonl"
        resuming = partial_path.exists()
        try:
            import datasets as _ds  # noqa: F401

            method_hint = "via datasets lib"
        except ImportError:
            method_hint = "via REST API"
        label = "Resuming IFEval download" if resuming else "Downloading IFEval dataset"
        console.print()
        with console.status(
            f"[bold]{label} from HuggingFace…[/] [dim]({method_hint})[/]",
            spinner="dots",
        ) as status:

            def on_download(downloaded: int, total: int) -> None:
                pct = downloaded / total * 100 if total else 0
                status.update(
                    f"[bold]{label}…[/] [dim]{downloaded:,}/{total:,} prompts ({pct:.0f}%)[/]"
                )

            try:
                dataset_items = load_dataset(on_progress=on_download)
            except Exception as exc:
                console.print(
                    f"\n  [bold red]✗[/] Failed to download IFEval dataset: {exc}\n"
                    "  [dim]This is usually caused by HuggingFace rate limiting.\n"
                    "  Progress is saved — re-run to resume from where it stopped.\n"
                    "  Tip: pip install tool-eval-bench[hf] for rate-limit-free downloads.[/]"
                )
                return
        console.print(
            f"  [bold green]✓[/] Downloaded [bold]{len(dataset_items)}[/] prompts "
            f"[dim](cached to data/ifeval/prompts.jsonl)[/]"
        )

    # -- Phase 2: Evaluate with model --
    async def run() -> None:
        from rich.console import Group
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        eval_total = limit if limit > 0 else len(dataset_items)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[bold]{task.percentage:>3.0f}%[/]"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/]"),
            TimeRemainingColumn(),
            console=console,
        )

        stats_text = TextColumn("")
        stats_progress = Progress(stats_text, console=console)
        last_q_text = TextColumn("")
        last_q_progress = Progress(last_q_text, console=console)
        group = Group(progress, stats_progress, last_q_progress)

        prompts_passed = 0
        prompts_failed = 0
        errors_so_far = 0
        instructions_passed = 0
        instructions_total = 0
        t_start = time.monotonic()
        stats_progress.add_task("", total=None)
        last_q_progress.add_task("", total=None)

        with Live(group, console=console, refresh_per_second=4):
            task = progress.add_task("Evaluating…", total=eval_total)

            async def on_progress(current: int, total: int, item_info: dict) -> None:
                nonlocal prompts_passed, prompts_failed, errors_so_far
                nonlocal instructions_passed, instructions_total
                if item_info.get("is_error"):
                    errors_so_far += 1
                elif item_info.get("prompt_pass"):
                    prompts_passed += 1
                else:
                    prompts_failed += 1
                instructions_passed += item_info.get("instructions_passed", 0)
                instructions_total += item_info.get("instructions_total", 0)

                processed = prompts_passed + prompts_failed + errors_so_far
                prompt_pct = (prompts_passed / processed * 100) if processed > 0 else 0
                inst_pct = (
                    (instructions_passed / instructions_total * 100)
                    if instructions_total > 0
                    else 0
                )
                elapsed = time.monotonic() - t_start
                speed = current / elapsed * 60 if elapsed > 0 else 0

                status_parts = [
                    f"  [bold green]✓ {prompts_passed}[/]",
                    f"[bold red]✗ {prompts_failed}[/]",
                ]
                if errors_so_far > 0:
                    status_parts.append(f"[bold yellow]⚠ {errors_so_far}[/]")
                status_parts += [
                    "[dim]│[/]",
                    f"[bold green]{prompt_pct:.1f}%[/] prompt",
                    f"[bold cyan]{inst_pct:.1f}%[/] instr",
                    "[dim]│[/]",
                    f"[dim]{speed:.1f} p/min[/]",
                ]
                stats_text.text_format = "  ".join(status_parts)
                progress.update(task, completed=current, total=total)

                # Show last completed prompt
                if item_info.get("is_error"):
                    icon = "[yellow]⚠[/]"
                elif item_info.get("prompt_pass", False):
                    icon = "[green]✓[/]"
                else:
                    icon = "[red]✗[/]"
                ip = item_info.get("instructions_passed", 0)
                it = item_info.get("instructions_total", 0)
                prompt = (item_info.get("prompt") or "").replace("\n", " ").strip()
                if len(prompt) > 90:
                    prompt = prompt[:87] + "…"
                last_q_text.text_format = (
                    f"  {icon} [bold]{ip}[/]/{it} constraints  [dim italic]{prompt}[/]"
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

                progress.update(
                    task, completed=result.details["total"], description="[green]✓ Complete"
                )
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
                stats_text.text_format = parts
                last_q_text.text_format = ""
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
    for name in ("gsm8k", "mmlu", "ifeval"):
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
