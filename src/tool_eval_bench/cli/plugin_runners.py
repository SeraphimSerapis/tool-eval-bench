"""Shared lifecycle dispatch and compatibility facade for external benchmark plugins."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from typing import Any

from rich.console import Console

from tool_eval_bench.cli.command_registry import PLUGIN_FLAG_STEMS
from tool_eval_bench.cli.plugin_accuracy import (
    dataset_identity as _dataset_identity,  # noqa: F401
)
from tool_eval_bench.cli.plugin_accuracy import (
    print_truncated as _print_truncated,  # noqa: F401
)
from tool_eval_bench.cli.plugin_accuracy import (
    run_gsm8k_benchmark as _run_gsm8k_impl,
)
from tool_eval_bench.cli.plugin_accuracy import (
    run_ifeval_benchmark as _run_ifeval_impl,
)
from tool_eval_bench.cli.plugin_accuracy import (
    run_mmlu_benchmark as _run_mmlu_impl,
)
from tool_eval_bench.cli.plugin_datasets import load_dataset_with_progress
from tool_eval_bench.cli.plugin_decision import (
    print_decision_tables as _print_decision_tables,  # noqa: F401
)
from tool_eval_bench.cli.plugin_decision import (
    run_decision_benchmark as _run_decision_benchmark,
)
from tool_eval_bench.cli.plugin_lifecycle import (
    execute_plugin as _execute_plugin,  # noqa: F401
)
from tool_eval_bench.cli.plugin_lifecycle import (
    finalize_plugin_run as _finalize_plugin_run,  # noqa: F401
)
from tool_eval_bench.cli.plugin_needle import (
    _NEEDLE_MIN_TOKENS,  # noqa: F401
    _NEEDLE_PROMPT_OVERHEAD,  # noqa: F401
)
from tool_eval_bench.cli.plugin_needle import (
    needle_context_lengths as _needle_context_lengths,  # noqa: F401
)
from tool_eval_bench.cli.plugin_needle import (
    needle_depths as _needle_depths,  # noqa: F401
)
from tool_eval_bench.cli.plugin_needle import (
    print_needle_grid as _print_needle_grid,  # noqa: F401
)
from tool_eval_bench.cli.plugin_needle import (
    resolve_needle_context_size as _resolve_needle_context_size,  # noqa: F401
)
from tool_eval_bench.cli.plugin_needle import (
    run_needle_benchmark as _run_needle_benchmark,
)
from tool_eval_bench.cli.plugin_toolscale import (
    _TOKENS_PER_TOOL_ESTIMATE,  # noqa: F401
)
from tool_eval_bench.cli.plugin_toolscale import (
    OPENAI_TOOL_CAP as _OPENAI_TOOL_CAP,
)
from tool_eval_bench.cli.plugin_toolscale import (
    print_toolscale_effects as _print_toolscale_effects,  # noqa: F401
)
from tool_eval_bench.cli.plugin_toolscale import (
    print_toolscale_grid as _print_toolscale_grid,  # noqa: F401
)
from tool_eval_bench.cli.plugin_toolscale import (
    run_toolscale_benchmark as _run_toolscale_impl,
)
from tool_eval_bench.cli.plugin_toolscale import (
    toolscale_timeout as _toolscale_timeout,  # noqa: F401
)
from tool_eval_bench.domain.models import RunContext

PluginRunner = Callable[..., None]


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
    run_context: RunContext | None = None,
) -> None:
    """Run GSM8K, forwarding module-level ``load_dataset_with_progress`` for test seams."""
    _run_gsm8k_impl(
        console,
        model,
        display_name,
        base_url,
        api_key,
        args,
        extra_params=extra_params,
        output_dir=output_dir,
        run_context=run_context,
        load_dataset_fn=load_dataset_with_progress,
    )


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
    run_context: RunContext | None = None,
) -> None:
    """Run MMLU, forwarding module-level ``load_dataset_with_progress`` for test seams."""
    _run_mmlu_impl(
        console,
        model,
        display_name,
        base_url,
        api_key,
        args,
        extra_params=extra_params,
        output_dir=output_dir,
        run_context=run_context,
        load_dataset_fn=load_dataset_with_progress,
    )


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
    run_context: RunContext | None = None,
) -> None:
    """Run IFEval, forwarding module-level ``load_dataset_with_progress`` for test seams."""
    _run_ifeval_impl(
        console,
        model,
        display_name,
        base_url,
        api_key,
        args,
        extra_params=extra_params,
        output_dir=output_dir,
        run_context=run_context,
        load_dataset_fn=load_dataset_with_progress,
    )


def _run_toolscale_benchmark(
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
) -> None:
    """Run Toolscale, forwarding module-level ``_OPENAI_TOOL_CAP`` for test seams."""
    _run_toolscale_impl(
        console,
        model,
        display_name,
        base_url,
        api_key,
        args,
        extra_params=extra_params,
        output_dir=output_dir,
        run_context=run_context,
        openai_tool_cap=_OPENAI_TOOL_CAP,
    )


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
    run_context: RunContext | None,
) -> bool:
    """Run requested plugins in stable order and report an ``only`` stop.

    Plugin implementations retain their benchmark-specific loading, scoring,
    and rendering.  This function owns the common selection/invocation/stop
    lifecycle so combined and plugin-only modes cannot drift apart.
    """
    for name, stem in PLUGIN_FLAG_STEMS.items():
        selected = getattr(args, stem) or getattr(args, f"{stem}_only")
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
        if getattr(args, f"{stem}_only"):
            return True
    return False


__all__ = [
    "PluginRunner",
    "_OPENAI_TOOL_CAP",
    "_finalize_plugin_run",
    "_needle_context_lengths",
    "_needle_depths",
    "_print_toolscale_effects",
    "_resolve_needle_context_size",
    "_run_decision_benchmark",
    "_run_gsm8k_benchmark",
    "_run_ifeval_benchmark",
    "_run_mmlu_benchmark",
    "_run_needle_benchmark",
    "_run_toolscale_benchmark",
    "_toolscale_timeout",
    "load_dataset_with_progress",
    "run_selected_plugins",
]
