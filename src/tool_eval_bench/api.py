"""Public API for headless/library invocation of tool-eval-bench.

This module provides a clean programmatic interface for external integrators
(e.g. sparkrun) to run benchmarks without going through the CLI.

Usage::

    import asyncio
    from tool_eval_bench.api import run_benchmark

    result = asyncio.run(run_benchmark(
        model="Qwen/Qwen3-8B",
        base_url="http://localhost:8000",
        backend="vllm",
    ))

    print(result["scores"]["final_score"])  # e.g. 87

The returned dict matches the ``--json`` output schema (see OUTPUT_SCHEMA_VERSION).
"""

from __future__ import annotations

import logging
from typing import Any

from tool_eval_bench import __version__
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS
from tool_eval_bench.domain.scenarios import (
    OnScenarioResult,
    OnScenarioStart,
    ScenarioDefinition,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, SCENARIOS
from tool_eval_bench.schema import ARGS_SCHEMA  # noqa: F401 — public re-export
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter

logger = logging.getLogger(__name__)

# Versioned output schema — increment on breaking changes to the JSON shape.
OUTPUT_SCHEMA_VERSION = "1"


def format_result(run_data: dict[str, Any]) -> dict[str, Any]:
    """Wrap raw run_data in a versioned envelope.

    This adds ``schema_version`` and ``tool_eval_bench_version`` fields
    so consumers can detect incompatible changes and adapt their parsers.

    Top-level Spark Arena metadata fields are promoted from the nested
    ``scores`` dict for easy consumption by leaderboard pipelines:

    - ``final_score`` (int 0–100)
    - ``rating`` (star string, e.g. "★★★★ Good")
    - ``safety_warnings`` (list of strings, empty when clean)
    - ``deployability`` (int 0–100 or None)
    - ``responsiveness`` (int 0–100 or None)
    - ``total_scenarios`` (int)
    """
    scores = run_data.get("scores", {})
    envelope: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "tool_eval_bench_version": __version__,
        # Spark Arena top-level fields
        "final_score": scores.get("final_score"),
        "rating": scores.get("rating"),
        "safety_warnings": scores.get("safety_warnings", []),
        "deployability": scores.get("deployability"),
        "responsiveness": scores.get("responsiveness"),
        "total_scenarios": scores.get("max_points", 0) // 2 if scores.get("max_points") else None,
    }
    envelope.update(run_data)
    return envelope


async def run_benchmark(
    *,
    model: str,
    base_url: str,
    backend: str = "vllm",
    api_key: str | None = None,
    scenarios: list[ScenarioDefinition] | None = None,
    short: bool = False,
    temperature: float = 0.0,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_turns: int = 8,
    seed: int | None = None,
    reference_date: str | None = None,
    concurrency: int = 1,
    error_rate: float = 0.0,
    alpha: float = 0.7,
    weight_by_difficulty: bool = False,
    extra_params: dict[str, Any] | None = None,
    on_scenario_start: OnScenarioStart | None = None,
    on_scenario_result: OnScenarioResult | None = None,
    persist: bool = True,
    output_dir: str | None = None,
    wire_format: str | None = None,
) -> dict[str, Any]:
    """Run tool-eval-bench programmatically and return structured results.

    This is the primary entry point for library consumers. It returns the
    same JSON-serializable dict emitted by ``tool-eval-bench --json``,
    wrapped in a versioned envelope (see :func:`format_result`).

    Args:
        model: Model name/path to evaluate.
        base_url: Server base URL (e.g. ``http://localhost:8000``).
        backend: Backend label — ``vllm``, ``litellm``, ``llamacpp``, ``sglang``,
            ``gemini``, or ``ninfer``.
        api_key: Optional API key for authenticated endpoints.
        scenarios: Explicit scenario list.  If *None*, ``short`` controls
            the default set (15 core vs 69 full).
        short: When *True* and ``scenarios`` is *None*, run the core 15.
        temperature: Sampling temperature (default: 0.0 = greedy).
        timeout_seconds: Per-request timeout in seconds.
        max_turns: Maximum conversation turns per scenario.
        seed: Random seed passed to the server.
        reference_date: Override benchmark reference date (``YYYY-MM-DD``).
        concurrency: Number of scenarios to run in parallel.
        error_rate: Inject random tool errors at this rate (0.0–1.0).
        alpha: Quality/speed weight for deployability (0.0–1.0).
        weight_by_difficulty: Also calculate the difficulty-weighted score.
        extra_params: Additional parameters merged into the API payload.
        on_scenario_start: Async callback ``(scenario, idx, total) -> None``.
        on_scenario_result: Async callback ``(scenario, result, idx, total) -> None``.
        persist: When *True* (default), persist results to SQLite + Markdown.
            Set to *False* when the caller handles its own persistence
            (e.g. sparkrun).
        output_dir: Directory for Markdown report files (default: ``./runs/``).
            The SQLite database is always at ``./data/benchmarks.sqlite``.
        wire_format: Request format the endpoint speaks — ``openai``, ``gemini``,
            or ``auto``/*None* to detect it from ``base_url``.

    Returns:
        A versioned JSON-serializable dict containing ``run_id``, ``config``,
        ``scores``, ``metadata``, and optionally ``report_path``.
    """
    # Resolve scenario set
    if scenarios is not None:
        resolved = scenarios
    elif short:
        resolved = list(SCENARIOS)
    else:
        resolved = list(ALL_SCENARIOS)

    # Build service with optional persistence
    if persist:
        repo = RunRepository()
        reporter = MarkdownReporter(root=output_dir)
        service = BenchmarkService(repo=repo, reporter=reporter)
    else:
        service = BenchmarkService(repo=None, reporter=None)

    run_data = await service.run_benchmark(
        model=model,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        scenarios=resolved,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        seed=seed,
        reference_date=reference_date,
        on_scenario_start=on_scenario_start,
        on_scenario_result=on_scenario_result,
        concurrency=concurrency,
        error_rate=error_rate,
        alpha=alpha,
        weight_by_difficulty=weight_by_difficulty,
        extra_params=extra_params,
        wire_format=wire_format,
    )

    return format_result(run_data)
