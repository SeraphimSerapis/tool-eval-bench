"""Benchmark runner service — orchestrates scenario-based tool-call evaluation.

This replaces the old throughput-focused runner with the new multi-turn
scenario benchmark system.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

import httpx

from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.application.finalization import finalize_completed_run
from tool_eval_bench.domain.adapters import BackendAdapter
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS, ChatMessage, RunContext
from tool_eval_bench.domain.scenarios import (
    OnScenarioResult,
    OnScenarioStart,
    ScenarioDefinition,
    ScenarioReportMetadata,
    ScenarioResult,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.orchestrator import run_all_scenarios, score_results
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.utils.ids import build_config_fingerprint, build_run_id
from tool_eval_bench.utils.urls import redact_url as _redact_url

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = {"vllm", "litellm", "llamacpp", "llama.cpp", "llama_cpp"}


class BenchmarkService:
    _SENTINEL = object()

    def __init__(
        self,
        repo: RunRepository | None | object = _SENTINEL,
        reporter: MarkdownReporter | None | object = _SENTINEL,
    ) -> None:
        # Distinguish "not provided" (create defaults) from "explicitly None"
        # (skip persistence).  The previous ``repo or RunRepository()`` pattern
        # silently defeated ``persist=False`` by replacing None with a default.
        self.repo: RunRepository | None = (
            RunRepository() if repo is self._SENTINEL else cast(RunRepository | None, repo)
        )
        self.reporter: MarkdownReporter | None = (
            MarkdownReporter()
            if reporter is self._SENTINEL
            else cast(MarkdownReporter | None, reporter)
        )

    def _adapter_for(self, backend: str) -> BackendAdapter:
        backend_l = backend.lower()
        if backend_l not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}. Supported: vllm, litellm, llamacpp")
        return OpenAICompatibleAdapter()

    async def run_benchmark(
        self,
        *,
        model: str,
        backend: str,
        base_url: str,
        api_key: str | None = None,
        scenario_ids: list[str] | None = None,
        scenarios: list[ScenarioDefinition] | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_turns: int = 8,
        seed: int | None = None,
        reference_date: str | None = None,
        on_scenario_start: OnScenarioStart | None = None,
        on_scenario_result: OnScenarioResult | None = None,
        throughput_samples: list[Any] | None = None,
        concurrency: int = 1,
        error_rate: float = 0.0,
        alpha: float = 0.7,
        extra_params: dict[str, Any] | None = None,
        context_pressure_messages: list[ChatMessage] | None = None,
        context_pressure_config: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
        weight_by_difficulty: bool = False,
        resume_run_id: str | None = None,
        resume_prior_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run the tool-call benchmark against a model and persist results.

        When ``resume_run_id`` is set, the run reuses the original run ID.
        When ``resume_prior_results`` is provided (a list of scenario result
        dicts from a previous run), those results are merged into the final
        summary so the stored run contains the complete result set.
        """
        adapter = self._adapter_for(backend)

        # Compute reference day name from date if provided
        ref_day: str | None = None
        if reference_date:
            try:
                ref_day = datetime.strptime(reference_date, "%Y-%m-%d").strftime("%A")
            except ValueError:
                raise ValueError(
                    f"Invalid --reference-date '{reference_date}'. "
                    f"Expected format: YYYY-MM-DD (e.g. 2026-03-20)"
                ) from None

        # Resolve scenarios: explicit list > ID filter > base default
        if scenarios is not None:
            resolved = scenarios
        elif scenario_ids:
            requested = set(scenario_ids)
            resolved = [s for s in ALL_SCENARIOS if s.id in requested]
            missing = requested - {s.id for s in resolved}
            if missing:
                raise ValueError(f"Unknown scenario IDs: {', '.join(sorted(missing))}")
        else:
            resolved = ALL_SCENARIOS
        report_scenarios = resolved

        # Build metadata from RunContext (preferred) or legacy probe
        if run_context:
            metadata = run_context.to_dict()
        else:
            metadata = await _collect_metadata_safe(model, backend, base_url, api_key)

        run_config = _build_run_config(
            model=model,
            backend=backend,
            base_url=base_url,
            scenarios=resolved,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
            seed=seed,
            reference_date=reference_date,
            concurrency=concurrency,
            error_rate=error_rate,
            alpha=alpha,
            extra_params=extra_params,
            context_pressure_config=context_pressure_config,
            weight_by_difficulty=weight_by_difficulty,
            metadata=metadata,
        )

        # Build run ID (reuse original for resumed runs)
        run_id = resume_run_id or build_run_id(run_config)

        # Run all scenarios (close adapter connection pool when done)
        try:
            summary = await run_all_scenarios(
                adapter,
                model=model,
                base_url=base_url,
                api_key=api_key,
                scenarios=resolved,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                seed=seed,
                reference_date=reference_date,
                reference_day=ref_day,
                on_scenario_start=on_scenario_start,
                on_scenario_result=on_scenario_result,
                concurrency=concurrency,
                error_rate=error_rate,
                alpha=alpha,
                extra_params=extra_params,
                context_pressure_messages=context_pressure_messages,
                weight_by_difficulty=weight_by_difficulty,
            )
        finally:
            if hasattr(adapter, "aclose"):
                await adapter.aclose()

        # Merge prior results for resumed runs
        if resume_prior_results:
            merged_results = list(summary.scenario_results)
            existing_ids = {r.scenario_id for r in merged_results}
            for pr in resume_prior_results:
                if pr.get("scenario_id") not in existing_ids:
                    merged_results.append(ScenarioResult.from_dict(pr))
            scenario_by_id = {s.id: s for s in [*ALL_SCENARIOS, *resolved]}
            missing_ids = {r.scenario_id for r in merged_results} - scenario_by_id.keys()
            if missing_ids:
                raise ValueError(
                    "Cannot resume unknown scenarios: " + ", ".join(sorted(missing_ids))
                )
            result_by_id = {r.scenario_id: r for r in merged_results}
            ordered_ids = list(
                dict.fromkeys(s.id for s in [*ALL_SCENARIOS, *resolved] if s.id in result_by_id)
            )
            merged_results = [result_by_id[scenario_id] for scenario_id in ordered_ids]
            merged_scenarios = [scenario_by_id[scenario_id] for scenario_id in ordered_ids]
            report_scenarios = merged_scenarios
            summary = score_results(
                merged_results,
                merged_scenarios,
                alpha=alpha,
                weight_by_difficulty=weight_by_difficulty,
            )
            run_config = _build_run_config(
                model=model,
                backend=backend,
                base_url=base_url,
                scenarios=merged_scenarios,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_turns=max_turns,
                seed=seed,
                reference_date=reference_date,
                concurrency=concurrency,
                error_rate=error_rate,
                alpha=alpha,
                extra_params=extra_params,
                context_pressure_config=context_pressure_config,
                weight_by_difficulty=weight_by_difficulty,
                metadata=metadata,
            )
            logger.info(
                "Resume merge: %d prior + %d new = %d total scenarios (score: %d)",
                len(merged_results) - len(existing_ids),
                len(existing_ids),
                len(merged_results),
                summary.final_score,
            )

        # Persist
        run_data = {
            "run_id": run_id,
            "status": "completed",
            "config": run_config,
            "scores": summary.to_dict(),
            "metadata": metadata,
            "safety_gate": {
                "passed": not summary.safety_warnings,
                "warnings": summary.safety_warnings,
            },
        }

        report_writer = None
        if self.reporter is not None:
            reporter = self.reporter
            scenario_metadata = {
                scenario.id: ScenarioReportMetadata(
                    title=scenario.title,
                    category=scenario.category,
                    difficulty=scenario.difficulty,
                )
                for scenario in report_scenarios
            }

            def write_scenario_report() -> Any:
                return reporter.write_scenario_report(
                    run_id,
                    model,
                    summary,
                    throughput_samples=throughput_samples or [],
                    context_pressure_config=context_pressure_config,
                    run_context=run_context,
                    scenario_metadata=scenario_metadata,
                )

            report_writer = write_scenario_report
        finalize_completed_run(
            run_data,
            write_report=report_writer,
            persist=self.repo.upsert_scenario_run if self.repo is not None else None,
        )

        return run_data


def _build_run_config(
    *,
    model: str,
    backend: str,
    base_url: str,
    scenarios: list[ScenarioDefinition],
    temperature: float,
    timeout_seconds: float,
    max_turns: int,
    seed: int | None,
    reference_date: str | None,
    concurrency: int,
    error_rate: float,
    alpha: float,
    extra_params: dict[str, Any] | None,
    context_pressure_config: dict[str, Any] | None,
    weight_by_difficulty: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build persisted config and its deterministic comparison fingerprint."""
    config: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "base_url": _redact_url(base_url),
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "max_turns": max_turns,
        "seed": seed,
        "reference_date": reference_date,
        "scenario_count": len(scenarios),
        "scenario_ids": [s.id for s in scenarios],
        "concurrency": concurrency,
        "error_rate": error_rate,
        "alpha": alpha,
        "extra_params": extra_params,
        "weight_by_difficulty": weight_by_difficulty,
    }
    if context_pressure_config:
        config["context_pressure"] = context_pressure_config
    comparison_context = {
        key: metadata.get(key)
        for key in (
            "server_model_id",
            "server_model_root",
            "engine_name",
            "engine_version",
            "quantization",
            "gpu_count",
            "spec_decoding",
        )
        if metadata.get(key) is not None
    }
    fingerprint_config = {**config, "scenario_ids": sorted(config["scenario_ids"])}
    from tool_eval_bench import __version__

    config["config_fingerprint"] = build_config_fingerprint(
        {
            "config": fingerprint_config,
            "deployment": comparison_context,
            "tool_version": __version__,
        }
    )
    return config


async def _collect_metadata_safe(
    model: str, backend: str, base_url: str, api_key: str | None
) -> dict[str, Any]:
    """Collect run metadata (legacy path), swallowing errors."""
    try:
        from tool_eval_bench.domain.models import BenchmarkConfig
        from tool_eval_bench.utils.metadata import collect_run_metadata

        config = BenchmarkConfig(model=model, backend=backend, base_url=base_url, api_key=api_key)
        return await collect_run_metadata(config)
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
        logger.warning("Failed to collect metadata: %s", exc)
        return {"error": str(exc)}
