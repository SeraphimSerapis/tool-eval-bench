"""Benchmark runner service — orchestrates scenario-based tool-call evaluation.

This replaces the old throughput-focused runner with the new multi-turn
scenario benchmark system.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from tool_eval_bench.adapters.base import BackendAdapter
from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter
from tool_eval_bench.domain.scenarios import (
    OnScenarioResult,
    OnScenarioStart,
    ScenarioDefinition,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
from tool_eval_bench.runner.orchestrator import run_all_scenarios
from tool_eval_bench.storage.db import RunRepository
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.utils.ids import build_run_id
from tool_eval_bench.utils.metadata import collect_run_metadata

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = {"vllm", "litellm", "llamacpp", "llama.cpp", "llama_cpp"}


class BenchmarkService:
    def __init__(
        self,
        repo: RunRepository | None = None,
        reporter: MarkdownReporter | None = None,
    ) -> None:
        self.repo = repo or RunRepository()
        self.reporter = reporter or MarkdownReporter()

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
        timeout_seconds: float = 60.0,
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
    ) -> dict[str, Any]:
        """Run the tool-call benchmark against a model and persist results."""
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
                )

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

        # Build run ID
        run_config = {
            "model": model,
            "backend": backend,
            "base_url": base_url,
            "scenarios": [s.id for s in resolved],
        }
        run_id = build_run_id(run_config)
        metadata = await collect_run_metadata_safe(model, backend, base_url, api_key)

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
            )
        finally:
            if hasattr(adapter, "aclose"):
                await adapter.aclose()

        # Persist
        run_data = {
            "run_id": run_id,
            "status": "completed",
            "config": {
                "model": model,
                "backend": backend,
                "base_url": base_url,
                "temperature": temperature,
                "timeout_seconds": timeout_seconds,
                "max_turns": max_turns,
                "seed": seed,
                "reference_date": reference_date,
                "scenario_count": len(resolved),
                "scenario_ids": [s.id for s in resolved],
            },
            "scores": summary.to_dict(),
            "metadata": metadata,
        }

        self.repo.upsert_scenario_run(run_data)
        report_path = self.reporter.write_scenario_report(
            run_id, model, summary,
            throughput_samples=throughput_samples or [],
        )

        run_data["report_path"] = str(report_path)
        return run_data


async def collect_run_metadata_safe(
    model: str, backend: str, base_url: str, api_key: str | None
) -> dict[str, Any]:
    """Collect run metadata, swallowing errors."""
    try:
        from tool_eval_bench.domain.models import BenchmarkConfig

        config = BenchmarkConfig(model=model, backend=backend, base_url=base_url, api_key=api_key)
        return await collect_run_metadata(config)
    except Exception as exc:
        logger.warning("Failed to collect metadata: %s", exc)
        return {"error": str(exc)}

