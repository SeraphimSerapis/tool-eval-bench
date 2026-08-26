"""Benchmark runner service — orchestrates scenario-based tool-call evaluation.

This replaces the old throughput-focused runner with the new multi-turn
scenario benchmark system.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

import httpx

from tool_eval_bench.adapters.factory import build_adapter
from tool_eval_bench.adapters.openai_compat import RateLimitObserver
from tool_eval_bench.application.finalization import finalize_completed_run
from tool_eval_bench.application.run_config import RunSettings, build_run_config
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
from tool_eval_bench.storage.db import (
    RUN_STATUS_INTERRUPTED,
    RUN_STATUS_RUNNING,
    RunRepository,
)
from tool_eval_bench.storage.reports import MarkdownReporter
from tool_eval_bench.utils.ids import build_run_id

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = {
    "vllm",
    "litellm",
    "llamacpp",
    "llama.cpp",
    "llama_cpp",
    "sglang",
    "gemini",
    "ninfer",
}


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

    def _adapter_for(
        self,
        backend: str,
        base_url: str = "",
        wire_format: str | None = None,
    ) -> BackendAdapter:
        """Return the adapter for a backend label and endpoint.

        ``backend`` is a reporting label (vllm, litellm, …); the request format
        follows the endpoint itself, detected from ``base_url`` unless
        ``wire_format`` names one explicitly.
        """
        backend_l = backend.lower()
        if backend_l not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                "Supported: vllm, litellm, llamacpp, sglang, gemini, ninfer"
            )
        return build_adapter(base_url, wire_format=wire_format)

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
        resume_scenarios: list[ScenarioDefinition] | None = None,
        scenario_packs: list[dict[str, Any]] | None = None,
        rate_limit_observer: RateLimitObserver | None = None,
        wire_format: str | None = None,
    ) -> dict[str, Any]:
        """Run the tool-call benchmark against a model and persist results.

        When ``resume_run_id`` is set, the run reuses the original run ID.
        When ``resume_prior_results`` is provided (a list of scenario result
        dicts from a previous run), those results are merged into the final
        summary so the stored run contains the complete result set.
        ``resume_scenarios`` retains definitions that are not in the rerun
        subset, including held-out pack and Hard Mode scenarios.
        """
        adapter = self._adapter_for(backend, base_url, wire_format)
        if rate_limit_observer is not None:
            setter = getattr(adapter, "set_rate_limit_observer", None)
            if setter is not None:
                setter(rate_limit_observer)

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

        # Resume executes only the unresolved subset, but the durable run
        # configuration must keep the original complete protocol.  Otherwise a
        # second interruption would overwrite its scenario identity and make a
        # later resume appear incompatible.
        config_scenarios = resume_scenarios or resolved
        # Captured once: the resume path below rebuilds the config against a
        # different scenario list, and the fingerprint must not otherwise move.
        settings = RunSettings(
            model=model,
            backend=backend,
            base_url=base_url,
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
        )
        run_config = build_run_config(
            settings,
            scenarios=config_scenarios,
            metadata=metadata,
            scenario_packs=scenario_packs,
        )

        # Build run ID (reuse original for resumed runs)
        run_id = resume_run_id or build_run_id(run_config)

        # Claim the run row up front and checkpoint each finished scenario, so an
        # interrupted long run can be resumed instead of thrown away.
        self._claim_run(run_id, model, run_config, metadata)
        checkpointing_result_cb = self._checkpointing_callback(run_id, on_scenario_result)

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
                on_scenario_result=checkpointing_result_cb,
                concurrency=concurrency,
                error_rate=error_rate,
                alpha=alpha,
                extra_params=extra_params,
                context_pressure_messages=context_pressure_messages,
                weight_by_difficulty=weight_by_difficulty,
            )
        except BaseException:
            # Covers KeyboardInterrupt and CancelledError as well as errors —
            # the partial results stay on disk under this run ID.
            self._mark_interrupted(run_id)
            raise
        finally:
            status = getattr(adapter, "rate_limit_status", None)
            if status is not None and status.retries:
                logger.info(
                    "Endpoint rate-limited this run: %d retries, %.0fs spent waiting",
                    status.retries,
                    status.total_wait_seconds,
                )
            if hasattr(adapter, "aclose"):
                await adapter.aclose()

        # Merge prior results for resumed runs
        if resume_prior_results:
            merged_results = list(summary.scenario_results)
            existing_ids = {r.scenario_id for r in merged_results}
            for pr in resume_prior_results:
                if pr.get("scenario_id") not in existing_ids:
                    merged_results.append(ScenarioResult.from_dict(pr))
            scenario_by_id = {
                s.id: s for s in [*(resume_scenarios or []), *ALL_SCENARIOS, *resolved]
            }
            missing_ids = {r.scenario_id for r in merged_results} - scenario_by_id.keys()
            if missing_ids:
                raise ValueError(
                    "Cannot resume unknown scenarios: " + ", ".join(sorted(missing_ids))
                )
            result_by_id = {r.scenario_id: r for r in merged_results}
            ordered_ids = list(
                dict.fromkeys(
                    s.id
                    for s in [*(resume_scenarios or []), *ALL_SCENARIOS, *resolved]
                    if s.id in result_by_id
                )
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
            run_config = build_run_config(
                settings,
                scenarios=merged_scenarios,
                metadata=metadata,
                scenario_packs=scenario_packs,
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
                    held_out=scenario.held_out,
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
                    scenario_packs=scenario_packs,
                )

            report_writer = write_scenario_report
        try:
            finalize_completed_run(
                run_data,
                write_report=report_writer,
                persist=self.repo.upsert_scenario_run if self.repo is not None else None,
            )
        except BaseException:
            # The scenarios ran but the run never reached a reportable state.
            # Leave the checkpoints in place so the work can be recovered.
            self._mark_interrupted(run_id)
            raise
        # Final scores now hold everything the checkpoints held.
        if self.repo is not None:
            try:
                self.repo.clear_checkpoints(run_id)
            except Exception as exc:  # noqa: BLE001 — cleanup must not fail a good run
                logger.warning("Failed to clear checkpoints for run %s: %s", run_id, exc)

        return run_data

    # -- Crash-resilience helpers ------------------------------------------

    def _claim_run(
        self,
        run_id: str,
        model: str,
        run_config: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Insert a ``running`` row so an interrupted run is discoverable."""
        if self.repo is None:
            return
        try:
            existing = self.repo.get(run_id, include_traces=False)
            if isinstance(existing, dict) and existing.get("status") == "completed":
                raise ValueError(
                    f"Run {run_id} is already completed and its results are immutable."
                )
            self.repo.upsert_scenario_run(
                {
                    "run_id": run_id,
                    "status": RUN_STATUS_RUNNING,
                    "config": run_config,
                    "scores": {},
                    "metadata": metadata,
                }
            )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 — never block a run on bookkeeping
            logger.warning("Could not claim run row %s: %s", run_id, exc)

    def _mark_interrupted(self, run_id: str) -> None:
        if self.repo is None:
            return
        try:
            self.repo.mark_run_status(run_id, RUN_STATUS_INTERRUPTED)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not mark run %s interrupted: %s", run_id, exc)
        else:
            logger.info("Run %s interrupted; resume with --resume %s", run_id, run_id)

    def _checkpointing_callback(
        self, run_id: str, inner: OnScenarioResult | None
    ) -> OnScenarioResult | None:
        """Wrap the caller's result callback with durable checkpointing."""
        if self.repo is None:
            return inner
        repo = self.repo

        async def on_result(
            scenario: ScenarioDefinition,
            result: ScenarioResult,
            index: int,
            total: int,
        ) -> None:
            try:
                # Offloaded to a worker thread: a synchronous fsync here stalls
                # every scenario in flight once --parallel is above 1.
                await repo.acheckpoint_scenario_result(run_id, result.to_dict())
            except Exception as exc:  # noqa: BLE001 — a lost checkpoint is not fatal
                logger.warning("Failed to checkpoint %s: %s", result.scenario_id, exc)
            if inner is not None:
                await inner(scenario, result, index, total)

        return on_result


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
