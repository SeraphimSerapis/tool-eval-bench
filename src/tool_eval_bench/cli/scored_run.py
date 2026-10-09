"""The inputs of one scored tool-call run, read from the CLI namespace once.

The live, plain, and JSON runners and the resume compatibility check all need
the same values. Before this module each built them from ``args`` by hand, so a
flag could reach the service call but not the resume comparison, or the
reverse. ``ScoredRun`` is now the one place that turns the flags, and the
``args._resume_*``, ``args._request_headers`` and ``args._session_header``
attributes ``dispatch`` leaves on the namespace, into ``run_benchmark`` kwargs
and the resume ``RunSettings``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tool_eval_bench.application.decision_audit import decision_judge_config, with_selected_checks
from tool_eval_bench.application.run_config import RunSettings
from tool_eval_bench.domain.models import ChatMessage, RunContext
from tool_eval_bench.domain.scenarios import ScenarioDefinition

DECISION_JUDGE_API_KEY_ENV = "TOOL_EVAL_DECISION_JUDGE_API_KEY"


@dataclass(frozen=True)
class JudgeConnection:
    """The decision-judge flags as given; ``None`` on ``ScoredRun`` when none was."""

    base_url: str | None
    model: str | None
    judge_set: str | None
    # Read only with a base URL: the judge key is never sent anywhere else.
    api_key: str | None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> JudgeConnection | None:
        base_url = getattr(args, "decision_judge_base_url", None)
        model = getattr(args, "decision_judge_model", None)
        judge_set = getattr(args, "decision_judge", None)
        if base_url is None and model is None and judge_set is None:
            return None
        api_key = os.environ.get(DECISION_JUDGE_API_KEY_ENV) if base_url is not None else None
        return cls(base_url=base_url, model=model, judge_set=judge_set, api_key=api_key)

    def service_kwargs(self) -> dict[str, Any]:
        """The service's judge kwargs; empty without a base URL, as the service expects."""
        if self.base_url is None:
            return {}
        return {
            "decision_judge_base_url": self.base_url,
            "decision_judge_model": self.model,
            "decision_judge_api_key": self.api_key,
            "decision_judge": self.judge_set,
        }


@dataclass(frozen=True)
class ScoredRun:
    """Every ``BenchmarkService.run_benchmark`` input except callbacks and perf samples."""

    model: str
    backend: str
    base_url: str
    api_key: str | None
    wire_format: str
    extra_headers: Mapping[str, str] | None
    session_header: str | None
    scenarios: list[ScenarioDefinition]
    temperature: float
    timeout_seconds: float
    max_turns: int
    reference_date: str | None
    seed: int | None
    variant_seed: int | None
    concurrency: int
    error_rate: float
    alpha: float
    weight_by_difficulty: bool
    system_prompt: str | None
    extra_params: dict[str, Any] | None
    context_pressure_messages: list[ChatMessage] | None
    context_pressure_config: dict[str, Any] | None
    run_context: RunContext | None
    scenario_packs: list[dict[str, Any]] | None
    resume_run_id: str | None
    resume_prior_results: list[dict[str, Any]] | None
    resume_scenarios: list[ScenarioDefinition] | None
    judge: JudgeConnection | None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        model: str,
        backend: str,
        base_url: str,
        api_key: str | None,
        wire_format: str,
        scenarios: list[ScenarioDefinition],
        extra_params: dict[str, Any] | None,
        scenario_packs: list[dict[str, Any]] | None,
        context_pressure_messages: list[ChatMessage] | None = None,
        context_pressure_config: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
        resume_scenarios: list[ScenarioDefinition] | None = None,
    ) -> ScoredRun:
        """Read the run from ``args`` at the moment of the call.

        Call it where the run starts, not earlier: ``dispatch`` scales
        ``args.timeout`` for context pressure and rewrites the resume
        attributes before the run, and both must be seen. A non-None
        ``resume_scenarios`` replaces ``args._resume_scenarios``.
        """
        return cls(
            model=model,
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            wire_format=wire_format,
            extra_headers=getattr(args, "_request_headers", None),
            session_header=getattr(args, "_session_header", None),
            scenarios=scenarios,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            max_turns=args.max_turns,
            reference_date=args.reference_date,
            seed=args.seed,
            variant_seed=getattr(args, "variant_seed", None),
            concurrency=args.parallel,
            error_rate=args.error_rate,
            alpha=args.alpha,
            weight_by_difficulty=getattr(args, "weight_by_difficulty", False),
            system_prompt=getattr(args, "system_prompt", None),
            extra_params=extra_params,
            context_pressure_messages=context_pressure_messages,
            context_pressure_config=context_pressure_config,
            run_context=run_context,
            scenario_packs=scenario_packs,
            resume_run_id=getattr(args, "_resume_run_id", None),
            resume_prior_results=getattr(args, "_resume_prior_results", None),
            resume_scenarios=(
                resume_scenarios
                if resume_scenarios is not None
                else getattr(args, "_resume_scenarios", None)
            ),
            judge=JudgeConnection.from_args(args),
        )

    @property
    def audits_answers(self) -> bool:
        """Whether a decision judge runs, so the runner should attach an audit callback."""
        return self.judge is not None and self.judge.base_url is not None

    def service_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``run_benchmark``, without callbacks or perf samples.

        The judge kwargs are present only when a judge base URL is configured.
        """
        return {
            "model": self.model,
            "backend": self.backend,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "scenarios": self.scenarios,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_turns": self.max_turns,
            "reference_date": self.reference_date,
            "system_prompt": self.system_prompt,
            "variant_seed": self.variant_seed,
            "seed": self.seed,
            "concurrency": self.concurrency,
            "error_rate": self.error_rate,
            "alpha": self.alpha,
            "extra_params": self.extra_params,
            "context_pressure_messages": self.context_pressure_messages,
            "context_pressure_config": self.context_pressure_config,
            "run_context": self.run_context,
            "weight_by_difficulty": self.weight_by_difficulty,
            "resume_run_id": self.resume_run_id,
            "resume_prior_results": self.resume_prior_results,
            "resume_scenarios": self.resume_scenarios,
            "scenario_packs": self.scenario_packs,
            "wire_format": self.wire_format,
            "extra_headers": self.extra_headers,
            "session_header": self.session_header,
            **(self.judge.service_kwargs() if self.judge is not None else {}),
        }

    def config_scenarios(self) -> list[ScenarioDefinition]:
        """The scenarios the service records in the run config.

        A resumed run records its complete original protocol, not the rerun
        subset, with the run's variants applied, exactly as the service does.
        """
        from tool_eval_bench.evals.variants import apply_variants

        return apply_variants(self.resume_scenarios or self.scenarios, self.variant_seed)

    def run_settings(self) -> RunSettings:
        """The persisted-config view of this run; resume compares it with the stored one."""
        judge_config = None
        if self.judge is not None:
            # Validates as the service does, so an incomplete judge still raises.
            judge_config = decision_judge_config(
                self.judge.base_url, self.judge.model, judge_set=self.judge.judge_set
            )
        return RunSettings(
            model=self.model,
            backend=self.backend,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            max_turns=self.max_turns,
            seed=self.seed,
            reference_date=self.reference_date,
            concurrency=self.concurrency,
            error_rate=self.error_rate,
            alpha=self.alpha,
            extra_params=self.extra_params,
            context_pressure_config=self.context_pressure_config,
            weight_by_difficulty=self.weight_by_difficulty,
            system_prompt=self.system_prompt,
            decision_judge=(
                with_selected_checks(judge_config, self.config_scenarios())
                if judge_config is not None
                else None
            ),
        )
