"""Authoritative metadata for the public CLI command surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONNECTION = (
    "model",
    "backend",
    "base_url",
    "api_key",
    "provider",
    "format",
    "header",
    "session_header",
)
SAMPLING = (
    "temperature",
    "no_think",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "seed",
    "backend_kwargs",
)
OUTPUT = (
    "json",
    "json_file",
    "no_live",
    "redact_url",
    "alpha",
    "no_probe_engine",
    "output_dir",
    "label",
)
RUN_CONTROL = (
    "timeout",
    "max_turns",
    "trials",
    "parallel",
    "error_rate",
    "no_warmup",
    "no_preflight",
    "reference_date",
    "system_prompt",
    "system_prompt_file",
)
DECISION_JUDGE = ("decision_judge", "decision_judge_base_url", "decision_judge_model")
# Only scored tool-call runs produce safety warnings, so plugin help omits it.
SAFETY_GATE = ("fail_on_safety",)
SCENARIOS = (
    "scenarios",
    "categories",
    "short",
    "hardmode",
    "hardmode_only",
    "variant_seed",
    "scenario_pack",
    "pack_only",
    "include_held_out",
)
PERF = (
    "perf",
    "perf_only",
    "pp",
    "tg",
    "depth",
    "concurrency",
    "benchy_runs",
    "benchy_latency_mode",
    "benchy_args",
    "tokenizer",
    "skip_coherence",
)
SPEC = (
    "spec_bench",
    "spec_method",
    "baseline_tgs",
    "spec_prompts",
    "spec_prompt_file",
    "spec_runs",
    "metrics_url",
)
PRESSURE = ("context_pressure", "context_size", "context_pressure_sweep", "sweep_steps")
PLUGIN_LEGACY = (
    "gsm8k",
    "gsm8k_only",
    "gsm8k_shots",
    "gsm8k_limit",
    "gsm8k_shuffle",
    "mmlu",
    "mmlu_only",
    "mmlu_shots",
    "mmlu_limit",
    "mmlu_subjects",
    "ifeval",
    "ifeval_only",
    "ifeval_limit",
    "needle",
    "needle_only",
    "needle_depths",
    "needle_lengths",
    "decision_bench",
    "decision_bench_only",
)
# Plugin name -> stem of its flat flags (``--<stem>`` / ``--<stem>-only``) and
# Namespace attributes. The decision plugin's flags are ``--decision-bench*``
# so they cannot be confused with the ``--decision-judge-*`` answer audit.
PLUGIN_FLAG_STEMS = {
    "gsm8k": "gsm8k",
    "mmlu": "mmlu",
    "ifeval": "ifeval",
    "needle": "needle",
    "decision": "decision_bench",
}


@dataclass(frozen=True)
class CommandSpec:
    """Declarative command metadata shared by help, translation, and schema."""

    name: str
    description: str
    translation: str = "prefix"
    legacy_prefix: tuple[str, ...] = ()
    help_dests: tuple[str, ...] = ()
    legacy_flags: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    alias_for: str | None = None

    def schema(self) -> dict[str, Any]:
        data: dict[str, Any] = {"description": self.description}
        if not self.alias_for:
            data["legacy_flags"] = list(self.legacy_flags)
        if self.choices:
            data["choices"] = list(self.choices)
        if self.modes:
            data["modes"] = list(self.modes)
        if self.alias_for:
            data["alias_for"] = self.alias_for
        return data


COMMAND_SPECS = (
    CommandSpec(
        "run",
        "Run tool-call scenarios",
        translation="passthrough",
        help_dests=CONNECTION
        + SAMPLING
        + SCENARIOS
        + RUN_CONTROL
        + OUTPUT
        + PRESSURE
        + DECISION_JUDGE
        + SAFETY_GATE
        + ("dry_run", "resume", "diff", "weight_by_difficulty"),
    ),
    CommandSpec(
        "probe",
        "Check whether an inference server is reachable",
        legacy_prefix=("--probe",),
        help_dests=CONNECTION + ("json", "redact_url"),
        legacy_flags=("probe",),
    ),
    CommandSpec(
        "bench",
        "Run throughput, speculative, or pressure benchmarks",
        translation="passthrough",
        help_dests=CONNECTION
        + SAMPLING
        + RUN_CONTROL
        + OUTPUT
        + PERF
        + SPEC
        + PRESSURE
        + PLUGIN_LEGACY
        + SCENARIOS
        + SAFETY_GATE
        + ("skip_tool_eval",),
        legacy_flags=("perf", "perf_only", "spec_bench", "context_pressure_sweep"),
    ),
    CommandSpec(
        "spec-live",
        "Live-monitor speculative decoding metrics",
        legacy_prefix=("--spec-live",),
        help_dests=CONNECTION + ("spec_live_interval", "spec_method", "metrics_url", "redact_url"),
        legacy_flags=("spec_live",),
    ),
    CommandSpec(
        "decision-live",
        "Live-monitor a decision model with canary probes",
        legacy_prefix=("--decision-live",),
        help_dests=CONNECTION + ("decision_live_interval", "metrics_url", "redact_url"),
        legacy_flags=("decision_live",),
    ),
    CommandSpec(
        "plugin",
        "Run an external accuracy benchmark",
        translation="plugin",
        help_dests=CONNECTION + SAMPLING + RUN_CONTROL + OUTPUT,
        legacy_flags=tuple(f"{stem}_only" for stem in PLUGIN_FLAG_STEMS.values()),
        choices=tuple(PLUGIN_FLAG_STEMS),
    ),
    CommandSpec(
        "compare",
        "Compare stored runs or Markdown reports",
        translation="compare",
        legacy_flags=("compare",),
        modes=("runs", "report"),
    ),
    CommandSpec(
        "compare-report",
        "Alias for compare --report",
        translation="alias",
        alias_for="compare",
    ),
    CommandSpec(
        "history",
        "List recent runs",
        legacy_prefix=("--history",),
        legacy_flags=("history",),
    ),
    CommandSpec(
        "leaderboard",
        "Show the model leaderboard",
        legacy_prefix=("--leaderboard",),
        legacy_flags=("leaderboard",),
    ),
    CommandSpec(
        "export",
        "Export stored runs",
        translation="export",
        legacy_flags=("export",),
    ),
    CommandSpec(
        "resume",
        "Resume an incomplete run",
        translation="resume",
        help_dests=CONNECTION
        + SAMPLING
        + SCENARIOS
        + RUN_CONTROL
        + OUTPUT
        + DECISION_JUDGE
        + SAFETY_GATE,
        legacy_flags=("resume",),
    ),
)

COMMAND_REGISTRY = {spec.name: spec for spec in COMMAND_SPECS}
KNOWN_COMMANDS = frozenset(COMMAND_REGISTRY)


def commands_schema() -> dict[str, dict[str, Any]]:
    """Return the public command mapping in registry order."""
    return {name: spec.schema() for name, spec in COMMAND_REGISTRY.items()}
