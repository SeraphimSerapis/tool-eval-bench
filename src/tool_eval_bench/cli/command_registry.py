"""Authoritative metadata for the public CLI command surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONNECTION = ("model", "backend", "base_url", "api_key")
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
)
RUN_CONTROL = (
    "timeout",
    "max_turns",
    "trials",
    "parallel",
    "error_rate",
    "no_warmup",
    "reference_date",
)
SCENARIOS = ("scenarios", "categories", "short", "hardmode", "hardmode_only")
PERF = (
    "perf",
    "perf_only",
    "perf_legacy",
    "perf_legacy_only",
    "pp",
    "tg",
    "depth",
    "concurrency",
    "benchy_runs",
    "benchy_latency_mode",
    "benchy_args",
    "skip_coherence",
)
SPEC = (
    "spec_bench",
    "spec_method",
    "baseline_tgs",
    "spec_prompts",
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
)


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
        "plugin",
        "Run an external accuracy benchmark",
        translation="plugin",
        help_dests=CONNECTION + SAMPLING + RUN_CONTROL + OUTPUT,
        legacy_flags=("gsm8k_only", "mmlu_only", "ifeval_only"),
        choices=("gsm8k", "mmlu", "ifeval"),
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
        help_dests=CONNECTION + SAMPLING + SCENARIOS + RUN_CONTROL + OUTPUT,
        legacy_flags=("resume",),
    ),
)

COMMAND_REGISTRY = {spec.name: spec for spec in COMMAND_SPECS}
KNOWN_COMMANDS = frozenset(COMMAND_REGISTRY)


def commands_schema() -> dict[str, dict[str, Any]]:
    """Return the public command mapping in registry order."""
    return {name: spec.schema() for name, spec in COMMAND_REGISTRY.items()}
