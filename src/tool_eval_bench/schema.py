"""Machine-readable argument schema for tool-eval-bench.

Consumed by external integrators (e.g. sparkrun recipe validation) to
validate ``benchmark.args`` without running the CLI.

Usage::

    from tool_eval_bench.api import ARGS_SCHEMA
    # or
    from tool_eval_bench.schema import ARGS_SCHEMA

The schema is a list of dicts, one per argument, with type, default,
choices, and description.  This is NOT JSON Schema — it's a lightweight
format optimized for recipe validation and CLI introspection.

Every *public* parser argument (i.e. not decorated with
``help=argparse.SUPPRESS``) MUST appear here.  The drift-detection test
``tests/test_api.py::TestArgsSchema::test_all_parser_args_in_schema_or_hidden``
enforces this automatically — add new flags here when you add them to
``cli/bench.py``.

Parser-internal destinations are listed in ``cli/bench._HIDDEN_ARGS`` and are
excluded from this schema.
"""

from __future__ import annotations

from typing import Any

from tool_eval_bench.cli.command_registry import commands_schema
from tool_eval_bench.domain.models import DEFAULT_REQUEST_TIMEOUT_SECONDS

# Argument schema version — bump when adding/removing/renaming args.
SCHEMA_VERSION = "7"

COMMANDS_SCHEMA: dict[str, dict[str, Any]] = commands_schema()

ARGS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "version",
        "type": "bool",
        "default": False,
        "description": "Show tool-eval-bench version and exit",
    },
    # -- Connection --
    {
        "name": "model",
        "type": "string",
        "default": None,
        "description": "Model name/path (auto-detected from /v1/models if omitted)",
    },
    {
        "name": "backend",
        "type": "string",
        "default": "vllm",
        "choices": ["vllm", "litellm", "llamacpp", "sglang", "gemini", "ninfer"],
        "description": "Backend label for reports",
    },
    {
        "name": "base_url",
        "type": "string",
        "default": None,
        "description": "Server base URL (default: auto-discover on localhost, or from .env)",
    },
    {
        "name": "api_key",
        "type": "string",
        "default": None,
        "description": "API key (optional; can also be set via TOOL_EVAL_API_KEY env var)",
    },
    {
        "name": "format",
        "type": "string",
        "default": "auto",
        "choices": ["auto", "openai", "gemini"],
        "description": "Endpoint wire format (default: detect from the base URL)",
    },
    {
        "name": "probe",
        "type": "bool",
        "default": False,
        "description": "Check server reachability and exit (0 = ready, 1 = not found)",
    },
    # -- Sampling --
    {
        "name": "temperature",
        "type": "float",
        "default": 0.0,
        "description": "Sampling temperature (0.0 = greedy)",
    },
    {
        "name": "no_think",
        "type": "bool",
        "default": False,
        "description": "Disable thinking/reasoning (sets enable_thinking=false)",
    },
    {
        "name": "top_p",
        "type": "float",
        "default": None,
        "description": "Top-p (nucleus) sampling",
    },
    {
        "name": "top_k",
        "type": "int",
        "default": None,
        "description": "Top-k sampling",
    },
    {
        "name": "min_p",
        "type": "float",
        "default": None,
        "description": "Min-p sampling threshold",
    },
    {
        "name": "repeat_penalty",
        "type": "float",
        "default": None,
        "description": "Repetition penalty",
    },
    {
        "name": "seed",
        "type": "int",
        "default": None,
        "description": "Random seed (passed to server)",
    },
    {
        "name": "backend_kwargs",
        "type": "string",
        "default": None,
        "description": "JSON dict merged into API payload; overrides individual flags",
    },
    # -- Scenario selection --
    {
        "name": "scenarios",
        "type": "list[string]",
        "default": None,
        "description": "Specific scenario IDs to run (e.g. TC-01 TC-07)",
    },
    {
        "name": "categories",
        "type": "list[string]",
        "default": None,
        "choices": list("ABCDEFGHIJKLMNOP"),
        "description": "Run only specific categories (A–P)",
    },
    {
        "name": "short",
        "type": "bool",
        "default": False,
        "description": "Run only the core 15 scenarios",
    },
    {
        "name": "hardmode",
        "type": "bool",
        "default": False,
        "description": "Include Hard Mode scenarios (Category P)",
    },
    {
        "name": "hardmode_only",
        "type": "bool",
        "default": False,
        "description": "Run ONLY Hard Mode scenarios (Category P)",
    },
    {
        "name": "scenario_pack",
        "type": "list[string]",
        "default": None,
        "description": "Add a held-out YAML scenario pack directory (repeatable)",
    },
    {
        "name": "pack_only",
        "type": "bool",
        "default": False,
        "description": "Run ONLY scenarios from --scenario-pack",
    },
    # -- Run control --
    {
        "name": "timeout",
        "type": "float",
        "default": DEFAULT_REQUEST_TIMEOUT_SECONDS,
        "description": "Request timeout in seconds",
    },
    {
        "name": "max_turns",
        "type": "int",
        "default": 8,
        "description": "Max turns per scenario",
    },
    {
        "name": "trials",
        "type": "int",
        "default": 1,
        "description": "Number of trial runs for statistical rigor",
    },
    {
        "name": "parallel",
        "type": "int",
        "default": 1,
        "description": "Run N scenarios concurrently (1 = sequential)",
    },
    {
        "name": "error_rate",
        "type": "float",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "description": "Inject random tool errors at this rate for robustness testing",
    },
    {
        "name": "no_warmup",
        "type": "bool",
        "default": False,
        "description": "Skip server warm-up request",
    },
    {
        "name": "no_preflight",
        "type": "bool",
        "default": False,
        "description": "Skip the strict model availability pre-flight check",
    },
    {
        "name": "reference_date",
        "type": "string",
        "default": None,
        "description": "Override benchmark reference date (YYYY-MM-DD)",
    },
    {
        "name": "skip_tool_eval",
        "type": "bool",
        "default": False,
        "description": "Skip tool-call scenarios (use with --perf or --spec-bench)",
    },
    # -- Output --
    {
        "name": "json",
        "type": "bool",
        "default": False,
        "description": "Output raw JSON instead of rich display",
    },
    {
        "name": "json_file",
        "type": "string",
        "default": None,
        "description": "Write JSON results to PATH (implies --json; keeps stdout clean)",
    },
    {
        "name": "no_live",
        "type": "bool",
        "default": False,
        "description": "Disable live updating display",
    },
    {
        "name": "alpha",
        "type": "float",
        "default": 0.7,
        "min": 0.0,
        "max": 1.0,
        "description": "Quality/speed weight for deployability score",
    },
    {
        "name": "no_probe_engine",
        "type": "bool",
        "default": False,
        "description": "Skip inference engine probing",
    },
    {
        "name": "redact_url",
        "type": "bool",
        "default": False,
        "description": "Mask the server URL in reports",
    },
    {
        "name": "output_dir",
        "type": "string",
        "default": None,
        "description": "Directory for report files (default: ./runs/)",
    },
    {
        "name": "label",
        "type": "string",
        "default": None,
        "description": (
            "Free-form annotation recorded on every report this run generates "
            "(shown in the report body; a safe slug is added to report filenames)"
        ),
    },
    {
        "name": "dry_run",
        "type": "bool",
        "default": False,
        "description": "Show which scenarios would run, then exit (no server needed)",
    },
    {
        "name": "fail_on_safety",
        "type": "bool",
        "default": False,
        "description": "Exit with status 2 when safety-critical scenarios fail",
    },
    # -- Throughput benchmark (llama-benchy) --
    {
        "name": "perf",
        "type": "bool",
        "default": False,
        "description": "Run llama-benchy throughput benchmark before tool-call scenarios",
    },
    {
        "name": "perf_only",
        "type": "bool",
        "default": False,
        "description": "Run ONLY llama-benchy throughput benchmark (skip tool-call scenarios)",
    },
    {
        "name": "pp",
        "type": "int",
        "default": 2048,
        "description": "Prompt tokens for throughput benchmark",
    },
    {
        "name": "tg",
        "type": "int",
        "default": 128,
        "description": "Generation tokens for throughput benchmark",
    },
    {
        "name": "depth",
        "type": "string",
        "default": "0,4096,8192",
        "description": "Context depths for throughput sweep, comma separated",
    },
    {
        "name": "concurrency",
        "type": "string",
        "default": "1,2,4",
        "description": "Concurrency levels for throughput sweep",
    },
    {
        "name": "benchy_runs",
        "type": "int",
        "default": 3,
        "description": "Measurement runs per test point (llama-benchy)",
    },
    {
        "name": "benchy_latency_mode",
        "type": "string",
        "default": "generation",
        "choices": ["api", "generation", "none"],
        "description": "Latency measurement mode for llama-benchy",
    },
    {
        "name": "benchy_args",
        "type": "string",
        "default": None,
        "description": "Pass-through args for llama-benchy (quoted string)",
    },
    {
        "name": "tokenizer",
        "type": "string",
        "default": None,
        "description": "Local tokenizer.json path for llama-benchy on offline hosts",
    },
    {
        "name": "skip_coherence",
        "type": "bool",
        "default": True,
        "description": "Deprecated: llama-benchy coherence check is now always skipped (retained for compatibility)",
    },
    # -- Speculative decoding benchmark --
    {
        "name": "spec_bench",
        "type": "bool",
        "default": False,
        "description": "Run spec-decode / MTP benchmark (effective t/s, acceptance rate)",
    },
    {
        "name": "spec_live",
        "type": "bool",
        "default": False,
        "description": "Live-monitor speculative decoding stats (polls /metrics, runs until Ctrl+C)",
    },
    {
        "name": "spec_live_interval",
        "type": "float",
        "default": 1.0,
        "description": "Poll interval for --spec-live in seconds",
    },
    {
        "name": "spec_method",
        "type": "string",
        "default": "auto",
        "choices": [
            "auto",
            "mtp",
            "nextn",
            "draft",
            "standalone",
            "dflash",
            "dspark",
            "ngram",
            "ngram_gpu",
            "eagle",
            "eagle3",
            "medusa",
            "mlp_speculator",
            "suffix",
            "custom_class",
        ],
        "description": "Spec-decode method hint",
    },
    {
        "name": "baseline_tgs",
        "type": "float",
        "default": None,
        "description": "Baseline tg t/s for speedup ratio calculation",
    },
    {
        "name": "spec_prompts",
        "type": "string",
        "default": "filler,code,structured",
        "description": "Prompt types for spec-bench, comma separated",
    },
    {
        "name": "metrics_url",
        "type": "string",
        "default": None,
        "description": "Prometheus /metrics URL (when API is behind a proxy)",
    },
    # -- Context pressure --
    {
        "name": "context_pressure",
        "type": "float",
        "default": None,
        "min": 0.0,
        "max": 1.0,
        "description": "Fill context to this ratio before each scenario",
    },
    {
        "name": "context_size",
        "type": "int",
        "default": None,
        "description": "Override auto-detected context window size (tokens)",
    },
    {
        "name": "context_pressure_sweep",
        "type": "string",
        "default": None,
        "description": "Sweep pressure from START to END (e.g. 0.5-1.0)",
    },
    {
        "name": "sweep_steps",
        "type": "int",
        "default": 5,
        "description": "Number of pressure levels to test in a sweep",
    },
    # -- GSM8K benchmark --
    {
        "name": "gsm8k",
        "type": "bool",
        "default": False,
        "description": "Run GSM8K (Grade School Math) benchmark after tool-call scenarios",
    },
    {
        "name": "gsm8k_only",
        "type": "bool",
        "default": False,
        "description": "Run ONLY the GSM8K benchmark (skip tool-call scenarios)",
    },
    {
        "name": "gsm8k_shots",
        "type": "int",
        "default": 8,
        "description": "Number of few-shot CoT examples for GSM8K (0–8)",
    },
    {
        "name": "gsm8k_limit",
        "type": "int",
        "default": 200,
        "description": "Max GSM8K questions to evaluate (0 = all 1319)",
    },
    {
        "name": "gsm8k_shuffle",
        "type": "bool",
        "default": False,
        "description": "Shuffle GSM8K question order",
    },
    # -- MMLU benchmark --
    {
        "name": "mmlu",
        "type": "bool",
        "default": False,
        "description": "Run MMLU benchmark after tool-call scenarios",
    },
    {
        "name": "mmlu_only",
        "type": "bool",
        "default": False,
        "description": "Run ONLY the MMLU benchmark (skip tool-call scenarios)",
    },
    {
        "name": "mmlu_shots",
        "type": "int",
        "default": 5,
        "description": "Number of few-shot examples per subject for MMLU (0–5)",
    },
    {
        "name": "mmlu_limit",
        "type": "int",
        "default": 500,
        "description": "Max MMLU questions to evaluate (0 = all 14042)",
    },
    {
        "name": "mmlu_subjects",
        "type": "string",
        "default": None,
        "description": "Comma-separated subjects or categories for MMLU",
    },
    # -- IFEval benchmark --
    {
        "name": "ifeval",
        "type": "bool",
        "default": False,
        "description": "Run IFEval benchmark after tool-call scenarios",
    },
    {
        "name": "ifeval_only",
        "type": "bool",
        "default": False,
        "description": "Run ONLY the IFEval benchmark (skip tool-call scenarios)",
    },
    {
        "name": "ifeval_limit",
        "type": "int",
        "default": 0,
        "description": "Max IFEval prompts to evaluate (0 = all 541)",
    },
    # -- Needle in a haystack --
    {
        "name": "needle",
        "type": "bool",
        "default": False,
        "description": "Run needle-in-a-haystack retrieval after tool-call scenarios",
    },
    {
        "name": "needle_only",
        "type": "bool",
        "default": False,
        "description": "Run ONLY the needle-in-a-haystack benchmark (skip tool-call scenarios)",
    },
    {
        "name": "needle_depths",
        "type": "int",
        "default": 5,
        "description": "Needle depths to probe, evenly spaced 0-100% of the haystack",
    },
    {
        "name": "needle_lengths",
        "type": "int",
        "default": 4,
        "description": "Haystack sizes to probe, up to the context window",
    },
    # -- History & comparison --
    {
        "name": "diff",
        "type": "string",
        "default": None,
        "description": "Compare against a previous run (use 'latest' for most recent)",
    },
    {
        "name": "compare",
        "type": "list[string]",
        "default": None,
        "description": "Compare two stored runs by ID (provide exactly two run IDs)",
    },
    {
        "name": "history",
        "type": "bool",
        "default": False,
        "description": "List recent benchmark runs and exit",
    },
    {
        "name": "leaderboard",
        "type": "bool",
        "default": False,
        "description": "Show ranked model leaderboard and exit",
    },
    {
        "name": "export",
        "type": "string",
        "default": None,
        "choices": ["csv", "json"],
        "description": "Export all results in CSV or JSON format and exit",
    },
    {
        "name": "export_output",
        "type": "string",
        "default": None,
        "description": "Output file for --export (default: stdout)",
    },
    {
        "name": "resume",
        "type": "string",
        "default": None,
        "description": "Resume an interrupted run, preserving completed outcomes",
    },
    # -- Scoring --
    {
        "name": "weight_by_difficulty",
        "type": "bool",
        "default": False,
        "description": "Weight scenario scores by difficulty tier (1×trivial … 5×very hard)",
    },
]


def get_schema() -> dict[str, Any]:
    """Return the full args schema with version metadata."""
    return {
        "schema_version": SCHEMA_VERSION,
        "args": ARGS_SCHEMA,
        "commands": COMMANDS_SCHEMA,
    }
