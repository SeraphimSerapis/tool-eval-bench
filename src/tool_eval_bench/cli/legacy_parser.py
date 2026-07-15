"""Permanent flat-parser compatibility surface."""

from __future__ import annotations

import argparse

from tool_eval_bench.cli.command_registry import COMMAND_SPECS


def _make_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Extracted from ``main()`` so that tests and external tools can introspect
    the full argument list without calling ``parse_args()`` (which would consume
    sys.argv).
    """
    parser = argparse.ArgumentParser(
        prog="tool-eval-bench",
        description="Run tool-eval-bench agentic tool-call benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from tool_eval_bench import __version__

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show tool-eval-bench version and exit",
    )

    # -- Connection --------------------------------------------------------
    conn = parser.add_argument_group("connection")
    conn.add_argument("--model", default=None, help="Model name/path (auto-detected if omitted)")
    conn.add_argument(
        "--backend",
        default=None,
        help="Backend label for reports: vllm, litellm, llamacpp "
        "(all use the same OpenAI-compatible adapter; default: env/vllm)",
    )
    conn.add_argument(
        "--base-url",
        default=None,
        help="Server base URL (default: auto-discover on localhost, or from .env)",
    )
    conn.add_argument("--api-key", default=None, help="API key")
    conn.add_argument(
        "--probe",
        action="store_true",
        help="Check if a server is reachable and exit (exit 0 = ready, exit 1 = not found)",
    )

    # -- Sampling ----------------------------------------------------------
    sampling = parser.add_argument_group("sampling")
    sampling.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature (default: 0.0)"
    )
    sampling.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking/reasoning (sets enable_thinking=false)",
    )
    sampling.add_argument(
        "--top-p", type=float, default=None, metavar="P", help="Top-p (nucleus) sampling (e.g. 0.9)"
    )
    sampling.add_argument(
        "--top-k", type=int, default=None, metavar="K", help="Top-k sampling (e.g. 40)"
    )
    sampling.add_argument(
        "--min-p",
        type=float,
        default=None,
        metavar="P",
        help="Min-p sampling threshold (e.g. 0.05)",
    )
    sampling.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        metavar="V",
        help="Repetition penalty (e.g. 1.1)",
    )
    sampling.add_argument("--seed", type=int, default=None, help="Random seed (passed to server)")
    sampling.add_argument(
        "--backend-kwargs",
        type=str,
        default=None,
        metavar="JSON",
        help="JSON dict merged into API payload; overrides individual flags "
        '(e.g. \'{"temperature": 0.6, "top_p": 0.9}\')',
    )

    # -- Scenario selection ------------------------------------------------
    select = parser.add_argument_group("scenario selection")
    select.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Specific scenario IDs to run (e.g. TC-01 TC-07). Default: all.",
    )
    select.add_argument(
        "--categories",
        nargs="*",
        default=None,
        metavar="CAT",
        help="Run only specific categories (e.g. --categories K A J). "
        "Letters A–O map to the 15 benchmark categories.",
    )
    select.add_argument(
        "--short",
        action="store_true",
        help="Run only the core 15 scenarios (skip extended + agentic)",
    )
    select.add_argument(
        "--hardmode",
        action="store_true",
        help="Include Hard Mode scenarios (Category P) — ceiling-breaking difficulty "
        "for models that score 100%% on the standard benchmark",
    )
    select.add_argument(
        "--hardmode-only",
        action="store_true",
        help="Run ONLY Hard Mode scenarios (Category P) — shortcut for --hardmode --categories P",
    )

    # -- Run control -------------------------------------------------------
    run_ctrl = parser.add_argument_group("run control")
    run_ctrl.add_argument(
        "--timeout", type=float, default=60.0, help="Request timeout in seconds (default: 60)"
    )
    run_ctrl.add_argument(
        "--max-turns", type=int, default=8, help="Max turns per scenario (default: 8)"
    )
    run_ctrl.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trial runs for statistical rigor (default: 1)",
    )
    run_ctrl.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run N scenarios concurrently (default: 1 = sequential)",
    )
    run_ctrl.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        metavar="RATE",
        help="Inject random tool errors at this rate (0.0–1.0) for robustness testing",
    )
    run_ctrl.add_argument("--no-warmup", action="store_true", help="Skip server warm-up request")
    run_ctrl.add_argument(
        "--reference-date", default=None, help="Override benchmark reference date (YYYY-MM-DD)"
    )
    run_ctrl.add_argument(
        "--skip-tool-eval",
        action="store_true",
        help="Skip tool-call scenarios (use with --perf / --spec-bench)",
    )

    # -- Output ------------------------------------------------------------
    output = parser.add_argument_group("output")
    output.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of rich display"
    )
    output.add_argument(
        "--json-file",
        default=None,
        metavar="PATH",
        help="Write JSON results to PATH instead of stdout "
        "(implies --json; keeps stdout clean for logging)",
    )
    output.add_argument("--no-live", action="store_true", help="Disable live updating display")
    output.add_argument(
        "--redact-url",
        action="store_true",
        help="Mask the server URL in display output (for screenshots/recordings)",
    )
    output.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        metavar="W",
        help="Quality/speed weight for deployability score (0–1, default: 0.7)",
    )
    output.add_argument(
        "--no-probe-engine",
        action="store_true",
        help="Skip inference engine probing (no /version, /health HTTP calls)",
    )
    output.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for report files (default: ./runs/)",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which scenarios would run, then exit (no server needed)",
    )
    output.add_argument(
        "--fail-on-safety",
        action="store_true",
        help="Exit with status 2 when safety-critical scenarios fail",
    )

    # -- Throughput (llama-benchy) -----------------------------------------
    perf_grp = parser.add_argument_group("throughput benchmark (llama-benchy)")
    perf_grp.add_argument(
        "--perf", action="store_true", help="Run throughput benchmark before tool-call scenarios"
    )
    perf_grp.add_argument(
        "--perf-only",
        action="store_true",
        help="Run ONLY throughput benchmark (skip tool-call scenarios)",
    )
    perf_grp.add_argument(
        "--perf-legacy",
        action="store_true",
        help="Use built-in throughput benchmark (no external deps)",
    )
    perf_grp.add_argument(
        "--perf-legacy-only", action="store_true", help="Run ONLY built-in throughput benchmark"
    )
    perf_grp.add_argument("--pp", type=int, default=2048, help="Prompt tokens (default: 2048)")
    perf_grp.add_argument("--tg", type=int, default=128, help="Generation tokens (default: 128)")
    perf_grp.add_argument(
        "--depth",
        type=str,
        default="0,4096,8192",
        help="Context depths, comma separated (default: '0,4096,8192')",
    )
    perf_grp.add_argument(
        "--concurrency", type=str, default="1,2,4", help="Concurrency levels (default: '1,2,4')"
    )
    perf_grp.add_argument(
        "--benchy-runs", type=int, default=3, help="Measurement runs per test point (default: 3)"
    )
    perf_grp.add_argument(
        "--benchy-latency-mode",
        default="generation",
        choices=["api", "generation", "none"],
        help="Latency measurement mode (default: generation)",
    )
    perf_grp.add_argument(
        "--benchy-args",
        type=str,
        default=None,
        help="Pass-through args for llama-benchy (quoted string)",
    )
    perf_grp.add_argument(
        "--skip-coherence",
        action="store_true",
        help="Deprecated: llama-benchy coherence check is now always skipped (retained for compatibility)",
    )

    # -- GSM8K benchmark ----------------------------------------------------
    gsm8k_grp = parser.add_argument_group("GSM8K benchmark")
    gsm8k_grp.add_argument(
        "--gsm8k",
        action="store_true",
        help="Run GSM8K (Grade School Math) benchmark after tool-call scenarios",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-only",
        action="store_true",
        help="Run ONLY the GSM8K benchmark (skip tool-call scenarios)",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-shots",
        type=int,
        default=8,
        metavar="N",
        help="Number of few-shot CoT examples (0–8, default: 8)",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-limit",
        type=int,
        default=200,
        metavar="N",
        help="Max questions to evaluate (default: 200, 0 = all 1319)",
    )
    gsm8k_grp.add_argument(
        "--gsm8k-shuffle",
        action="store_true",
        help="Shuffle question order (uses --seed for reproducibility)",
    )

    # -- MMLU benchmark -----------------------------------------------------
    mmlu_grp = parser.add_argument_group("MMLU benchmark")
    mmlu_grp.add_argument(
        "--mmlu",
        action="store_true",
        help="Run MMLU (Massive Multitask Language Understanding) benchmark",
    )
    mmlu_grp.add_argument(
        "--mmlu-only",
        action="store_true",
        help="Run ONLY the MMLU benchmark (skip tool-call scenarios)",
    )
    mmlu_grp.add_argument(
        "--mmlu-shots",
        type=int,
        default=5,
        metavar="N",
        help="Number of few-shot examples per subject (0–5, default: 5)",
    )
    mmlu_grp.add_argument(
        "--mmlu-limit",
        type=int,
        default=500,
        metavar="N",
        help="Max questions to evaluate (default: 500, 0 = all 14042)",
    )
    mmlu_grp.add_argument(
        "--mmlu-subjects",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated subjects or categories (e.g. 'STEM,abstract_algebra')",
    )

    # -- IFEval benchmark ---------------------------------------------------
    ifeval_grp = parser.add_argument_group("IFEval benchmark")
    ifeval_grp.add_argument(
        "--ifeval", action="store_true", help="Run IFEval (Instruction Following) benchmark"
    )
    ifeval_grp.add_argument(
        "--ifeval-only",
        action="store_true",
        help="Run ONLY the IFEval benchmark (skip tool-call scenarios)",
    )
    ifeval_grp.add_argument(
        "--ifeval-limit",
        type=int,
        default=0,
        metavar="N",
        help="Max prompts to evaluate (default: 0 = all 541)",
    )

    # -- Speculative decoding benchmark ------------------------------------
    spec_grp = parser.add_argument_group("speculative decoding benchmark")
    spec_grp.add_argument(
        "--spec-bench",
        action="store_true",
        help="Run spec-decode / MTP benchmark (effective t/s, acceptance rate)",
    )
    spec_grp.add_argument(
        "--spec-live",
        action="store_true",
        help="Live-monitor speculative decoding stats (polls /metrics, runs until Ctrl+C)",
    )
    spec_grp.add_argument(
        "--spec-live-interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Poll interval for --spec-live in seconds (default: 1.0)",
    )
    spec_grp.add_argument(
        "--spec-method",
        default="auto",
        choices=["auto", "mtp", "draft", "dflash", "ngram", "eagle"],
        help="Spec-decode method hint (default: auto-detect)",
    )
    spec_grp.add_argument(
        "--baseline-tgs",
        type=float,
        default=None,
        metavar="TPS",
        help="Baseline tg t/s for speedup ratio calculation",
    )
    spec_grp.add_argument(
        "--spec-prompts",
        type=str,
        default="filler,code,structured",
        help="Prompt types, comma separated (default: 'filler,code,structured')",
    )
    spec_grp.add_argument(
        "--metrics-url",
        type=str,
        default=None,
        metavar="URL",
        help="Prometheus /metrics URL for acceptance rate (when API is behind a proxy)",
    )

    # -- Context pressure --------------------------------------------------
    pressure = parser.add_argument_group("context pressure")
    pressure.add_argument(
        "--context-pressure",
        type=float,
        default=None,
        metavar="RATIO",
        help="Fill context to RATIO (0.0–1.0) before each scenario",
    )
    pressure.add_argument(
        "--context-size",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Override auto-detected context window size (tokens)",
    )
    pressure.add_argument(
        "--context-pressure-sweep",
        type=str,
        default=None,
        metavar="START-END",
        help="Sweep pressure from START to END (e.g. 0.5-1.0)",
    )
    pressure.add_argument(
        "--sweep-steps",
        type=int,
        default=5,
        metavar="N",
        help="Number of pressure levels to test (default: 5)",
    )

    # -- History & comparison ----------------------------------------------
    hist_grp = parser.add_argument_group("history & comparison")
    hist_grp.add_argument(
        "--diff",
        metavar="RUN_ID",
        default=None,
        help="Compare against a previous run (use 'latest' for most recent)",
    )
    hist_grp.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        default=None,
        help="Compare two stored runs by ID",
    )
    hist_grp.add_argument(
        "--history", action="store_true", help="List recent benchmark runs and exit"
    )
    hist_grp.add_argument(
        "--leaderboard", action="store_true", help="Show ranked model leaderboard and exit"
    )
    hist_grp.add_argument(
        "--export",
        metavar="FORMAT",
        default=None,
        choices=["csv", "json"],
        help="Export all results in CSV or JSON format and exit",
    )
    hist_grp.add_argument(
        "--export-output",
        metavar="FILE",
        default=None,
        help="Output file for --export (default: stdout)",
    )
    hist_grp.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="Resume a previous run — skip scenarios that already passed",
    )

    # -- Scoring options ---------------------------------------------------
    scoring = parser.add_argument_group("scoring")
    scoring.add_argument(
        "--weight-by-difficulty",
        action="store_true",
        help="Weight scenario scores by difficulty tier (1×trivial … 5×very hard)",
    )

    # -- Subcommands -------------------------------------------------------
    subparsers = parser.add_subparsers(dest="command")
    compare_report = subparsers.add_parser(
        "compare-report",
        help="Generate a browser HTML comparison report from two Markdown reports",
        description="Generate a browser HTML comparison report from two Markdown reports.",
    )
    compare_report.add_argument("report_a", help="First Markdown report")
    compare_report.add_argument("report_b", help="Second Markdown report")
    compare_report.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output HTML path",
    )
    compare_report.add_argument(
        "--kind",
        choices=["auto", "summary", "tool-eval"],
        default="auto",
        help="Report type to compare (default: auto-detect from headings)",
    )
    for spec in COMMAND_SPECS:
        if spec.name != "compare-report":
            subparsers.add_parser(spec.name, help=spec.description, add_help=False)

    return parser


make_parser = _make_parser

__all__ = ["_make_parser", "make_parser"]
