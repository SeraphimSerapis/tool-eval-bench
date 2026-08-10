"""Tests for the subcommand-to-legacy compatibility parser."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

from tool_eval_bench.cli.bench import _make_parser
from tool_eval_bench.cli.parser import KNOWN_COMMANDS, parse_cli_args, translate_argv
from tool_eval_bench.cli.plugin_runners import _finalize_plugin_run, run_selected_plugins
from tool_eval_bench.schema import COMMANDS_SCHEMA, get_schema


@pytest.mark.parametrize(
    ("subcommand", "legacy"),
    [
        (["run", "--short", "--model", "m"], ["--short", "--model", "m"]),
        (["probe", "--base-url", "http://host"], ["--probe", "--base-url", "http://host"]),
        (["bench", "--perf-only"], ["--perf-only"]),
        (["spec-live", "--spec-live-interval", "2"], ["--spec-live", "--spec-live-interval", "2"]),
        (["history"], ["--history"]),
        (["leaderboard"], ["--leaderboard"]),
        (["resume", "run-123", "--json"], ["--resume", "run-123", "--json"]),
    ],
)
def test_simple_subcommands_translate_to_legacy(subcommand: list[str], legacy: list[str]) -> None:
    assert translate_argv(subcommand) == legacy


def test_legacy_arguments_are_unchanged() -> None:
    argv = ["--short", "--model", "m"]
    assert translate_argv(argv) == argv


@pytest.mark.parametrize(
    ("argv", "flag", "shots", "limit"),
    [
        (["plugin", "gsm8k", "--shots", "3", "--limit", "20"], "gsm8k_only", 3, 20),
        (["plugin", "mmlu", "--shots", "2", "--limit", "10"], "mmlu_only", 2, 10),
        (["plugin", "ifeval", "--limit", "5"], "ifeval_only", None, 5),
    ],
)
def test_plugin_subcommands_populate_legacy_namespace(
    argv: list[str], flag: str, shots: int | None, limit: int
) -> None:
    _, args = parse_cli_args(_make_parser, argv)
    assert getattr(args, flag) is True
    prefix = flag.removesuffix("_only")
    if shots is not None:
        assert getattr(args, f"{prefix}_shots") == shots
    assert getattr(args, f"{prefix}_limit") == limit


def test_plugin_translation_preserves_shared_legacy_options() -> None:
    _, args = parse_cli_args(
        _make_parser,
        ["plugin", "gsm8k", "--limit", "5", "--model", "m", "--json", "--parallel", "2"],
    )
    assert args.gsm8k_only is True
    assert args.gsm8k_limit == 5
    assert args.model == "m"
    assert args.json is True
    assert args.parallel == 2


def test_perf_tokenizer_flag_parses() -> None:
    """--tokenizer should flow through the legacy perf path."""
    _, args = parse_cli_args(
        _make_parser,
        ["bench", "--perf-only", "--tokenizer", "/models/tokenizer.json"],
    )
    assert args.perf_only is True
    assert args.tokenizer == "/models/tokenizer.json"


def test_compare_run_subcommand() -> None:
    _, args = parse_cli_args(_make_parser, ["compare", "run-a", "run-b"])
    assert args.compare == ["run-a", "run-b"]


def test_compare_report_subcommand() -> None:
    _, args = parse_cli_args(
        _make_parser,
        ["compare", "--report", "a.md", "b.md", "--output", "out.html", "--kind", "summary"],
    )
    assert args.command == "compare-report"
    assert args.report_a == "a.md"
    assert args.report_b == "b.md"
    assert args.output == "out.html"
    assert args.kind == "summary"


def test_compare_report_alias_is_unchanged() -> None:
    argv = ["compare-report", "a.md", "b.md", "-o", "out.html"]
    assert translate_argv(argv) == argv


def test_export_subcommand() -> None:
    _, args = parse_cli_args(_make_parser, ["export", "--format", "csv", "--output", "runs.csv"])
    assert args.export == "csv"
    assert args.export_output == "runs.csv"


def test_removed_noop_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(_make_parser, ["--llm-judge"])
    with pytest.raises(SystemExit):
        parse_cli_args(_make_parser, ["--experimental-async"])


def test_schema_v6_describes_every_command() -> None:
    schema = get_schema()
    assert schema["schema_version"] == "6"
    assert schema["commands"] is COMMANDS_SCHEMA
    assert set(COMMANDS_SCHEMA) == KNOWN_COMMANDS


def test_plugin_lifecycle_runs_selected_plugins_in_stable_order() -> None:
    _, args = parse_cli_args(_make_parser, ["--gsm8k", "--ifeval"])
    calls: list[str] = []

    def runner(name: str):
        def run(*_args, **_kwargs) -> None:
            calls.append(name)

        return run

    stopped = run_selected_plugins(
        Console(),
        "model",
        "display",
        "http://server",
        None,
        args,
        runners={name: runner(name) for name in ("gsm8k", "mmlu", "ifeval")},
        extra_params=None,
        output_dir=None,
        run_context=None,
    )
    assert calls == ["gsm8k", "ifeval"]
    assert stopped is False


def test_plugin_only_lifecycle_stops_before_tool_scenarios() -> None:
    _, args = parse_cli_args(_make_parser, ["plugin", "mmlu", "--limit", "1"])
    calls: list[str] = []

    def runner(*_args, **_kwargs) -> None:
        calls.append("mmlu")

    stopped = run_selected_plugins(
        Console(),
        "model",
        "display",
        "http://server",
        None,
        args,
        runners={"gsm8k": runner, "mmlu": runner, "ifeval": runner},
        extra_params=None,
        output_dir=None,
        run_context=None,
    )
    assert calls == ["mmlu"]
    assert stopped is True


@pytest.mark.parametrize(
    ("command", "included", "excluded"),
    [
        ("run", "--short", "--perf-only"),
        ("probe", "--base-url", "--short"),
        ("bench", "--perf-only", "--dry-run"),
        ("spec-live", "--metrics-url", "--short"),
        ("plugin", "--shots", "--perf-only"),
        ("compare", "--report", "--base-url"),
        ("history", "List recent runs", "--base-url"),
        ("leaderboard", "model leaderboard", "--base-url"),
        ("export", "--format", "--base-url"),
        ("resume", "run_id", "--perf-only"),
    ],
)
def test_subprocess_help_is_command_specific(command: str, included: str, excluded: str) -> None:
    completed = subprocess.run(  # noqa: S603 - fixed local module invocation
        [sys.executable, "-m", "tool_eval_bench.cli.bench", command, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert included in completed.stdout
    assert excluded not in completed.stdout
    assert len(completed.stdout.splitlines()) < 180


def test_module_execution_version_path() -> None:
    completed = subprocess.run(  # noqa: S603 - trusted interpreter and static arguments
        [sys.executable, "-m", "tool_eval_bench.cli.bench", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.startswith("tool-eval-bench ")


def test_subprocess_run_and_legacy_dry_run_are_equivalent() -> None:
    base = [sys.executable, "-m", "tool_eval_bench.cli.bench"]
    legacy = subprocess.run(  # noqa: S603 - fixed local module invocation
        [*base, "--dry-run", "--short", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    command = subprocess.run(  # noqa: S603 - fixed local module invocation
        [*base, "run", "--dry-run", "--short", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(command.stdout) == json.loads(legacy.stdout)
    assert json.loads(command.stdout)["total_scenarios"] == 15


@pytest.mark.parametrize(
    ("mode", "title", "metrics"),
    [
        ("gsm8k", "GSM8K", ["- **Accuracy**: **75.0%**"]),
        ("mmlu", "MMLU", ["- **Accuracy**: **75.0%**"]),
        (
            "ifeval",
            "IFEval",
            ["- **Prompt Accuracy**: **75.0%**", "- **Instruction Accuracy**: **80.0%**"],
        ),
    ],
)
def test_shared_plugin_finalization_writes_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    mode: str,
    title: str,
    metrics: list[str],
) -> None:
    from tool_eval_bench.cli import plugin_runners

    persisted: list[dict] = []
    monkeypatch.setattr(plugin_runners, "_persist_plugin_run", persisted.append)
    monkeypatch.setattr(
        plugin_runners,
        "_with_config_fingerprint",
        lambda config: {**config, "config_fingerprint": "fingerprint"},
    )
    result = SimpleNamespace(
        score=75.0,
        rating="Good",
        details={"total": 4, "correct": 3},
    )

    run_id = _finalize_plugin_run(
        mode=mode,
        title=title,
        display_name="Display Model",
        result=result,
        config={"model": "model", "mode": mode},
        report_metrics=metrics,
        report_lines=["plugin-specific report"],
        output_dir=str(tmp_path),
        run_context=None,
    )

    reports = list(tmp_path.rglob(f"{run_id}.md"))
    assert len(reports) == 1
    text = reports[0].read_text()
    assert f"# {title} Benchmark — Display Model" in text
    assert all(metric in text for metric in metrics)
    assert "plugin-specific report" in text
    assert persisted[0]["run_type"] == mode
    assert persisted[0]["run_id"] == run_id
    assert persisted[0]["config"]["config_fingerprint"] == "fingerprint"
    assert persisted[0]["scores"]["accuracy"] == 75.0


# ---------------------------------------------------------------------------
# --label flag
# ---------------------------------------------------------------------------


def test_label_flag_parses_on_run_subcommand() -> None:
    _, args = parse_cli_args(_make_parser, ["run", "--label", "tonyd2wild tool hardening 646c55f"])
    assert args.label == "tonyd2wild tool hardening 646c55f"


def test_label_defaults_to_none() -> None:
    _, args = parse_cli_args(_make_parser, ["run"])
    assert args.label is None


def test_plugin_subcommand_preserves_label() -> None:
    _, args = parse_cli_args(_make_parser, ["plugin", "gsm8k", "--label", "my tag"])
    assert args.label == "my tag"


def test_finalize_plugin_run_renders_label_and_slugifies_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.domain.models import RunContext

    LABEL = "aiden-sparkrun aiden-3.75-sparkrun toolargs no proxy 229a985"
    SLUG = "aiden-sparkrun-aiden-3.75-sparkrun-toolargs-no-proxy-229a985"

    persisted: list[dict] = []
    monkeypatch.setattr(plugin_runners, "_persist_plugin_run", persisted.append)
    monkeypatch.setattr(
        plugin_runners,
        "_with_config_fingerprint",
        lambda config: {**config, "config_fingerprint": "fingerprint"},
    )
    result = SimpleNamespace(
        score=75.0,
        rating="Good",
        details={"total": 4, "correct": 3},
    )
    ctx = RunContext(
        tool_version="2.5.0",
        git_sha="abc123",
        hostname="h",
        platform_info="p",
        python_version="3.12",
        model="model",
        backend="vllm",
        base_url="http://***:8000",
        label=LABEL,
    )

    run_id = _finalize_plugin_run(
        mode="gsm8k",
        title="GSM8K",
        display_name="Display Model",
        result=result,
        config={"model": "model", "mode": "gsm8k"},
        report_metrics=["- **Accuracy**: **75.0%**"],
        report_lines=["plugin-specific report"],
        output_dir=str(tmp_path),
        run_context=ctx,
    )

    reports = list(tmp_path.rglob(f"{run_id}--{SLUG}.md"))
    assert len(reports) == 1
    text = reports[0].read_text()
    assert f"- **Label**: <code>{LABEL}</code>" in text
    assert persisted[0]["metadata"]["label"] == LABEL
