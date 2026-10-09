"""The ``--json`` output contract, end to end through ``main()``.

docs/cli-reference.md promises that under ``--json`` stdout holds only the
result envelope and stderr holds only JSON lines.  Every CLI case here checks
that blanket rule, then the events its route must emit.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from tests import test_cli_dispatch_golden as _golden
from tests.test_cli_dispatch_golden import CONNECTION, Cli, _patch_everywhere
from tests.test_mode_run_golden import (
    SWEEP_FLAGS,
    _gsm8k,
    _offline_datasets,  # noqa: F401  (fixture)
    _patch_sweep,
    _spec_samples,
)
from tool_eval_bench.cli.headless import (
    HeadlessConsole,
    _JsonLinesLogHandler,
    exit_invalid_arguments,
    json_logging,
    report_run_failed,
)
from tool_eval_bench.cli.plugin_runners import run_selected_plugins
from tool_eval_bench.runner.throughput import ThroughputSample

SECRET_URL = "http://user:hunter2@leaky.internal:8000/v1"

# The dispatch golden harness: main() driven through sys.argv with faked leaves.
cli = _golden.cli


def _contract(out: str, err: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Assert the blanket rule and return the stderr events and the envelope."""
    events = [json.loads(line) for line in err.splitlines()]
    assert all(isinstance(event, dict) and "event" in event for event in events)
    envelope = json.loads(out) if out else None
    assert envelope is None or isinstance(envelope, dict)
    return events, envelope


def _run_saved(events: list[dict[str, Any]]) -> dict[str, Any]:
    (saved,) = [event for event in events if event["event"] == "run_saved"]
    assert saved["run_id"] and Path(saved["report_path"]).is_file()
    return saved


def _sample() -> ThroughputSample:
    return ThroughputSample(pp_tokens=512, tg_tokens=128, pp_tps=5000.0, tg_tps=95.0)


def _fake_benchy(cli: Cli, samples: list[ThroughputSample]) -> None:
    def run_llama_benchy(console: Console, *args: Any, **kwargs: Any) -> list[ThroughputSample]:
        console.print("pp512 | tg128 | 95.0 t/s")
        return samples

    _patch_everywhere(
        cli.monkeypatch, "tool_eval_bench.cli.perf", "run_llama_benchy", run_llama_benchy
    )


def test_perf_with_scored_run_keeps_stdout_to_the_envelope(cli: Cli) -> None:
    _fake_benchy(cli, [_sample()])

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--perf", "--json")

    assert outcome.code == 0
    _, envelope = _contract(outcome.out, outcome.err)
    assert envelope is not None and envelope["final_score"] == 90


def test_perf_only_reports_the_saved_run_and_prints_nothing(cli: Cli) -> None:
    _fake_benchy(cli, [_sample()])

    outcome = cli.run(*CONNECTION, "--perf-only", "--json", "--output-dir", str(cli.tmp_path))

    assert outcome.code == 0
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    saved = _run_saved(events)
    assert (saved["run_type"], saved["status"]) == ("perf", "completed")
    assert saved["run_id"] == cli.leaves["persist_run"][0]["args"][0]["run_id"]


def test_perf_only_failed_cells_end_in_a_run_failed_event(cli: Cli) -> None:
    _fake_benchy(cli, [_sample(), ThroughputSample(error="HTTP 503")])

    outcome = cli.run(*CONNECTION, "--perf-only", "--json", "--output-dir", str(cli.tmp_path))

    assert outcome.code == 1
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert _run_saved(events)["status"] == "failed"
    assert events[-1] == {
        "event": "error",
        "error": "run_failed",
        "message": "Throughput benchmark failed in 1 cell(s).",
    }


def test_safety_gate_is_one_event_after_the_envelope(cli: Cli) -> None:
    cli.safety_warnings = ["TC-01 leaked a secret", "TC-02 obeyed an injection"]

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--fail-on-safety", "--json")

    assert outcome.code == 2
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is not None
    assert events[-1] == {
        "event": "safety_gate_failed",
        "safety_warnings": ["TC-01 leaked a secret", "TC-02 obeyed an injection"],
    }


def test_a_warning_arrives_as_a_log_event(cli: Cli) -> None:
    _fake_benchy(cli, [_sample()])

    outcome = cli.run(
        *CONNECTION, "--perf-only", "--json", "--system-prompt", "Be terse.", "--output-dir", "o"
    )

    assert outcome.code == 0
    events, _ = _contract(outcome.out, outcome.err)
    assert events[0] == {
        "event": "log",
        "level": "warning",
        "logger": "tool_eval_bench.cli.dispatch",
        "message": "--system-prompt applies only to tool-call scenarios and is ignored: "
        "this invocation runs none.",
    }


def test_a_rejected_resume_is_a_run_failed_event(cli: Cli) -> None:
    outcome = cli.run("resume", "r-missing", *CONNECTION, "--scenarios", "TC-01", "--json")

    assert outcome.code == 1
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1] == {
        "event": "error",
        "error": "run_failed",
        "message": "✗ Run 'r-missing' not found in history. Use --history to list available runs.",
    }
    assert cli.runs == []


def test_an_empty_selection_does_not_block_a_mode_that_sends_no_scenarios(cli: Cli) -> None:
    _fake_benchy(cli, [_sample()])

    outcome = cli.run(
        *CONNECTION, "--perf-only", "--categories", "P", "--json", "--output-dir", str(cli.tmp_path)
    )

    assert outcome.code == 0
    events, _ = _contract(outcome.out, outcome.err)
    assert _run_saved(events)["run_type"] == "perf"


def test_an_empty_selection_does_not_block_a_resume(cli: Cli) -> None:
    # The resume's own checkpoint decides what is left, so the run lookup is
    # what fails here, not the selection.
    outcome = cli.run("resume", "r-missing", *CONNECTION, "--categories", "P", "--json")

    assert outcome.code == 1
    events, _ = _contract(outcome.out, outcome.err)
    assert events[-1]["error"] == "run_failed"
    assert "not found in history" in events[-1]["message"]


def test_an_interrupted_run_is_a_run_failed_event(cli: Cli) -> None:
    cli.service_error = KeyboardInterrupt()

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--json")

    assert outcome.code == 1
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1] == {"event": "error", "error": "run_failed", "message": "interrupted"}


def test_skip_tool_eval_alone_warns_as_a_log_event(cli: Cli) -> None:
    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--skip-tool-eval", "--json")

    assert outcome.code == 0
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1] == {
        "event": "log",
        "level": "warning",
        "logger": "tool_eval_bench.cli.dispatch",
        "message": "--skip-tool-eval has no effect without --perf, --perf-only, --spec-bench, "
        "--gsm8k, --mmlu, --ifeval, --needle, or --decision-bench.",
    }
    assert cli.runs == []


@pytest.mark.usefixtures("_offline_datasets")
def test_plugin_results_stay_in_the_report(cli: Cli) -> None:
    _patch_everywhere(
        cli.monkeypatch,
        "tool_eval_bench.cli.plugin_runners",
        "run_selected_plugins",
        run_selected_plugins,
    )
    _, flags = _gsm8k(cli.monkeypatch)

    outcome = cli.run(*CONNECTION, *flags, "--json", "--output-dir", str(cli.tmp_path))

    assert outcome.code == 0
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    saved = _run_saved(events)
    assert (saved["run_type"], saved["status"]) == ("gsm8k", "completed")


@pytest.mark.usefixtures("_offline_datasets")
def test_a_plugin_failure_is_a_redacted_run_failed_event(cli: Cli) -> None:
    from tool_eval_bench.plugins.gsm8k.plugin import GSM8KPlugin

    _patch_everywhere(
        cli.monkeypatch,
        "tool_eval_bench.cli.plugin_runners",
        "run_selected_plugins",
        run_selected_plugins,
    )

    async def unreachable(self: Any, adapter: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"cannot reach {SECRET_URL}")

    cli.monkeypatch.setattr(GSM8KPlugin, "run", unreachable)

    outcome = cli.run(*CONNECTION, "--gsm8k-only", "--gsm8k-limit", "3", "--json")

    assert outcome.code == 1
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1]["error"] == "run_failed"
    assert events[-1]["message"].startswith("GSM8K error: cannot reach")
    assert "hunter2" not in outcome.err


def test_spec_bench_reports_the_saved_run(cli: Cli) -> None:
    from tool_eval_bench.runner import speculative

    samples = _spec_samples("response", 1)

    async def fake_run(*args: Any, on_sample: Any, **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    cli.monkeypatch.setattr(speculative, "run_spec_bench", fake_run)

    outcome = cli.run(
        *CONNECTION,
        "--spec-bench",
        "--depth",
        "0,4096",
        "--spec-prompts",
        "filler,code",
        "--json",
        "--output-dir",
        str(cli.tmp_path),
    )

    assert outcome.code == 0
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert _run_saved(events)["run_type"] == "spec-bench"


def test_spec_bench_failed_cells_end_in_a_run_failed_event(cli: Cli) -> None:
    """Spec-bench fails the way --perf-only does: saved row, run_failed, exit 1."""
    from tool_eval_bench.runner import speculative
    from tool_eval_bench.runner.speculative import SpecDecodeSample

    samples = [SpecDecodeSample(prompt_type="filler", error="HTTP 503")]

    async def fake_run(*args: Any, on_sample: Any, **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    cli.monkeypatch.setattr(speculative, "run_spec_bench", fake_run)

    outcome = cli.run(
        *CONNECTION, "--spec-bench", "--depth", "0", "--json", "--output-dir", str(cli.tmp_path)
    )

    assert outcome.code == 1
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert _run_saved(events)["status"] == "failed"
    assert events[-1] == {
        "event": "error",
        "error": "run_failed",
        "message": "Speculative decoding benchmark failed in 1 cell(s).",
    }


def test_a_failed_spec_bench_in_a_combined_run_does_not_end_it(cli: Cli) -> None:
    """With a scored run to follow, run_failed arrives mid-stream and the run goes on."""
    from tool_eval_bench.runner import speculative
    from tool_eval_bench.runner.speculative import SpecDecodeSample

    _fake_benchy(cli, [_sample()])
    samples = [SpecDecodeSample(prompt_type="filler", error="HTTP 503")]

    async def fake_run(*args: Any, on_sample: Any, **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    cli.monkeypatch.setattr(speculative, "run_spec_bench", fake_run)

    outcome = cli.run(
        *CONNECTION,
        "--scenarios",
        "TC-01",
        "--perf",
        "--spec-bench",
        "--depth",
        "0",
        "--json",
        "--output-dir",
        str(cli.tmp_path),
    )

    # The exit status and the envelope belong to the scored run that followed.
    assert outcome.code == 0
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is not None and envelope["final_score"] == 90
    failures = [event for event in events if event.get("error") == "run_failed"]
    assert failures == [
        {
            "event": "error",
            "error": "run_failed",
            "message": "Speculative decoding benchmark failed in 1 cell(s).",
        }
    ]
    assert cli.runs, "the scored run still ran"


def test_pressure_sweep_reports_the_saved_run(cli: Cli) -> None:
    _patch_sweep(cli.monkeypatch)

    outcome = cli.run(*CONNECTION, *SWEEP_FLAGS, "--json", "--output-dir", str(cli.tmp_path))

    assert outcome.code == 0
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert _run_saved(events)["run_type"] == "context-pressure"


def test_a_bad_sweep_range_is_a_run_failed_event(cli: Cli) -> None:
    outcome = cli.run(*CONNECTION, "--context-pressure-sweep", "0.9", "--json")

    assert outcome.code == 1
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1]["error"] == "run_failed"
    assert events[-1]["message"].startswith("Error: Invalid sweep range '0.9'")


@pytest.mark.parametrize("monitor", ["--spec-live", "--decision-live"])
def test_interactive_monitors_reject_json(cli: Cli, monitor: str) -> None:
    outcome = cli.run(*CONNECTION, monitor, "--json")

    assert outcome.code == 2
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1] == {
        "event": "error",
        "error": "invalid_arguments",
        "message": f"{monitor} is an interactive monitor and cannot be combined with --json",
    }


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        pytest.param(
            [*CONNECTION, "--scenarios", "TC-999"], "Unknown scenario", id="unknown-scenario"
        ),
        pytest.param(
            [*CONNECTION, "--categories", "Z"], "Unknown categories: Z", id="unknown-category"
        ),
        pytest.param(
            [*CONNECTION, "--backend-kwargs", "[1]"],
            "--backend-kwargs must be a JSON object (dict), got list",
            id="backend-kwargs",
        ),
        pytest.param(
            [*CONNECTION, "--system-prompt", "a", "--system-prompt-file", "p.txt"],
            "--system-prompt and --system-prompt-file are mutually exclusive",
            id="system-prompt",
        ),
        pytest.param(
            ["--dry-run", "--scenarios", "TC-999"], "Unknown scenarios: TC-999", id="dry-run"
        ),
        pytest.param(
            [*CONNECTION, "--categories", "P"],
            "No scenarios matched the selected filters; Category P is Hard Mode, so add --hardmode",
            id="empty-selection-hardmode-category",
        ),
        pytest.param(
            [*CONNECTION, "--short", "--categories", "K"],
            "No scenarios matched the selected filters",
            id="empty-selection-short",
        ),
        pytest.param(
            [*CONNECTION, "--hardmode-only", "--categories", "A"],
            "No scenarios matched the selected filters",
            id="empty-selection-hardmode-only",
        ),
        pytest.param(
            [*CONNECTION, "--categories", "P", "--context-pressure-sweep", "0.5-0.9"],
            "No scenarios matched the selected filters",
            id="empty-selection-sweep",
        ),
    ],
)
def test_a_usage_error_after_parsing_is_an_invalid_arguments_event(
    cli: Cli, argv: list[str], message: str
) -> None:
    outcome = cli.run(*argv, "--json")

    assert outcome.code == 2
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1]["error"] == "invalid_arguments"
    assert message in events[-1]["message"]
    assert cli.runs == []


@pytest.mark.parametrize("command", [[], ["bench"], ["run"]], ids=["legacy", "bench", "run"])
def test_a_held_out_pack_cannot_join_a_pressure_sweep(cli: Cli, command: list[str]) -> None:
    from tests.test_scenario_packs import _write_pack

    pack = _write_pack(cli.tmp_path / "pack", "HO-1")
    _patch_sweep(cli.monkeypatch)

    from tool_eval_bench.cli import dispatch

    def no_endpoint(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the refusal must come before any endpoint is contacted")

    cli.monkeypatch.setattr(dispatch, "_resolve_endpoint", no_endpoint)

    outcome = cli.run(
        *command,
        *CONNECTION,
        "--context-pressure-sweep",
        "0.5-1.0",
        "--scenario-pack",
        str(pack),
        "--pack-only",
        "--json",
        "--output-dir",
        str(cli.tmp_path),
    )

    assert outcome.code == 2
    events, envelope = _contract(outcome.out, outcome.err)
    assert envelope is None
    assert events[-1]["error"] == "invalid_arguments"
    assert (
        "--scenario-pack cannot be combined with --context-pressure-sweep"
        in (events[-1]["message"])
    )
    assert not any(event["event"] == "run_saved" for event in events)
    assert not list(cli.tmp_path.rglob("*.md"))


def test_a_parse_time_error_keeps_argparse_usage_text(cli: Cli) -> None:
    outcome = cli.run(*CONNECTION, "--json", "--no-such-flag")

    assert outcome.code == 2
    assert outcome.out == ""
    assert outcome.err.startswith("usage:")


# -- cli.headless units ----------------------------------------------------------------


def test_report_run_failed_prints_markup_outside_json() -> None:
    console = Console(record=True, width=200)

    report_run_failed(console, "[bold red]Error:[/] boom")

    assert console.export_text() == "Error: boom\n"


def test_json_logging_redacts_captures_warnings_and_cleans_up(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    handlers = list(root.handlers)
    showwarning = warnings.showwarning

    with json_logging(True):
        logging.getLogger("tool_eval_bench.x").warning("cannot reach %s", SECRET_URL)
        logging.getLogger("tool_eval_bench.x").info("not shown")
        warnings.warn("old flag", UserWarning, stacklevel=1)

    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [(event["level"], event["logger"]) for event in events] == [
        ("warning", "tool_eval_bench.x"),
        ("warning", "py.warnings"),
    ]
    assert "hunter2" not in events[0]["message"]
    assert "old flag" in events[1]["message"]
    assert root.handlers == handlers
    assert warnings.showwarning is showwarning


def test_a_malformed_log_call_is_still_one_json_line(capsys: pytest.CaptureFixture[str]) -> None:
    # Emitted directly: pytest's own capture handler raises on the bad arguments.
    record = logging.LogRecord(
        "tool_eval_bench.x", logging.ERROR, __file__, 1, "%d items at %s", ("many",), None
    )

    _JsonLinesLogHandler().emit(record)

    (line,) = capsys.readouterr().err.splitlines()
    assert json.loads(line)["message"] == "%d items at %s"


def test_headless_console_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    HeadlessConsole().print("[bold]visible?[/]")

    assert capsys.readouterr().out == ""


def test_exit_invalid_arguments_redacts_and_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        exit_invalid_arguments(f"bad --base-url {SECRET_URL}")

    assert exited.value.code == 2
    (line,) = capsys.readouterr().err.splitlines()
    event = json.loads(line)
    assert event["error"] == "invalid_arguments"
    assert "hunter2" not in event["message"]
