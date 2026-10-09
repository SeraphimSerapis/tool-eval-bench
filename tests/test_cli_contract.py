"""The CLI's documented contract: exit codes, --json streams, and mode routing.

Driven through ``cli.dispatch.main()`` with the dispatch golden harness, so each
test exercises the path a user's invocation takes. ``docs/cli-reference.md`` is
the contract these pin.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from rich.console import Console

from tests import test_cli_dispatch_golden as _golden
from tests.test_cli_dispatch_golden import BASE_URL, CONNECTION, Cli
from tool_eval_bench.cli import dispatch
from tool_eval_bench.runner.throughput import ThroughputSample
from tool_eval_bench.utils.urls import validate_http_url

cli = _golden.cli

SECRET_URL = "http://user:hunter2@gpu.test:8000/v1"


def _events(err: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in err.splitlines() if line.strip()]


def _fake_benchy(cli: Cli) -> list[dict[str, Any]]:
    return cli.record(
        "tool_eval_bench.cli.perf",
        "run_llama_benchy",
        lambda: [ThroughputSample(pp_tokens=512, tg_tokens=128, pp_tps=5000.0, tg_tps=95.0)],
    )


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)


# --reference-date ------------------------------------------------------------


@pytest.mark.parametrize("value", ["2026-13-01", "2026-02-29", "tomorrow", "2026/03/20", ""])
def test_invalid_reference_date_is_an_argument_error_before_any_work(cli: Cli, value: str) -> None:
    benchy = _fake_benchy(cli)

    outcome = cli.run(*CONNECTION, "--reference-date", value, "--perf", "--json")

    if value == "":
        # An empty value means "not set", as it always has.
        assert outcome.code == 0
        return
    assert outcome.code == 2
    (event,) = _events(outcome.err)
    assert event["error"] == "invalid_arguments"
    assert f"Invalid --reference-date '{value}'" in event["message"]
    assert not benchy and cli.runs == [] and cli.contexts == []


def test_valid_reference_date_reaches_the_service_unchanged(cli: Cli) -> None:
    outcome = cli.run(
        *CONNECTION, "--scenarios", "TC-01", "--reference-date", "2028-02-29", "--json"
    )

    assert outcome.code == 0
    assert cli.runs[0]["reference_date"] == "2028-02-29"


# --resume ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        (None, "Run 'r-1' not found in history."),
        ({"run_id": "r-1", "status": "completed", "config": {}}, "run is already completed"),
    ],
)
def test_unresumable_target_is_rejected_before_the_perf_sweep(
    cli: Cli, stored: dict[str, Any] | None, message: str
) -> None:
    _golden._FakeRepository.stored = stored
    benchy = _fake_benchy(cli)

    outcome = cli.run(*CONNECTION, "--resume", "r-1", "--perf", "--json")

    assert outcome.code == 1
    assert not benchy, "the throughput sweep ran before the resume target was checked"
    assert cli.calls("preflight_model_check") == []
    (event,) = _events(outcome.err)
    assert event["error"] == "run_failed"
    assert message in event["message"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--perf-only"],
        ["--skip-tool-eval"],
        ["--gsm8k-only"],
        ["--spec-bench"],
        ["--context-pressure-sweep", "0.1-0.2"],
    ],
)
def test_resume_needs_a_scenario_run(cli: Cli, extra: list[str]) -> None:
    _golden._FakeRepository.stored = {"run_id": "r-1", "status": "interrupted", "config": {}}

    outcome = cli.run(*CONNECTION, "--resume", "r-1", *extra, "--json")

    assert outcome.code == 2
    (event,) = _events(outcome.err)
    assert event == {
        "event": "error",
        "error": "invalid_arguments",
        "message": "--resume requires a run of tool-call scenarios",
    }


# Mode conflicts ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (
            ["--spec-live", "--perf"],
            "--spec-live is a live monitor and cannot be combined with --perf",
        ),
        (
            ["--decision-live", "--gsm8k"],
            "--decision-live is a live monitor and cannot be combined with --gsm8k",
        ),
        (
            ["--spec-live", "--decision-live"],
            "--spec-live is a live monitor and cannot be combined with --decision-live",
        ),
        (
            ["--perf-only", "--gsm8k"],
            "--perf-only runs throughput alone and cannot be combined with --gsm8k",
        ),
        (["--perf-only", "--spec-bench"], "cannot be combined with --spec-bench; use --perf"),
        (
            ["--perf-only", "--context-pressure-sweep", "0.1-0.2"],
            "cannot be combined with --context-pressure-sweep",
        ),
        (
            ["--spec-bench", "--context-pressure-sweep", "0.1-0.2"],
            "--context-pressure-sweep cannot be combined with --spec-bench",
        ),
        (
            ["--context-pressure-sweep", "0.1-0.2", "--needle"],
            "--context-pressure-sweep cannot be combined with --needle",
        ),
        (
            ["--gsm8k-only", "--mmlu"],
            "--gsm8k-only runs one plugin alone and cannot be combined with --mmlu",
        ),
        (
            ["--gsm8k-only", "--mmlu-only"],
            "--gsm8k-only runs one plugin alone and cannot be combined with --mmlu-only",
        ),
        (
            ["--gsm8k", "--mmlu-only"],
            "--mmlu-only runs one plugin alone and cannot be combined with --gsm8k",
        ),
        (
            ["--perf", "--spec-bench", "--context-pressure-sweep", "0.1-0.2"],
            "--context-pressure-sweep cannot be combined with --spec-bench",
        ),
    ],
)
@pytest.mark.parametrize("json_mode", [False, True], ids=["console", "json"])
def test_a_mode_that_would_be_dropped_is_rejected(
    cli: Cli, flags: list[str], message: str, json_mode: bool
) -> None:
    benchy = _fake_benchy(cli)

    outcome = cli.run(*CONNECTION, *flags, *(["--json"] if json_mode else []))

    assert outcome.code == 2
    assert not benchy and cli.runs == []
    if json_mode and not any(flag.endswith("-live") for flag in flags):
        (event,) = _events(outcome.err)
        assert event["error"] == "invalid_arguments"
        assert message in event["message"]
    elif not json_mode:
        assert message in outcome.flat_err


@pytest.mark.parametrize(
    "flags",
    [
        ["--gsm8k", "--gsm8k-only"],
        ["--perf", "--gsm8k-only"],
        ["--perf", "--perf-only"],
        ["--perf", "--context-pressure-sweep", "0.1-0.2"],
        ["--gsm8k", "--mmlu", "--skip-tool-eval"],
        ["--spec-bench", "--gsm8k", "--skip-tool-eval"],
        # The sweep is its own run, so --skip-tool-eval has nothing to drop.
        ["--context-pressure-sweep", "0.1-0.2", "--skip-tool-eval"],
    ],
)
def test_combinations_where_every_mode_runs_are_not_rejected(cli: Cli, flags: list[str]) -> None:
    _fake_benchy(cli)
    cli.record("tool_eval_bench.cli.spec_bench", "run_spec_bench")
    cli.record("tool_eval_bench.cli.pressure", "run_pressure_sweep")

    outcome = cli.run(*CONNECTION, *flags, "--no-live")

    assert outcome.code == 0, outcome.flat_err


# --redact-url ------------------------------------------------------------------


def test_probe_redact_url_hides_the_host_when_ready(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "m"}]}))

    outcome = cli.run("probe", "--base-url", SECRET_URL, "--redact-url")

    assert outcome.code == 0
    assert "is ready" in outcome.out
    assert "gpu.test" not in outcome.out and "hunter2" not in outcome.out


def test_probe_redact_url_hides_the_host_when_not_ready(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused by {request.url}", request=request)

    _mock_http(monkeypatch, refuse)

    outcome = cli.run("probe", "--base-url", SECRET_URL, "--redact-url")

    assert outcome.code == 1
    assert "is not ready" in outcome.out
    assert "gpu.test" not in outcome.out and "hunter2" not in outcome.out


def test_probe_without_redact_url_still_shows_the_host(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_http(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "m"}]}))

    outcome = cli.run("probe", "--base-url", BASE_URL)

    assert outcome.code == 0
    assert "gpu.test" in outcome.out


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (lambda request: httpx.ConnectError("refused", request=request), 2),
        (lambda request: httpx.ReadError(f"reset by {request.url}", request=request), 3),
    ],
    ids=["connect", "unexpected"],
)
def test_preflight_failure_respects_the_display_url(
    monkeypatch: pytest.MonkeyPatch, error: Any, code: int
) -> None:
    from tool_eval_bench.cli.probe import preflight_model_check

    def handler(request: httpx.Request) -> httpx.Response:
        raise error(request)

    _mock_http(monkeypatch, handler)
    console = Console(record=True, width=200)

    with pytest.raises(SystemExit) as exc_info:
        preflight_model_check(console, SECRET_URL, "m", None, display_url="http://***:8000/v1")

    assert exc_info.value.code == code
    text = console.export_text()
    assert "gpu.test" not in text and "hunter2" not in text


_HTML_PAGE = httpx.Response(200, text="<html>Sign in</html>", headers={"Content-Type": "text/html"})


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_HTML_PAGE, id="html"),
        pytest.param(httpx.Response(200, json=["not", "an", "object"]), id="json-array"),
        pytest.param(httpx.Response(200, text=""), id="empty"),
    ],
)
def test_preflight_rejects_a_2xx_body_that_is_not_a_json_object(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], response: httpx.Response
) -> None:
    from tool_eval_bench.cli.probe import preflight_model_check

    _mock_http(monkeypatch, lambda request: response)

    with pytest.raises(SystemExit) as console_exit:
        preflight_model_check(
            Console(record=True, width=300), SECRET_URL, "m", None, display_url="http://***:8000"
        )
    with pytest.raises(SystemExit) as json_exit:
        preflight_model_check(Console(), SECRET_URL, "m", None, headless=True)

    assert console_exit.value.code == json_exit.value.code == 2
    (event,) = _events(capsys.readouterr().err)
    assert event["error"] == "invalid_response"
    assert "not a JSON object" in event["message"]


def test_preflight_invalid_response_console_message_respects_the_display_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.cli.probe import preflight_model_check

    _mock_http(
        monkeypatch,
        lambda request: httpx.Response(200, text=f"<a href='{SECRET_URL}'>login</a>"),
    )
    console = Console(record=True, width=300)

    with pytest.raises(SystemExit):
        preflight_model_check(console, SECRET_URL, "m", None, display_url="http://***:8000/v1")

    text = console.export_text()
    assert "Invalid response" in text
    assert "gpu.test" not in text and "hunter2" not in text


def test_preflight_accepts_a_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli.probe import preflight_model_check

    _mock_http(monkeypatch, lambda request: httpx.Response(200, json={"choices": []}))

    preflight_model_check(Console(), SECRET_URL, "m", None)


@pytest.mark.parametrize("redact", [True, False])
def test_warmup_failure_respects_the_display_url(
    monkeypatch: pytest.MonkeyPatch, redact: bool
) -> None:
    from tool_eval_bench.cli.probe import warmup_server

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused by {request.url}", request=request)

    _mock_http(monkeypatch, refuse)
    console = Console(record=True, width=300)

    warmup_server(
        console,
        SECRET_URL,
        "m",
        None,
        display_url="http://***:8000/v1" if redact else SECRET_URL,
    )

    text = console.export_text()
    assert "Warm-up failed" in text
    assert ("gpu.test" in text) is not redact


@pytest.mark.parametrize("url", ["http://user:hunter2@/v1", "https://user:hunter2@:8000"])
def test_hostless_url_error_does_not_echo_userinfo(url: str) -> None:
    with pytest.raises(ValueError, match="missing a host") as exc_info:
        validate_http_url(url, what="--base-url")
    assert "hunter2" not in str(exc_info.value)


# --json failure stream ---------------------------------------------------------


def test_failed_run_with_json_file_writes_the_envelope_and_emits_run_failed(cli: Cli) -> None:
    cli.service_error = RuntimeError(f"upstream {SECRET_URL} said no")

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--json-file", "r.json")

    assert outcome.code == 1
    envelope = json.loads((cli.tmp_path / "r.json").read_text(encoding="utf-8"))
    assert "hunter2" not in envelope["error"] and "upstream" in envelope["error"]
    events = _events(outcome.err)
    assert "benchmark_complete" not in [event["event"] for event in events]
    assert events[-1]["error"] == "run_failed"
    assert events[-1]["message"] == envelope["error"]


def test_unwritable_json_file_still_emits_run_failed(cli: Cli) -> None:
    cli.service_error = RuntimeError("boom")
    (cli.tmp_path / "taken").mkdir()

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--json-file", "taken")

    assert outcome.code == 1
    event = _events(outcome.err)[-1]
    assert event["error"] == "run_failed"
    assert event["message"].startswith("boom (and the --json-file could not be written")


def test_successful_json_file_run_still_reports_completion(cli: Cli) -> None:
    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--json-file", "r.json")

    assert outcome.code == 0
    assert _events(outcome.err)[-1]["event"] == "benchmark_complete"


# Ctrl+C before the scenarios ---------------------------------------------------


def _interrupt(*args: Any, **kwargs: Any) -> None:
    raise KeyboardInterrupt


def test_interrupt_during_setup_keeps_json_stderr_parseable(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch, "_resolve_endpoint", _interrupt)

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--json")

    assert outcome.code == 1
    assert outcome.out == ""
    assert _events(outcome.err) == [
        {"event": "error", "error": "run_failed", "message": "interrupted"}
    ]


def test_interrupt_during_setup_prints_no_traceback(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dispatch, "_resolve_endpoint", _interrupt)

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--no-live")

    assert outcome.code == 1
    assert "Interrupted." in outcome.out
    assert "Traceback" not in outcome.out + outcome.err


# Storage commands and --json ---------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["--history"], "--history"),
        (["--leaderboard"], "--leaderboard"),
        (["--compare", "a", "b"], "--compare"),
        (["history"], "--history"),
    ],
)
def test_table_commands_reject_json(cli: Cli, argv: list[str], flag: str) -> None:
    outcome = cli.run(*argv, "--json")

    assert outcome.code == 2
    assert outcome.out == ""
    (event,) = _events(outcome.err)
    assert event["error"] == "invalid_arguments"
    assert event["message"].startswith(f"{flag} does not support --json")


def test_export_json_is_still_allowed_with_json(cli: Cli) -> None:
    exports = cli.record("tool_eval_bench.cli.leaderboard", "export_runs")

    outcome = cli.run("--export", "json", "--json")

    assert outcome.code == 0
    assert len(exports) == 1


# --dry-run ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ["--perf-only"],
        ["--spec-bench"],
        ["--gsm8k-only"],
        ["--gsm8k", "--skip-tool-eval"],
        # Selection flags are inert when no scenarios run, so an empty
        # selection is not an error here.
        ["--perf-only", "--categories", "P"],
    ],
)
def test_dry_run_reports_no_scenarios_when_none_run(cli: Cli, flags: list[str]) -> None:
    outcome = cli.run("--dry-run", *flags, "--json")

    assert outcome.code == 0
    assert json.loads(outcome.out)["total_scenarios"] == 0


def test_dry_run_console_says_when_no_scenarios_run(cli: Cli) -> None:
    outcome = cli.run("--dry-run", "--perf-only")

    assert outcome.code == 0
    assert "0 scenarios would execute" in outcome.flat_out
    assert "This invocation runs no tool-call scenarios." in outcome.flat_out


def test_dry_run_still_lists_scenarios_for_a_scenario_run(cli: Cli) -> None:
    outcome = cli.run("--dry-run", "--perf", "--scenarios", "TC-01", "TC-02", "--json")

    assert outcome.code == 0
    assert json.loads(outcome.out)["total_scenarios"] == 2


def test_dry_run_still_rejects_an_empty_selection_for_a_scenario_run(cli: Cli) -> None:
    outcome = cli.run("--dry-run", "--categories", "P", "--json")

    assert outcome.code == 2
    assert _events(outcome.err)[0]["error"] == "invalid_arguments"


# --spec-bench routing ----------------------------------------------------------


def test_failed_spec_bench_before_a_plugin_lets_the_plugin_decide(cli: Cli) -> None:
    """--spec-bench --gsm8k --skip-tool-eval runs the plugin even after a failed cell."""
    from tool_eval_bench.runner import speculative
    from tool_eval_bench.runner.speculative import SpecDecodeSample

    samples = [SpecDecodeSample(prompt_type="filler", error="HTTP 503")]

    async def fake_run(*args: Any, on_sample: Any, **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    cli.monkeypatch.setattr(speculative, "run_spec_bench", fake_run)

    outcome = cli.run(
        *CONNECTION,
        "--spec-bench",
        "--gsm8k",
        "--skip-tool-eval",
        "--depth",
        "0",
        "--json",
        "--output-dir",
        str(cli.tmp_path),
    )

    assert outcome.code == 0
    assert len(cli.leaves["run_selected_plugins"]) == 1
    failures = [event for event in _events(outcome.err) if event.get("error") == "run_failed"]
    assert [event["message"] for event in failures] == [
        "Speculative decoding benchmark failed in 1 cell(s)."
    ]
    assert cli.runs == []
