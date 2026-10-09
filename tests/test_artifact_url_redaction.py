"""A credentialed base URL never reaches a report under runs/ or a stored row.

Reports and SQLite rows are shareable artifacts, so every URL in them is
redacted whatever ``--redact-url`` says; that flag only governs the console.
Each mode runs with the credentialed URL as both ``base_url`` and
``display_url``, which is what dispatch passes without ``--redact-url``.  The
failing-endpoint cases drive a real adapter against a 503, because httpx's
``HTTPStatusError`` message quotes the full request URL and that message is
what ends up in traces and error fields.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from tests.conftest import open_repository
from tests.test_mode_run_golden import (
    SWEEP_FLAGS,
    _args,
    _context,
    _gsm8k,
    _offline_datasets,  # noqa: F401  (fixture)
    _patch_sweep,
    _spec_samples,
    _throughput_samples,
    persisted,  # noqa: F401  (fixture)
)
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.utils.urls import redact_url, redact_urls

SECRET_URL = "http://user:hunter2@leaky-host.internal:8000/v1?api_key=sk-leak"
SECRETS = ("hunter2", "sk-leak", "leaky-host")
REDACTED = "http://***:8000/v1"


def _assert_redacted(*texts: str) -> None:
    for text in texts:
        for secret in SECRETS:
            assert secret not in text


def _row_and_report(rows: list[dict[str, Any]]) -> tuple[str, str]:
    assert len(rows) == 1
    row = rows[0]
    return json.dumps(row), Path(row["report_path"]).read_text(encoding="utf-8")


def _real_context() -> RunContext:
    # collect_run_context always stores the redacted form.
    return dataclasses.replace(_context(), base_url=redact_url(SECRET_URL))


def _target(args: argparse.Namespace, run_context: RunContext | None) -> Any:
    from tool_eval_bench.cli.dispatch import _Target

    return _Target(
        args=args,
        parser=argparse.ArgumentParser(),
        console=Console(record=True, width=200),
        model="m",
        display_name="Leak Model",
        backend="vllm",
        base_url=SECRET_URL,
        display_url=SECRET_URL,
        api_key=None,
        wire_format="openai",
        extra_params={},
        run_context=run_context,
    )


def _failing_adapter(*_args: Any, **_kwargs: Any) -> Any:
    from tool_eval_bench.adapters.openai_compat import OpenAICompatibleAdapter

    adapter = OpenAICompatibleAdapter(max_retries=0)
    adapter._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="down"))
    )
    return adapter


def test_redact_urls_masks_every_url_in_text() -> None:
    text = (
        "503 for url 'http://user:hunter2@leaky-host.internal:8000/v1/chat/completions"
        "?api_key=sk-leak' and https://a.example/x."
    )

    redacted = redact_urls(text)

    assert redacted == "503 for url 'http://***:8000/v1/chat/completions' and https://***/x."


def test_redact_urls_leaves_text_without_urls_alone() -> None:
    assert redact_urls("Connection refused (user:pw@host)") == "Connection refused (user:pw@host)"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HTTP://user:pw@h.internal/v1 and HTTPS://h/x", "http://***/v1 and https://***/x"),
        ("http://a:1@h1/x,http://b:2@h2/y", "http://***/x,http://***/y"),
        ("(http://u:p@h/v1?k=x)", "(http://***/v1)"),
        ("see http://u:p@h/v1; then http://u:p@h/v2.", "see http://***/v1; then http://***/v2."),
        ("at `http://u:p@h:8000/v1`", "at `http://***:8000/v1`"),
        # A balanced bracket belongs to the URL, an unbalanced one to the prose.
        ("(http://u:p@[::1]:8000/v1)", "(http://***:8000/v1)"),
        ("http://h/wiki/A_(b)", "http://***/wiki/A_(b)"),
    ],
    ids=[
        "uppercase-scheme",
        "comma-joined",
        "parenthesised",
        "prose",
        "backticks",
        "ipv6",
        "parens",
    ],
)
def test_redact_urls_finds_the_url_boundary(text: str, expected: str) -> None:
    assert redact_urls(text) == expected


@pytest.mark.parametrize(
    "url",
    ["http://[::1/x", "http://user:hunter2@[::1/v1?api_key=sk-leak"],
    ids=["bare", "credentialed"],
)
def test_redaction_fails_closed_on_an_unparsable_authority(url: str) -> None:
    assert redact_url(url) == "http://***"
    assert redact_urls(f"boom at '{url}'") == "boom at 'http://***'"


def test_headless_error_events_never_carry_the_credentialed_url(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tool_eval_bench.cli.helpers import emit_headless_error

    with pytest.raises(SystemExit) as exited:
        emit_headless_error("CONNECTION_FAILED", f"Could not connect to {SECRET_URL}.", exit_code=2)

    assert exited.value.code == 2
    event = json.loads(capsys.readouterr().err)
    assert event["message"] == f"Could not connect to {REDACTED}."


@pytest.mark.parametrize("ready", [True, False], ids=["ready", "failed"])
def test_probe_result_events_never_carry_the_credentialed_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], ready: bool
) -> None:
    from tool_eval_bench.cli import model_probe

    def run(coro: Any) -> Any:
        coro.close()
        if ready:
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        request = httpx.Request("GET", f"{SECRET_URL}/models")
        return httpx.Response(401, request=request).raise_for_status()

    monkeypatch.setattr(model_probe.asyncio, "run", run)

    with pytest.raises(SystemExit) as exited:
        model_probe._probe_server(Console(record=True), SECRET_URL, None, headless=True)

    assert exited.value.code == (0 if ready else 1)
    err = capsys.readouterr().err
    event = json.loads(err)
    assert event["base_url"] == redact_url(SECRET_URL)
    if not ready:
        assert "401" in event["error"]
    _assert_redacted(err)


@pytest.mark.asyncio
async def test_adapter_status_errors_do_not_quote_the_request_url() -> None:
    adapter = _failing_adapter()

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await adapter.chat_completion(
            model="m", messages=[{"role": "user", "content": "hi"}], base_url=SECRET_URL
        )
    await adapter.aclose()

    assert raised.value.response.status_code == 503
    assert REDACTED in str(raised.value)
    _assert_redacted(str(raised.value))


@pytest.mark.asyncio
async def test_throughput_sample_errors_do_not_quote_the_request_url() -> None:
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient
    from tool_eval_bench.runner.throughput import _stream_one

    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="down"))
    async with HTTPMeasurementClient(
        base_url=SECRET_URL, timeout=5.0, transport=transport
    ) as client:
        sample = await _stream_one(
            client, SECRET_URL, "m", [{"role": "user", "content": "hi"}], 8, None, None
        )

    assert sample.error is not None
    assert REDACTED in sample.error
    _assert_redacted(sample.error)


@pytest.mark.asyncio
async def test_non_stream_completion_status_errors_do_not_quote_the_request_url() -> None:
    from tool_eval_bench.adapters.measurement import HTTPMeasurementClient

    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"e": "down"}))
    async with HTTPMeasurementClient(
        base_url=SECRET_URL, timeout=5.0, transport=transport
    ) as client:
        response = await client.completion({"model": "m"})

    # Everything but raise_for_status is the httpx response unchanged.
    assert response.status_code == 503
    assert response.json() == {"e": "down"}
    with pytest.raises(httpx.HTTPStatusError) as raised:
        response.raise_for_status()
    assert raised.value.response.status_code == 503
    assert REDACTED in str(raised.value)
    _assert_redacted(str(raised.value))


def test_report_writers_redact_the_server_url_they_are_given(tmp_path: Path) -> None:
    from tool_eval_bench.storage.reports import MarkdownReporter
    from tool_eval_bench.storage.reports.throughput import throughput_report

    path = MarkdownReporter(root=str(tmp_path)).write_pressure_sweep_report(
        run_id="r",
        model="m",
        backend="vllm",
        display_url=SECRET_URL,
        context_size=1024,
        level_results=[],
        breaking_point=None,
        first_degradation=None,
    )
    report = throughput_report(
        "m", [], label=None, backend="vllm", server=SECRET_URL, served_model="m", model_root=None
    )

    sweep = path.read_text(encoding="utf-8")
    assert f"- **Server**: {REDACTED}\n" in sweep
    assert f"- **Server**: {REDACTED}" in report.header
    _assert_redacted(sweep, "\n".join(report.header))


# -- modes -----------------------------------------------------------------------------


def _sweep(monkeypatch: pytest.MonkeyPatch, out: Path, ctx: RunContext | None) -> None:
    from tool_eval_bench.adapters import factory
    from tool_eval_bench.cli.dispatch import _run_pressure_sweep_mode
    from tool_eval_bench.runner import orchestrator

    real_run_all = orchestrator.run_all_scenarios
    _patch_sweep(monkeypatch)
    # Real scenarios against a failing endpoint, so each trace carries the adapter error.
    monkeypatch.setattr(orchestrator, "run_all_scenarios", real_run_all)
    monkeypatch.setattr(factory, "build_adapter", _failing_adapter)
    _run_pressure_sweep_mode(_target(_args(out, *SWEEP_FLAGS), ctx))


def _sweep_level_error(monkeypatch: pytest.MonkeyPatch, out: Path, ctx: RunContext | None) -> None:
    from tool_eval_bench.cli.dispatch import _run_pressure_sweep_mode
    from tool_eval_bench.runner import orchestrator

    _patch_sweep(monkeypatch)

    async def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"Server error '503' for url '{SECRET_URL}/chat/completions'")

    monkeypatch.setattr(orchestrator, "run_all_scenarios", explode)
    _run_pressure_sweep_mode(_target(_args(out, *SWEEP_FLAGS), ctx))


def _spec(monkeypatch: pytest.MonkeyPatch, out: Path, ctx: RunContext | None) -> None:
    from tool_eval_bench.cli.dispatch import _run_spec_bench_mode
    from tool_eval_bench.runner import speculative

    samples = _spec_samples("response", 1)

    async def fake_run(*args: Any, on_sample: Callable[..., Any], **kwargs: Any) -> list[Any]:
        for index, sample in enumerate(samples):
            await on_sample(sample, index, len(samples))
        return samples

    monkeypatch.setattr(speculative, "run_spec_bench", fake_run)
    args = _args(out, "--spec-bench", "--depth", "0,4096", "--spec-prompts", "filler,code")
    _run_spec_bench_mode(_target(args, ctx))


def _throughput(monkeypatch: pytest.MonkeyPatch, out: Path, ctx: RunContext | None) -> None:
    from tool_eval_bench.cli import dispatch

    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: _throughput_samples(False))
    dispatch._run_throughput_mode(_target(_args(out, "--perf-only"), ctx))


def _plugin(monkeypatch: pytest.MonkeyPatch, out: Path, ctx: RunContext | None) -> None:
    runner, flags = _gsm8k(monkeypatch)
    runner(
        Console(record=True, width=200),
        "m",
        "Leak Model",
        SECRET_URL,
        None,
        _args(out, *flags),
        extra_params=None,
        output_dir=str(out),
        run_context=ctx,
    )


def _plugin_failing_endpoint(
    monkeypatch: pytest.MonkeyPatch, out: Path, ctx: RunContext | None
) -> None:
    from tool_eval_bench.adapters import factory
    from tool_eval_bench.cli import plugin_runners
    from tool_eval_bench.plugins.gsm8k.dataset import GSM8KItem

    item = GSM8KItem(index=0, question="1 + 1?", raw_answer="#### 2", ground_truth=2.0)
    monkeypatch.setattr(plugin_runners, "load_dataset_with_progress", lambda *a, **k: [item])
    monkeypatch.setattr(factory, "build_adapter", _failing_adapter)
    plugin_runners._run_gsm8k_benchmark(
        Console(record=True, width=200),
        "m",
        "Leak Model",
        SECRET_URL,
        None,
        _args(out, "--gsm8k-only", "--gsm8k-limit", "1", "--gsm8k-shots", "0"),
        extra_params=None,
        output_dir=str(out),
        run_context=ctx,
    )


@pytest.mark.usefixtures("_offline_datasets")
@pytest.mark.parametrize("with_context", [True, False], ids=["context", "no-context"])
@pytest.mark.parametrize(
    ("mode", "evidence"),
    [
        (_sweep, ("503", REDACTED)),
        (_sweep_level_error, ("503", REDACTED)),
        (_spec, ()),
        (_throughput, ()),
        (_plugin, ()),
        # GSM8K keeps only the error count, not the message.
        (_plugin_failing_endpoint, ('"errors": 1',)),
    ],
    ids=[
        "sweep",
        "sweep-level-error",
        "spec-bench",
        "throughput",
        "plugin",
        "plugin-failing-endpoint",
    ],
)
def test_mode_artifacts_never_carry_the_credentialed_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persisted: list[dict[str, Any]],  # noqa: F811
    mode: Callable[[pytest.MonkeyPatch, Path, RunContext | None], None],
    evidence: tuple[str, ...],
    with_context: bool,
) -> None:
    mode(monkeypatch, tmp_path, _real_context() if with_context else None)

    row, report = _row_and_report(persisted)

    _assert_redacted(row, report)
    # The failure was recorded, just without the URL's secrets.
    for expected in evidence:
        assert expected in row


@pytest.mark.parametrize("mode", [_sweep, _throughput], ids=["sweep", "throughput"])
@pytest.mark.usefixtures("_offline_datasets")
def test_server_header_shows_the_redacted_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persisted: list[dict[str, Any]],  # noqa: F811
    mode: Callable[[pytest.MonkeyPatch, Path, RunContext | None], None],
) -> None:
    mode(monkeypatch, tmp_path, None)

    _, report = _row_and_report(persisted)

    assert f"- **Server**: {REDACTED}\n" in report


# -- scored run ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scored_run_artifacts_never_carry_the_credentialed_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.application.service import BenchmarkService
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS
    from tool_eval_bench.storage.reports import MarkdownReporter

    with open_repository(db_path=str(tmp_path / "bench.sqlite")) as repo:
        service = BenchmarkService(repo=repo, reporter=MarkdownReporter(root=str(tmp_path)))
        monkeypatch.setattr(service, "_adapter_for", _failing_adapter)

        run_data = await service.run_benchmark(
            model="m",
            backend="vllm",
            base_url=SECRET_URL,
            scenarios=[s for s in ALL_SCENARIOS if s.id == "TC-01"],
            run_context=_real_context(),
        )

        stored = repo.get(run_data["run_id"])

    row = json.dumps(stored)
    report = Path(run_data["report_path"]).read_text(encoding="utf-8")
    _assert_redacted(row, report)
    assert "503" in row
    assert REDACTED in row


# -- headless --json -------------------------------------------------------------------


def test_headless_json_error_envelope_never_carries_the_credentialed_url(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tool_eval_bench.cli import dispatch

    class Service:
        async def run_benchmark(self, **kwargs: Any) -> dict[str, Any]:
            # A status error raised outside the adapters, so nothing upstream redacted it.
            request = httpx.Request("POST", f"{SECRET_URL}/chat/completions")
            httpx.Response(503, request=request).raise_for_status()
            raise AssertionError("unreachable")

    json_file = tmp_path / "result.json"
    args = _args(tmp_path, "--json", "--json-file", str(json_file), "--scenarios", "TC-01")

    with pytest.raises(SystemExit) as exited:
        dispatch._run_json(Service(), "m", "vllm", SECRET_URL, None, args)  # type: ignore[arg-type]

    assert exited.value.code == 1
    envelope = json_file.read_text(encoding="utf-8")
    assert "503" in envelope
    assert REDACTED in envelope
    _assert_redacted(envelope, capsys.readouterr().out)
