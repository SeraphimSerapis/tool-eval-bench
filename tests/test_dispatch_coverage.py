"""High-value deterministic coverage for reporting, performance CLI, and datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from tests.coverage_helpers import async_return
from tests.coverage_helpers import throughput_sample as _sample


def test_dispatch_legacy_parser_builds_every_argument_group() -> None:
    from tool_eval_bench.cli.dispatch import _make_parser

    parser = _make_parser()
    args = parser.parse_args(
        [
            "--model",
            "m",
            "--base-url",
            "http://test",
            "--categories",
            "A",
            "--perf",
            "--gsm8k",
            "--mmlu",
            "--ifeval",
            "--spec-bench",
            "--context-pressure",
            "0.5",
        ]
    )

    assert args.model == "m"
    assert args.categories == ["A"]
    assert args.perf and args.gsm8k and args.mmlu and args.ifeval and args.spec_bench


@pytest.mark.parametrize("json_mode", [False, True])
def test_dispatch_main_dry_run_without_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], json_mode: bool
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch

    argv = ["tool-eval-bench", "--dry-run", "--short"]
    if json_mode:
        argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)

    with pytest.raises(SystemExit) as exc:
        dispatch.main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert ("total_scenarios" if json_mode else "Dry run") in output


@pytest.mark.parametrize(
    ("argv", "target"),
    [
        (["--history"], "_print_history"),
        (["--leaderboard"], "_print_leaderboard"),
        (["--export", "json"], "_export_runs"),
        (["--compare", "a", "b"], "_compare_runs"),
    ],
)
def test_dispatch_main_routes_storage_commands(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], target: str
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch

    called: list[tuple] = []
    monkeypatch.setattr(sys, "argv", ["tool-eval-bench", *argv])
    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, target, lambda *args, **kwargs: called.append((args, kwargs)))

    dispatch.main()

    assert called


def _dispatch_args(**overrides: object):
    import argparse

    values = dict(
        trials=1,
        temperature=0.0,
        timeout=1.0,
        max_turns=2,
        reference_date=None,
        seed=1,
        parallel=1,
        error_rate=0.0,
        alpha=0.7,
        weight_by_difficulty=False,
        json_file=None,
        diff=None,
        output_dir=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dispatch_json_and_plain_execution_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.domain.scenarios import (
        Category,
        ScenarioDefinition,
        ScenarioEvaluation,
        ScenarioStatus,
    )

    scenario = ScenarioDefinition(
        id="TC-X",
        title="x",
        category=Category.A,
        user_message="x",
        description="x",
        handle_tool_call=lambda s, c: {},
        evaluate=lambda s: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )
    payload = {
        "run_id": "r",
        "scores": {
            "final_score": 100,
            "rating": "Great",
            "weighted_score": 99,
            "scenario_results": [
                {"scenario_id": "TC-X", "status": "pass", "points": 2, "summary": "ok"}
            ],
        },
    }

    class Service:
        async def run_benchmark(self, **kwargs):
            return dict(payload)

    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [scenario])
    emitted: list[dict] = []
    monkeypatch.setattr(
        dispatch, "_emit_json_output", lambda value, **kwargs: emitted.append(value)
    )
    args = _dispatch_args(trials=2)
    dispatch._run_json(Service(), "m", "vllm", "url", None, args)
    assert emitted[0]["trial_statistics"]["trials"] == 2

    console = Console(record=True)
    dispatch._run_plain(Service(), console, "m", "Display", "vllm", "url", None, _dispatch_args())
    assert "Weighted Score" in console.export_text()


def test_dispatch_json_safety_gate_returns_status_two(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.domain.scenarios import (
        Category,
        ScenarioDefinition,
        ScenarioEvaluation,
        ScenarioStatus,
    )

    scenario = ScenarioDefinition(
        id="TC-K",
        title="safety",
        category=Category.K,
        user_message="x",
        description="x",
        handle_tool_call=lambda s, c: {},
        evaluate=lambda s: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )

    class Service:
        async def run_benchmark(self, **kwargs):
            return {
                "run_id": "safety",
                "scores": {
                    "safety_warnings": ["TC-K warning"],
                    "scenario_results": [],
                },
            }

    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [scenario])
    monkeypatch.setattr(dispatch, "_emit_json_output", lambda *args, **kwargs: None)
    args = _dispatch_args(fail_on_safety=True)

    with pytest.raises(SystemExit) as exc_info:
        dispatch._run_json(Service(), "m", "vllm", "url", None, args)

    assert exc_info.value.code == 2


def test_dispatch_trial_summary_all_variance_branches() -> None:
    from tool_eval_bench.cli import dispatch

    agg = {
        "trials": 3,
        "final_score_mean": 80,
        "final_score_stddev": 2,
        "final_score_ci95": (78, 82),
        "final_score_median": 80,
        "total_points_mean": 90,
        "total_points_stddev": 1,
        "pass_at_k": 90,
        "pass_hat_k": 70,
        "reliability_gap": 20,
        "per_category": {"A": {"label": "Safety", "mean_percent": 80, "stddev_percent": 5}},
        "per_scenario": {"TC-X": {"stddev": 1, "mean": 1, "points": [0, 1, 2]}},
    }
    console = Console(record=True)
    dispatch._print_trials_summary(console, agg)
    dispatch._print_trials_summary(console, {})
    assert "unstable scenario" in console.export_text()


def test_dispatch_main_skip_and_perf_only_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.storage import reports
    from tool_eval_bench.utils import metadata

    async def context(**kwargs):
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--skip-tool-eval",
            "--no-warmup",
            "--no-think",
            "--top-p",
            "0.9",
            "--backend-kwargs",
            '{"chat_template_kwargs":{"x":1}}',
        ],
    )
    dispatch.main()

    monkeypatch.setattr(dispatch, "_run_llama_benchy", lambda *a, **k: [_sample()])
    monkeypatch.setattr(
        reports.MarkdownReporter, "write_throughput_report", lambda *a, **k: tmp_path / "p.md"
    )
    monkeypatch.setattr(dispatch, "_persist_plugin_run", lambda value: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--perf-only",
            "--no-warmup",
            "--output-dir",
            str(tmp_path),
        ],
    )
    dispatch.main()


def test_detect_model_single_fallback_and_headless_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from tool_eval_bench.cli import dispatch

    class Client:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            Client.calls += 1
            if Client.calls == 1:
                return httpx.Response(404, request=httpx.Request("GET", url))
            return httpx.Response(
                200,
                json={"data": [{"id": "a", "root": "root-a"}, {"id": "b"}]},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())
    assert dispatch._detect_model("http://x/v1", "key", Console(), headless=True) == ("a", "root-a")


def test_dispatch_live_and_plain_multitrial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.cli import dispatch
    from tool_eval_bench.domain.scenarios import (
        Category,
        ScenarioDefinition,
        ScenarioEvaluation,
        ScenarioResult,
        ScenarioStatus,
    )
    from tool_eval_bench.storage import reports

    scenario = ScenarioDefinition(
        id="TC-X",
        title="x",
        category=Category.A,
        user_message="x",
        description="x",
        handle_tool_call=lambda s, c: {},
        evaluate=lambda s: ScenarioEvaluation(ScenarioStatus.PASS, 2, "ok"),
    )
    sr = ScenarioResult("TC-X", ScenarioStatus.PASS, 2, "ok")
    payload = {
        "run_id": "r",
        "report_path": str(tmp_path / "run.md"),
        "scores": {"final_score": 100, "rating": "Great", "scenario_results": [sr.to_dict()]},
    }

    class Service:
        async def run_benchmark(self, **kwargs):
            return dict(payload)

    class Display:
        def __init__(self, *args, **kwargs):
            self.results = {"TC-X": sr}

        def start(self):
            pass

        def stop(self):
            pass

        async def on_scenario_start(self, *args):
            pass

        async def on_scenario_result(self, *args):
            pass

        def set_finished(self, *args, **kwargs):
            pass

    monkeypatch.setattr(dispatch, "BenchmarkDisplay", Display)
    monkeypatch.setattr(dispatch, "_resolve_scenarios", lambda args: [scenario])
    monkeypatch.setattr(dispatch, "_print_diff", lambda *args: None)
    monkeypatch.setattr(
        reports.MarkdownReporter,
        "write_summary_report",
        lambda *args, **kwargs: tmp_path / "summary.md",
    )
    args = _dispatch_args(trials=2, diff="latest", output_dir=str(tmp_path))
    console = Console(record=True)
    dispatch._run_with_live_display(Service(), console, "m", "Display", "vllm", "url", None, args)
    dispatch._run_plain(Service(), console, "m", "Display", "vllm", "url", None, args)
    assert "Summary report" in console.export_text()


def test_probe_server_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from tool_eval_bench.cli import dispatch

    class Client:
        fail = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            if self.fail:
                raise RuntimeError("down")
            return httpx.Response(
                200, json={"data": [{"id": "m"}]}, request=httpx.Request("GET", url)
            )

    client = Client()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    with pytest.raises(SystemExit) as ready:
        dispatch._probe_server(Console(), "url", "key", headless=True)
    assert ready.value.code == 0
    client.fail = True
    with pytest.raises(SystemExit) as failed:
        dispatch._probe_server(Console(), "url", None)
    assert failed.value.code == 1


def test_dispatch_main_context_pressure_and_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.runner import context_pressure
    from tool_eval_bench.storage import db
    from tool_eval_bench.utils import metadata

    class Pressure:
        ratio = 0.5
        fill_tokens = 100
        detected_context = 1000

        def summary(self):
            return "50%"

        def budget_breakdown(self, **kwargs):
            return {"remaining_headroom_tokens": 100}

    async def prepare(*args, **kwargs):
        return Pressure()

    async def calibrate(messages, *args, **kwargs):
        return messages, 100

    async def context(**kwargs):
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(context_pressure, "prepare_context_pressure", prepare)
    monkeypatch.setattr(
        context_pressure,
        "build_pressure_messages",
        lambda *a, **k: [{"role": "user", "content": "fill"}],
    )
    monkeypatch.setattr(context_pressure, "calibrate_pressure_messages", calibrate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--context-pressure",
            "0.5",
            "--context-size",
            "1000",
            "--skip-tool-eval",
            "--no-warmup",
        ],
    )
    dispatch.main()

    class Repo:
        def get(self, run_id):
            return {
                "config": {"model": "m", "backend": "vllm"},
                "scores": {
                    "scenario_results": [
                        {"scenario_id": "TC-01", "status": "pass", "raw_log": "trace"}
                    ]
                },
            }

        def close(self):
            pass

    monkeypatch.setattr(db, "RunRepository", Repo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--backend",
            "vllm",
            "--base-url",
            "url",
            "--resume",
            "r",
            "--scenarios",
            "TC-01",
            "--no-warmup",
        ],
    )
    dispatch.main()


@pytest.mark.parametrize("kind", ["empty", "invalid", "http"])
def test_detect_model_failure_responses(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    import httpx

    from tool_eval_bench.cli import dispatch

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            request = httpx.Request("GET", url)
            if kind == "http":
                return httpx.Response(500, request=request)
            if kind == "invalid":
                return httpx.Response(200, text="nope", request=request)
            return httpx.Response(200, json={"data": []}, request=request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())
    with pytest.raises(SystemExit):
        dispatch._detect_model("url", None, Console(), headless=True)


def test_preflight_and_warmup_user_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from tool_eval_bench.cli import probe
    from tool_eval_bench.runner import throughput

    class Client:
        mode = "ok"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            if self.mode == "connect":
                raise httpx.ConnectError("down")
            if self.mode == "error":
                raise RuntimeError("unexpected")
            status = 500 if self.mode == "http" else 200
            return httpx.Response(status, text="bad", request=httpx.Request("POST", url))

    client = Client()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    console = Console(record=True)
    probe.preflight_model_check(console, "url", "m", "key")
    for mode, code in (("http", 3), ("connect", 2), ("error", 3)):
        client.mode = mode
        with pytest.raises(SystemExit) as exc:
            probe.preflight_model_check(console, "url", "m", None)
        assert exc.value.code == code

    monkeypatch.setattr(throughput, "warmup", async_return(20_000))
    probe.warmup_server(console, "url", "m", None)
    monkeypatch.setattr(throughput, "warmup", async_return(10))
    probe.warmup_server(console, "url", "m", None)

    async def fail(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(throughput, "warmup", fail)
    probe.warmup_server(console, "url", "m", None)
    assert "Warm-up failed" in console.export_text()


def test_dispatch_legacy_spec_and_sweep_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.storage import reports
    from tool_eval_bench.utils import metadata

    async def context(**kwargs):
        return None

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_do_warmup", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(
        reports.MarkdownReporter, "write_throughput_report", lambda *a, **k: tmp_path / "r.md"
    )
    monkeypatch.setattr(dispatch, "_persist_plugin_run", lambda value: None)
    monkeypatch.setattr(dispatch, "_run_throughput", lambda *a, **k: [_sample()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--perf-legacy-only",
            "--no-warmup",
            "--output-dir",
            str(tmp_path),
        ],
    )
    dispatch.main()

    called: list[str] = []
    monkeypatch.setattr(dispatch, "_run_spec_bench", lambda *a, **k: called.append("spec"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["tool-eval-bench", "--model", "m", "--base-url", "url", "--spec-bench", "--no-warmup"],
    )
    dispatch.main()
    monkeypatch.setattr(dispatch, "_run_pressure_sweep", lambda *a, **k: called.append("sweep"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool-eval-bench",
            "--model",
            "m",
            "--base-url",
            "url",
            "--context-pressure-sweep",
            "0.5-0.8",
            "--no-warmup",
        ],
    )
    dispatch.main()
    assert called == ["spec", "sweep"]
