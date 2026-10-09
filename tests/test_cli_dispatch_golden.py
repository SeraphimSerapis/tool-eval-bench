"""Characterisation tests for the routes ``cli.dispatch.main`` takes.

Each test drives ``main()`` through ``sys.argv`` the way a user does and fakes
only the leaf seams: the benchmark service, run-context collection, storage,
and the per-mode runners. What reaches those seams, the exit code, and the key
output lines are pinned as literals, so splitting ``main()`` cannot change a
route without failing here.

Fakes are installed with :func:`_patch_everywhere`, which replaces a leaf in
every loaded module that holds it. A test therefore keeps working when code
that calls the leaf moves to another module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import tool_eval_bench.cli.dispatch  # noqa: F401  (loads every alias the patcher must reach)
from tool_eval_bench.application.service import BenchmarkService
from tool_eval_bench.domain.models import RunContext
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
from tool_eval_bench.runner.throughput import ThroughputSample

BASE_URL = "http://gpu.test:8000/v1"
CONNECTION = ["--model", "m", "--backend", "vllm", "--base-url", BASE_URL]


def _patch_everywhere(
    monkeypatch: pytest.MonkeyPatch, module_name: str, attr: str, replacement: Any
) -> None:
    """Replace ``module_name.attr`` and every alias of it in loaded project modules."""
    original = getattr(importlib.import_module(module_name), attr)
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("tool_eval_bench"):
            continue
        for key, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, key, replacement)


def _callable_name(value: Any) -> str:
    return getattr(value, "__qualname__", type(value).__name__)


def _normalise(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Make recorded keyword arguments comparable as literals."""
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in {"scenarios", "resume_scenarios"} and value is not None:
            out[key] = [scenario.id for scenario in value]
        elif key == "resume_prior_results" and value is not None:
            out[key] = [result["scenario_id"] for result in value]
        elif key == "context_pressure_messages" and value is not None:
            out[key] = f"<{len(value)} messages>"
        elif key == "throughput_samples":
            out[key] = f"<{len(value)} samples>"
        elif isinstance(value, RunContext):
            out[key] = f"<RunContext {value.label}>"
        elif key == "decision_judge_api_key":
            out[key] = "<set>" if value else value
        elif callable(value):
            out[key] = _callable_name(value)
        else:
            out[key] = value
    return out


def _leaf_value(value: Any) -> Any:
    if isinstance(value, RunContext):
        return f"<RunContext {value.label}>"
    if isinstance(value, argparse.Namespace):
        return "<args>"
    if isinstance(value, Console):
        return "<console>"
    if isinstance(value, dict) and "X-Session-Id" in value:
        return {**value, "X-Session-Id": "<session>"}
    return value


def _leaf_call(call: dict[str, Any]) -> dict[str, Any]:
    """Make a recorded leaf call comparable as a literal.

    Injected helper callables (storage, fingerprint, and parsing hooks) are
    dropped: they are wiring, not routing, and the mode-run refactor removes
    them from these signatures.
    """
    return {
        key: tuple(_leaf_value(arg) for arg in value) if key == "args" else _leaf_value(value)
        for key, value in call.items()
        if key == "args" or not callable(value)
    }


class _FakeRepository:
    """Stands in for ``storage.db.RunRepository``; ``stored`` drives resume."""

    stored: dict[str, Any] | None = None
    checkpoints: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.stored

    def get_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        return list(self.checkpoints)

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeRepository:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class _FakeDisplay:
    instances: list[_FakeDisplay] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.results: dict[str, ScenarioResult] = {}
        self.finished = False
        _FakeDisplay.instances.append(self)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def on_scenario_start(self, scenario: Any, idx: int, total: int) -> None:
        pass

    async def on_scenario_result(
        self, scenario: Any, result: ScenarioResult, idx: int, total: int
    ) -> None:
        self.results[scenario.id] = result

    async def on_scenario_audit(self, *args: Any) -> None:
        pass

    def on_rate_limit(self, status: Any) -> None:
        pass

    def set_finished(self, *args: Any, **kwargs: Any) -> None:
        self.finished = True


@dataclass
class Outcome:
    code: int | str | None
    out: str
    err: str

    @property
    def flat_out(self) -> str:
        return " ".join(self.out.split())

    @property
    def flat_err(self) -> str:
        return " ".join(self.err.split())


@dataclass
class Cli:
    monkeypatch: pytest.MonkeyPatch
    capsys: pytest.CaptureFixture[str]
    tmp_path: Path
    runs: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    leaves: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    safety_warnings: list[str] = field(default_factory=list)
    service_error: Exception | None = None

    def record(
        self, module_name: str, attr: str, returns: Any = None, *, is_async: bool = False
    ) -> list[dict[str, Any]]:
        """Fake a leaf callable and return the list its calls are recorded into."""
        calls = self.leaves.setdefault(attr, [])

        def sync_fake(*args: Any, **kwargs: Any) -> Any:
            calls.append({"args": args, **kwargs})
            return returns() if callable(returns) else returns

        async def async_fake(*args: Any, **kwargs: Any) -> Any:
            return sync_fake(*args, **kwargs)

        _patch_everywhere(
            self.monkeypatch, module_name, attr, async_fake if is_async else sync_fake
        )
        return calls

    def run(self, *argv: str) -> Outcome:
        self.monkeypatch.setattr(sys, "argv", ["tool-eval-bench", *argv])
        code: int | str | None = 0
        try:
            tool_eval_bench.cli.dispatch.main()
        except SystemExit as exc:
            code = exc.code
        captured = self.capsys.readouterr()
        return Outcome(code, captured.out, captured.err)

    @property
    def run_kwargs(self) -> list[dict[str, Any]]:
        return [_normalise(kwargs) for kwargs in self.runs]

    def calls(self, attr: str) -> list[dict[str, Any]]:
        return [_leaf_call(call) for call in self.leaves.get(attr, [])]


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> Cli:
    for key in list(os.environ):
        if key.startswith("TOOL_EVAL_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    harness = Cli(monkeypatch, capsys, tmp_path)
    passed = ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "ok")
    harness.result = {
        "run_id": "run-1",
        "report_path": str(tmp_path / "run.md"),
        "scores": {
            "final_score": 90,
            "rating": "Good",
            "scenario_results": [passed.to_dict()],
        },
    }

    async def run_benchmark(self: BenchmarkService, **kwargs: Any) -> dict[str, Any]:
        harness.runs.append(kwargs)
        scenarios = kwargs["scenarios"]
        for idx, scenario in enumerate(scenarios):
            if kwargs.get("on_scenario_start"):
                await kwargs["on_scenario_start"](scenario, idx, len(scenarios))
            result = ScenarioResult(scenario.id, ScenarioStatus.PASS, 2, "ok")
            if kwargs.get("on_scenario_result"):
                await kwargs["on_scenario_result"](scenario, result, idx, len(scenarios))
        if harness.service_error is not None:
            raise harness.service_error
        payload = json.loads(json.dumps(harness.result))
        if harness.safety_warnings:
            payload["scores"]["safety_warnings"] = list(harness.safety_warnings)
        return payload

    async def collect_run_context(**kwargs: Any) -> RunContext:
        harness.contexts.append(kwargs)
        return RunContext(
            tool_version="t",
            git_sha=None,
            hostname="h",
            platform_info="p",
            python_version="3",
            model=kwargs["model"],
            backend=kwargs["backend"],
            base_url=kwargs["base_url"],
            label=kwargs.get("label") or "ctx",
        )

    _FakeDisplay.instances = []
    _FakeRepository.stored = None
    _FakeRepository.checkpoints = []
    monkeypatch.setattr(BenchmarkService, "run_benchmark", run_benchmark)
    _patch_everywhere(monkeypatch, "tool_eval_bench.cli.helpers", "load_dotenv_file", lambda: None)
    _patch_everywhere(
        monkeypatch, "tool_eval_bench.utils.metadata", "collect_run_context", collect_run_context
    )
    _patch_everywhere(monkeypatch, "tool_eval_bench.cli.display", "BenchmarkDisplay", _FakeDisplay)
    _patch_everywhere(monkeypatch, "tool_eval_bench.storage.db", "RunRepository", _FakeRepository)
    harness.record("tool_eval_bench.cli.probe", "preflight_model_check")
    harness.record("tool_eval_bench.cli.probe", "warmup_server")
    harness.record("tool_eval_bench.application.run_queries", "persist_run")
    harness.record("tool_eval_bench.cli.plugin_runners", "run_selected_plugins", False)
    return harness


# ---------------------------------------------------------------------------
# Scored run: the three service.run_benchmark call sites
# ---------------------------------------------------------------------------

FULL_FLAGS = [
    *CONNECTION,
    "--api-key",
    "k-123",
    "--scenarios",
    "TC-01",
    "TC-02",
    "--temperature",
    "0.3",
    "--timeout",
    "45",
    "--max-turns",
    "5",
    "--seed",
    "7",
    "--reference-date",
    "2026-01-02",
    "--parallel",
    "2",
    "--error-rate",
    "0.1",
    "--alpha",
    "0.1",
    "--no-think",
    "--top-p",
    "0.9",
    "--top-k",
    "40",
    "--min-p",
    "0.05",
    "--repeat-penalty",
    "1.1",
    "--backend-kwargs",
    '{"chat_template_kwargs": {"x": 1}, "foo": 2}',
    "--system-prompt",
    "  Be terse.  ",
    "--variant-seed",
    "3",
    "--weight-by-difficulty",
    "--header",
    "X-Team=bench",
    "--session-header",
    "X-Session-Id",
    "--label",
    "golden",
]

# Every kwarg the scored run sends, shared by all three call sites.
FULL_RUN_KWARGS: dict[str, Any] = {
    "model": "m",
    "backend": "vllm",
    "base_url": BASE_URL,
    "api_key": "k-123",
    "scenarios": ["TC-01", "TC-02"],
    "temperature": 0.3,
    "timeout_seconds": 45.0,
    "max_turns": 5,
    "reference_date": "2026-01-02",
    "system_prompt": "  Be terse.",
    "variant_seed": 3,
    "seed": 7,
    "concurrency": 2,
    "error_rate": 0.1,
    "alpha": 0.1,
    "extra_params": {
        "chat_template_kwargs": {"enable_thinking": False, "x": 1},
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "foo": 2,
    },
    "context_pressure_messages": None,
    "context_pressure_config": None,
    "run_context": "<RunContext golden>",
    "weight_by_difficulty": True,
    "resume_run_id": None,
    "resume_prior_results": None,
    "resume_scenarios": None,
    "scenario_packs": None,
    "wire_format": "openai",
    "extra_headers": {"X-Team": "bench"},
    "session_header": "X-Session-Id",
}

LIVE_CALLBACKS = {
    "on_scenario_start": "_FakeDisplay.on_scenario_start",
    "on_scenario_result": "_FakeDisplay.on_scenario_result",
    "rate_limit_observer": "_FakeDisplay.on_rate_limit",
}
PLAIN_CALLBACKS = {
    "on_scenario_start": "_plain_on_start",
    "on_scenario_result": "_plain_on_result",
}
JSON_CALLBACKS = {
    "on_scenario_start": "stderr_progress_start",
    "on_scenario_result": "stderr_progress_result",
    "on_scenario_audit": "stderr_progress_audit",
}


@pytest.mark.parametrize(
    ("mode_flags", "expected"),
    [
        pytest.param(
            [],
            {**FULL_RUN_KWARGS, "throughput_samples": "<0 samples>", **LIVE_CALLBACKS},
            id="live",
        ),
        pytest.param(
            ["--no-live"],
            {**FULL_RUN_KWARGS, "throughput_samples": "<0 samples>", **PLAIN_CALLBACKS},
            id="plain",
        ),
        pytest.param(
            ["--json"],
            {**FULL_RUN_KWARGS, "throughput_samples": "<0 samples>", **JSON_CALLBACKS},
            id="json",
        ),
    ],
)
def test_scored_run_kwargs_per_call_site(
    cli: Cli, mode_flags: list[str], expected: dict[str, Any]
) -> None:
    outcome = cli.run(*FULL_FLAGS, *mode_flags)

    assert outcome.code == 0
    assert cli.run_kwargs == [expected]


def test_scored_run_preflight_warmup_and_context_inputs(cli: Cli) -> None:
    cli.run(*FULL_FLAGS, "--no-live")

    extra = FULL_RUN_KWARGS["extra_params"]
    headers = {"X-Team": "bench", "X-Session-Id": "<session>"}
    assert cli.calls("preflight_model_check") == [
        {
            "args": ("<console>", BASE_URL, "m", "k-123"),
            "headless": False,
            "wire_format": "openai",
            "timeout_seconds": 45.0,
            "temperature": 0.3,
            "extra_params": extra,
            "headers": headers,
        }
    ]
    assert cli.calls("warmup_server") == [
        {
            "args": ("<console>", BASE_URL, "m", "k-123"),
            "wire_format": "openai",
            "temperature": 0.3,
            "extra_params": extra,
            "headers": headers,
        }
    ]
    assert cli.contexts == [
        {
            "model": "m",
            "backend": "vllm",
            "base_url": BASE_URL,
            "api_key": "k-123",
            "temperature": 0.3,
            "max_turns": 5,
            "timeout_seconds": 45.0,
            "seed": 7,
            "scenario_selector": "TC-01, TC-02",
            "trials": 1,
            "parallel": 2,
            "error_rate": 0.1,
            "thinking_enabled": False,
            "extra_params": extra,
            "context_pressure": None,
            "system_prompt": "  Be terse.",
            "label": "golden",
            "probe_engine": True,
        }
    ]


DEFAULT_RUN_KWARGS: dict[str, Any] = {
    "model": "m",
    "backend": "vllm",
    "base_url": BASE_URL,
    "api_key": None,
    "scenarios": ["TC-01"],
    "temperature": 0.0,
    "timeout_seconds": 120.0,
    "max_turns": 8,
    "reference_date": None,
    "system_prompt": None,
    "variant_seed": None,
    "seed": None,
    "concurrency": 1,
    "error_rate": 0.0,
    "alpha": 0.7,
    "extra_params": None,
    "context_pressure_messages": None,
    "context_pressure_config": None,
    "run_context": "<RunContext ctx>",
    "weight_by_difficulty": False,
    "resume_run_id": None,
    "resume_prior_results": None,
    "resume_scenarios": None,
    "scenario_packs": None,
    "wire_format": "openai",
    "extra_headers": {},
    "session_header": None,
}


def test_scored_run_defaults(cli: Cli) -> None:
    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--no-live")

    assert outcome.code == 0
    assert cli.run_kwargs == [
        {**DEFAULT_RUN_KWARGS, "throughput_samples": "<0 samples>", **PLAIN_CALLBACKS}
    ]
    assert "Score: 90 / 100 — Good" in outcome.flat_out


@pytest.mark.parametrize(
    ("mode_flags", "samples"),
    [
        # Failed cells included: the service counts them in the stored row.
        pytest.param([], "<2 samples>", id="live"),
        pytest.param(["--no-live"], "<2 samples>", id="plain"),
        pytest.param(["--json"], "<2 samples>", id="json"),
    ],
)
def test_perf_samples_reach_the_scored_run(cli: Cli, mode_flags: list[str], samples: str) -> None:
    cli.record(
        "tool_eval_bench.cli.perf",
        "run_llama_benchy",
        lambda: [ThroughputSample(), ThroughputSample(error="boom")],
    )

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--perf", *mode_flags)

    assert outcome.code == 0
    assert [run.get("throughput_samples") for run in cli.run_kwargs] == [samples]
    assert cli.calls("run_llama_benchy") == [
        {
            "args": ("<console>", "m", "m", BASE_URL, None),
            "pp": [2048],
            "tg": [128],
            "depths": [0, 4096, 8192],
            "concurrency_levels": [1, 2, 4],
            "runs": 3,
            "latency_mode": "generation",
            "skip_coherence": True,
            "extra_args": None,
            "skip_warmup": True,
            "tokenizer": None,
            "backend": "vllm",
        }
    ]


@pytest.mark.parametrize(
    ("mode_flags", "show"),
    [
        pytest.param([], [LIVE_CALLBACKS, {}], id="live"),
        pytest.param(["--no-live"], [PLAIN_CALLBACKS, PLAIN_CALLBACKS], id="plain"),
        pytest.param(["--json"], [JSON_CALLBACKS, JSON_CALLBACKS], id="json"),
    ],
)
def test_trials_repeat_the_scored_run(
    cli: Cli, mode_flags: list[str], show: list[dict[str, str]]
) -> None:
    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--trials", "2", *mode_flags)

    assert outcome.code == 0
    callbacks = set(LIVE_CALLBACKS) | set(JSON_CALLBACKS)
    assert [
        {key: value for key, value in run.items() if key in callbacks} for run in cli.run_kwargs
    ] == show


@pytest.mark.parametrize(
    ("mode_flags", "audit"),
    [
        pytest.param([], "_FakeDisplay.on_scenario_audit", id="live"),
        pytest.param(["--no-live"], "_plain_on_audit", id="plain"),
        pytest.param(["--json"], "stderr_progress_audit", id="json"),
    ],
)
def test_decision_judge_reaches_every_call_site(
    cli: Cli, mode_flags: list[str], audit: str
) -> None:
    cli.monkeypatch.setenv("TOOL_EVAL_DECISION_JUDGE_API_KEY", "judge-key")

    outcome = cli.run(
        *CONNECTION,
        "--scenarios",
        "TC-01",
        "--decision-judge-base-url",
        "http://judge.test/v1",
        "--decision-judge-model",
        "j",
        *mode_flags,
    )

    assert outcome.code == 0
    (run,) = cli.run_kwargs
    assert {key: run[key] for key in run if key.startswith("decision_judge")} == {
        "decision_judge_base_url": "http://judge.test/v1",
        "decision_judge_model": "j",
        "decision_judge_api_key": "<set>",
        "decision_judge": None,
    }
    assert run["on_scenario_audit"] == audit


@pytest.mark.parametrize(
    "mode_flags", [[], ["--no-live"], ["--json"]], ids=["live", "plain", "json"]
)
def test_context_pressure_reaches_the_scored_run(cli: Cli, mode_flags: list[str]) -> None:
    async def calibrate(messages: Any, target: int, *args: Any, **kwargs: Any) -> Any:
        return messages, target - 10

    _patch_everywhere(
        cli.monkeypatch,
        "tool_eval_bench.runner.context_pressure",
        "calibrate_pressure_messages",
        calibrate,
    )

    outcome = cli.run(
        *CONNECTION,
        "--scenarios",
        "TC-01",
        "--seed",
        "1",
        "--context-pressure",
        "0.5",
        "--context-size",
        "32768",
        *mode_flags,
    )

    assert outcome.code == 0
    (run,) = cli.run_kwargs
    assert run["context_pressure_config"] == {
        "ratio": 0.5,
        "fill_tokens": 8262,
        "fill_tokens_target": 8272,
        "context_size": 32768,
    }
    assert run["context_pressure_messages"] == "<8 messages>"
    # Scaled from the fill target: max(120, 120 + 8272 / 50000 * 60).
    assert run["timeout_seconds"] == pytest.approx(129.9264)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _checkpoint(scenario_id: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "status": status,
        "points": 2 if status == "pass" else 0,
        "summary": "s",
        "raw_log": "log",
        **extra,
    }


def _store_interrupted_run(checkpoints: list[dict[str, Any]], **config: Any) -> None:
    _FakeRepository.stored = {
        "status": "interrupted",
        "config": {"model": "m", "backend": "vllm", **config},
        "scores": {},
    }
    _FakeRepository.checkpoints = checkpoints


RESUME_ARGV = ["resume", "r-1", *CONNECTION, "--scenarios", "TC-01", "TC-02"]


@pytest.mark.parametrize(
    ("mode_flags", "callbacks"),
    [
        pytest.param([], LIVE_CALLBACKS, id="live"),
        pytest.param(["--no-live"], PLAIN_CALLBACKS, id="plain"),
        pytest.param(["--json"], JSON_CALLBACKS, id="json"),
    ],
)
def test_resume_reruns_only_what_needs_it(
    cli: Cli, mode_flags: list[str], callbacks: dict[str, str]
) -> None:
    _store_interrupted_run(
        [
            _checkpoint("TC-01", "pass"),
            _checkpoint("TC-02", "fail", failure_kind="timeout"),
        ]
    )

    outcome = cli.run(*RESUME_ARGV, *mode_flags)

    assert outcome.code == 0
    expected = {
        **DEFAULT_RUN_KWARGS,
        "scenarios": ["TC-02"],
        "resume_run_id": "r-1",
        "resume_prior_results": ["TC-01"],
        "resume_scenarios": ["TC-01", "TC-02"],
        "throughput_samples": "<0 samples>",
        **callbacks,
    }
    assert cli.run_kwargs == [expected]
    if mode_flags != ["--json"]:
        assert "Resume: preserving 1 completed outcomes in r-1, running 1 remaining" in (
            outcome.flat_out
        )


def test_resume_without_reusable_outcomes_runs_everything(cli: Cli) -> None:
    _store_interrupted_run([_checkpoint("TC-01", "fail", failure_kind="timeout")])

    outcome = cli.run(*RESUME_ARGV, "--no-live")

    assert outcome.code == 0
    (run,) = cli.run_kwargs
    assert (run["scenarios"], run["resume_run_id"], run["resume_prior_results"]) == (
        ["TC-01", "TC-02"],
        "r-1",
        None,
    )
    assert "No reusable scenario outcomes in run r-1 — running all." in outcome.flat_out


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        pytest.param(None, "Run 'r-1' not found in history.", id="missing"),
        pytest.param(
            {"status": "completed", "config": {}, "scores": {}},
            "Resume aborted: run is already completed",
            id="completed",
        ),
        pytest.param(
            {"status": "interrupted", "config": {"model": "other"}, "scores": {}},
            "Resume aborted: configuration mismatch Prior run differs in: model",
            id="mismatch",
        ),
    ],
)
def test_resume_refusals_exit_one(cli: Cli, stored: dict[str, Any] | None, message: str) -> None:
    _FakeRepository.stored = stored

    outcome = cli.run(*RESUME_ARGV, "--no-live")

    assert outcome.code == 1
    assert message in outcome.flat_out
    assert cli.runs == []


# ---------------------------------------------------------------------------
# Scored-run outcomes and the --json stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode_flags", [[], ["--no-live"], ["--json"]], ids=["live", "plain", "json"]
)
def test_safety_gate_exits_two(cli: Cli, mode_flags: list[str]) -> None:
    cli.safety_warnings = ["TC-01 leaked a secret"]

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--fail-on-safety", *mode_flags)

    assert outcome.code == 2
    if mode_flags == ["--json"]:
        assert json.loads(outcome.err.splitlines()[-1]) == {
            "event": "safety_gate_failed",
            "safety_warnings": ["TC-01 leaked a secret"],
        }
    else:
        assert "SAFETY GATE: TC-01 leaked a secret" in outcome.err


MODES = [
    pytest.param([], id="live"),
    pytest.param(["--no-live"], id="plain"),
    pytest.param(["--json"], id="json"),
]


@pytest.mark.parametrize("mode_flags", MODES)
def test_safety_gate_checks_every_trial(cli: Cli, mode_flags: list[str]) -> None:
    """An unsafe first trial fails the gate, and every requested trial still runs."""
    original = BenchmarkService.run_benchmark

    async def run_benchmark(self: BenchmarkService, **kwargs: Any) -> dict[str, Any]:
        cli.safety_warnings = {0: ["TC-01 leaked a secret"], 1: ["TC-01 sent mail"]}.get(
            len(cli.runs), []
        )
        return await original(self, **kwargs)

    cli.monkeypatch.setattr(BenchmarkService, "run_benchmark", run_benchmark)

    outcome = cli.run(
        *CONNECTION, "--scenarios", "TC-01", "--fail-on-safety", "--trials", "3", *mode_flags
    )

    assert outcome.code == 2
    assert len(cli.runs) == 3
    if mode_flags == ["--json"]:
        union = ["TC-01 leaked a secret", "TC-01 sent mail"]
        assert json.loads(outcome.out)["safety_warnings"] == union
        assert json.loads(outcome.err.splitlines()[-1]) == {
            "event": "safety_gate_failed",
            "safety_warnings": union,
        }


@pytest.mark.parametrize("mode_flags", MODES)
def test_resume_with_trials_runs_later_trials_fresh(cli: Cli, mode_flags: list[str]) -> None:
    _store_interrupted_run(
        [_checkpoint("TC-01", "pass"), _checkpoint("TC-02", "fail", failure_kind="timeout")]
    )

    outcome = cli.run(*RESUME_ARGV, "--trials", "2", *mode_flags)

    assert outcome.code == 0
    first, second = cli.run_kwargs
    assert (first["scenarios"], first["resume_run_id"]) == (["TC-02"], "r-1")
    assert (second["scenarios"], second["resume_run_id"]) == (["TC-01", "TC-02"], None)
    assert (second["resume_prior_results"], second["resume_scenarios"]) == (None, None)


@pytest.mark.parametrize("mode_flags", MODES[:2])
def test_diff_latest_compares_with_the_run_before_this_one(cli: Cli, mode_flags: list[str]) -> None:
    resolved = cli.record(
        "tool_eval_bench.application.run_queries", "resolve_run", ("run-0", {"run_id": "run-0"})
    )
    diffs = cli.record("tool_eval_bench.cli.history", "print_diff")

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--diff", "latest", *mode_flags)

    assert outcome.code == 0
    assert resolved == [{"args": ("latest",), "run_type": "tool_eval", "status": "completed"}]
    assert [call["args"][2] for call in diffs] == ["run-0"]


def test_diff_latest_without_an_earlier_run_says_so(cli: Cli) -> None:
    cli.record("tool_eval_bench.application.run_queries", "resolve_run", None)
    diffs = cli.record("tool_eval_bench.cli.history", "print_diff")

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--diff", "latest", "--no-live")

    assert outcome.code == 0
    assert not diffs
    assert "No previous runs found for comparison." in outcome.out


def test_diff_is_ignored_with_json(cli: Cli, caplog: pytest.LogCaptureFixture) -> None:
    resolved = cli.record("tool_eval_bench.application.run_queries", "resolve_run")

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--diff", "latest", "--json")

    assert outcome.code == 0
    assert not resolved
    assert "--diff prints a console table and is ignored with --json" in caplog.text


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--max-turns", "0"], "--max-turns must be at least 1"),
        (["--error-rate", "nan"], "--error-rate must be between 0 and 1"),
        (["--error-rate", "1.5"], "--error-rate must be between 0 and 1"),
        (["--alpha", "2"], "--alpha must be between 0 and 1"),
        (["--context-pressure", "1.5"], "--context-pressure must be between 0 and 1"),
    ],
)
def test_out_of_range_run_settings_are_invalid_arguments(
    cli: Cli, flags: list[str], message: str
) -> None:
    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", *flags, "--json")

    assert outcome.code == 2
    assert not cli.runs
    error = json.loads(outcome.err.splitlines()[-1])
    assert error["error"] == "invalid_arguments"
    assert message in error["message"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-turns", "1"],
        ["--error-rate", "0"],
        ["--error-rate", "1"],
        ["--alpha", "0"],
        ["--alpha", "1"],
        ["--context-pressure", "1", "--context-size", "32768"],
    ],
)
def test_boundary_run_settings_are_accepted(cli: Cli, flags: list[str]) -> None:
    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", *flags, "--json")

    assert outcome.code == 0, outcome.out
    assert len(cli.runs) == 1


@pytest.mark.parametrize(
    ("mode_flags", "stream", "message"),
    [
        pytest.param([], "out", "Error: boom", id="live"),
        pytest.param(["--no-live"], "out", "Error: boom", id="plain"),
        pytest.param(["--json"], "out", '"error": "boom"', id="json"),
    ],
)
def test_service_failure_exits_one(
    cli: Cli, mode_flags: list[str], stream: str, message: str
) -> None:
    cli.service_error = RuntimeError("boom")

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", *mode_flags)

    assert outcome.code == 1
    assert message in getattr(outcome, f"flat_{stream}")


def test_json_stream_is_one_envelope_on_stdout_and_jsonl_on_stderr(cli: Cli) -> None:
    async def hint(base_url: str, api_key: str | None) -> tuple[str, str]:
        return "vllm", "vLLM"

    _patch_everywhere(cli.monkeypatch, "tool_eval_bench.utils.metadata", "probe_backend_hint", hint)

    outcome = cli.run("--model", "m", "--base-url", BASE_URL, "--scenarios", "TC-01", "--json")

    assert outcome.code == 0
    events = [json.loads(line) for line in outcome.err.splitlines()]
    assert [event["event"] for event in events] == [
        "backend_detected",
        "scenario_start",
        "scenario_result",
    ]
    assert events[0] == {"event": "backend_detected", "backend": "vllm"}
    envelope = json.loads(outcome.out)
    assert envelope["run_id"] == "run-1"
    assert envelope["final_score"] == 90


def test_json_file_keeps_stdout_empty(cli: Cli) -> None:
    target = cli.tmp_path / "out" / "result.json"

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--json-file", str(target))

    assert outcome.code == 0
    assert outcome.out == ""
    events = [json.loads(line) for line in outcome.err.splitlines()]
    assert events[-1] == {
        "event": "benchmark_complete",
        "json_file": str(target),
        "final_score": 90,
    }
    assert json.loads(target.read_text())["run_id"] == "run-1"


# ---------------------------------------------------------------------------
# Modes that stop before the scored run
# ---------------------------------------------------------------------------


def test_probe_mode_stops_after_the_probe(cli: Cli) -> None:
    cli.record("tool_eval_bench.cli.model_probe", "_probe_server")

    outcome = cli.run(*CONNECTION, "--probe")

    assert outcome.code == 0
    assert cli.calls("_probe_server") == [
        {
            "args": ("<console>", BASE_URL, None),
            "headless": False,
            "wire_format": "openai",
            "headers": {},
        }
    ]
    assert cli.runs == [] and cli.contexts == []


def test_spec_live_maps_the_method_and_stops(cli: Cli) -> None:
    cli.record("tool_eval_bench.cli.spec_live_display", "run_spec_live", is_async=True)

    outcome = cli.run(*CONNECTION, "--spec-live", "--spec-method", "draft")

    assert outcome.code == 0
    assert cli.calls("run_spec_live") == [
        {
            "args": (BASE_URL,),
            "api_key": None,
            "metrics_url": None,
            "model_name": "m",
            "poll_interval": 1.0,
            "spec_method": "draft_model",
        }
    ]
    assert cli.runs == [] and cli.leaves["preflight_model_check"] == []


def test_decision_live_stops_and_reports_errors(cli: Cli) -> None:
    calls = cli.record(
        "tool_eval_bench.cli.decision_live_display", "run_decision_live", is_async=True
    )

    outcome = cli.run(*CONNECTION, "--decision-live")

    assert outcome.code == 0
    assert [
        _leaf_call({k: v for k, v in call.items() if k != "adapter_options"}) for call in calls
    ] == [
        {
            "args": (BASE_URL,),
            "model": "m",
            "api_key": None,
            "metrics_url": None,
            "display_url": BASE_URL,
            "interval": 0.5,
            "timeout_seconds": 120.0,
        }
    ]

    from tool_eval_bench.domain.decision import DecisionUnsupportedError

    async def unsupported(*args: Any, **kwargs: Any) -> None:
        raise DecisionUnsupportedError("no decisions here")

    _patch_everywhere(
        cli.monkeypatch,
        "tool_eval_bench.cli.decision_live_display",
        "run_decision_live",
        unsupported,
    )
    outcome = cli.run(*CONNECTION, "--decision-live")

    assert outcome.code == 1
    assert "Decision monitor error: no decisions here" in outcome.flat_out


def test_perf_only_persists_and_exits_by_cell_status(cli: Cli) -> None:
    samples = [ThroughputSample()]
    cli.record("tool_eval_bench.cli.perf", "run_llama_benchy", lambda: samples)
    output_dir = cli.tmp_path / "reports"

    outcome = cli.run(*CONNECTION, "--perf-only", "--output-dir", str(output_dir))

    assert outcome.code == 0
    (persisted,) = [call["args"][0] for call in cli.leaves["persist_run"]]
    assert (persisted["run_type"], persisted["status"], persisted["scores"]) == (
        "perf",
        "completed",
        {"samples": 1, "results": [samples[0].to_result()]},
    )
    assert {key: persisted["config"][key] for key in ("model", "backend", "mode")} == {
        "model": "m",
        "backend": "vllm",
        "mode": "perf-only",
    }
    assert Path(persisted["report_path"]).parent.parent.parent == output_dir
    assert "Report saved to" in outcome.flat_out
    assert cli.runs == []

    samples.append(ThroughputSample(error="boom"))
    outcome = cli.run(*CONNECTION, "--perf-only", "--output-dir", str(output_dir))

    assert outcome.code == 1
    assert cli.leaves["persist_run"][-1]["args"][0]["scores"] == {
        "samples": 2,
        "successful": 1,
        "failed": 1,
        "results": [samples[0].to_result()],
    }
    assert "Throughput benchmark failed in 1 cell(s)." in outcome.flat_out


SPEC_BENCH_CALL = {
    "args": ("<console>", "m", "m", BASE_URL, None),
    "pp": 2048,
    "tg": 128,
    "depths": [0, 4096, 8192],
    "spec_method": "auto",
    "baseline_tg_tps": None,
    "prompt_types": ["filler", "code", "structured"],
    "metrics_url": None,
    "runs": 3,
    "temperature": 0.0,
    "custom_prompts": {},
    "output_dir": None,
    "label": None,
    "run_context": "<RunContext ctx>",
}


@pytest.mark.parametrize(
    ("extra", "scored"),
    [
        pytest.param([], False, id="alone"),
        pytest.param(["--gsm8k"], True, id="with-plugin"),
        pytest.param(["--gsm8k", "--skip-tool-eval"], False, id="skip-tool-eval"),
    ],
)
def test_spec_bench_stops_unless_another_benchmark_follows(
    cli: Cli, extra: list[str], scored: bool
) -> None:
    cli.record("tool_eval_bench.cli.spec_bench", "run_spec_bench", returns=list)

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--spec-bench", "--no-live", *extra)

    assert outcome.code == 0
    assert cli.calls("run_spec_bench") == [SPEC_BENCH_CALL]
    assert bool(cli.runs) is scored


def test_pressure_sweep_stops_after_the_sweep(cli: Cli) -> None:
    cli.record("tool_eval_bench.cli.pressure", "run_pressure_sweep")

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--context-pressure-sweep", "0.5-1.0")

    assert outcome.code == 0
    assert cli.calls("run_pressure_sweep") == [
        {
            "args": ("<console>", "m", "m", "vllm", BASE_URL, None, "<args>"),
            "display_url": BASE_URL,
            "extra_params": None,
            "label": None,
            "run_context": "<RunContext ctx>",
        }
    ]
    assert cli.runs == []


@pytest.mark.parametrize(("handled", "scored"), [(True, False), (False, True)])
def test_plugins_stop_the_cli_when_they_handle_the_run(
    cli: Cli, handled: bool, scored: bool
) -> None:
    cli.record("tool_eval_bench.cli.plugin_runners", "run_selected_plugins", handled)

    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--gsm8k-only", "--no-live")

    assert outcome.code == 0
    (call,) = cli.leaves["run_selected_plugins"]
    assert sorted(call["runners"]) == ["decision", "gsm8k", "ifeval", "mmlu", "needle"]
    assert _leaf_call({k: v for k, v in call.items() if k != "runners"}) == {
        "args": ("<console>", "m", "m", BASE_URL, None, "<args>"),
        "extra_params": None,
        "output_dir": None,
        "run_context": "<RunContext ctx>",
    }
    assert bool(cli.runs) is scored


def test_skip_tool_eval_alone_warns_and_runs_nothing(cli: Cli) -> None:
    outcome = cli.run(*CONNECTION, "--scenarios", "TC-01", "--skip-tool-eval")

    assert outcome.code == 0
    assert "--skip-tool-eval has no effect without" in outcome.flat_out
    assert cli.runs == []


# ---------------------------------------------------------------------------
# Server-independent commands and early exits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "module", "attr", "expected"),
    [
        (["--history"], "tool_eval_bench.cli.history", "print_history", {"args": ("<console>",)}),
        (
            ["--leaderboard"],
            "tool_eval_bench.cli.leaderboard",
            "print_leaderboard",
            {"args": ("<console>",)},
        ),
        (
            ["export", "--format", "csv", "-o", "x.csv"],
            "tool_eval_bench.cli.leaderboard",
            "export_runs",
            {"args": ("<console>",), "fmt": "csv", "output": "x.csv"},
        ),
        (
            ["compare", "a", "b"],
            "tool_eval_bench.cli.history",
            "compare_runs",
            {"args": ("<console>", "a", "b")},
        ),
        (
            ["compare-report", "a.md", "b.md", "-o", "c.html"],
            "tool_eval_bench.cli.compare_report",
            "run_compare_report_command",
            {"args": ("<args>", "<console>")},
        ),
    ],
    ids=["history", "leaderboard", "export", "compare", "compare-report"],
)
def test_storage_commands_route_without_a_server(
    cli: Cli, argv: list[str], module: str, attr: str, expected: dict[str, Any]
) -> None:
    cli.record(module, attr)

    outcome = cli.run(*argv)

    assert outcome.code == 0
    assert cli.calls(attr) == [expected]
    assert cli.runs == [] and cli.contexts == []


def test_dry_run_lists_scenarios_and_exits_zero(cli: Cli) -> None:
    outcome = cli.run("--dry-run", "--scenarios", "TC-01", "--json")

    assert outcome.code == 0
    assert json.loads(outcome.out)["total_scenarios"] == 1


@pytest.mark.parametrize(
    ("mode_flags", "stream", "message"),
    [
        pytest.param(
            ["--json"],
            "err",
            '{"event": "error", "error": "no_server", "message": "No inference server found on localhost. Tried ports:',
            id="json",
        ),
        pytest.param(
            [],
            "err",
            "No inference server found on localhost. Use --base-url or set TOOL_EVAL_BASE_URL",
            id="plain",
        ),
    ],
)
def test_no_server_exits_two(cli: Cli, mode_flags: list[str], stream: str, message: str) -> None:
    cli.record("tool_eval_bench.cli.server", "discover_server", None)

    outcome = cli.run("--model", "m", "--scenarios", "TC-01", *mode_flags)

    assert outcome.code == 2
    assert message in getattr(outcome, f"flat_{stream}")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        pytest.param([*CONNECTION, "--scenarios", "TC-999"], "Unknown scenario", id="scenario"),
        pytest.param(
            [
                *CONNECTION,
                "--perf-only",
                "--decision-judge-base-url",
                "http://j/v1",
                "--decision-judge-model",
                "j",
            ],
            "Decision judge audits require a run or resume of tool-call scenarios",
            id="judge-without-scenarios",
        ),
    ],
)
def test_invalid_selection_is_a_usage_error(cli: Cli, argv: list[str], message: str) -> None:
    outcome = cli.run(*argv)

    assert outcome.code == 2
    assert message in outcome.flat_err
    assert cli.runs == [] and cli.contexts == []
