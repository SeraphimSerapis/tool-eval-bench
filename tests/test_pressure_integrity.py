"""Context-pressure results must describe what the model actually faced.

A sweep's breaking point is its headline number, so these pin the inputs that
used to corrupt it: infrastructure failures scored as model failures, windows
too small to hold any filler, interrupted sweeps, and stored configs that could
not tell two different fills apart.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from tool_eval_bench.domain.scenarios import (
    Category,
    FailureKind,
    ModelScoreSummary,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioStatus,
)
from tool_eval_bench.runner.context_pressure import (
    _RESERVED_FOR_OUTPUT,
    _RESERVED_FOR_SCENARIO,
    _TOKENS_PER_FILLER_CHUNK,
)

RESERVE = _RESERVED_FOR_OUTPUT + _RESERVED_FOR_SCENARIO


def _scenario(sid: str) -> ScenarioDefinition:
    return ScenarioDefinition(
        id=sid,
        title="Test",
        category=Category.A,
        user_message="test",
        description="test",
        handle_tool_call=lambda s, c: {},
        evaluate=lambda s: None,
    )


def _result(sid: str, status: str, kind: str | None = None) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        status=ScenarioStatus(status),
        points={"pass": 2, "partial": 1, "fail": 0}[status],
        summary=f"{sid} {status}",
        failure_kind=kind,
    )


def _summary(*results: ScenarioResult) -> ModelScoreSummary:
    return ModelScoreSummary(
        scenario_results=list(results),
        total_points=sum(r.points for r in results),
        max_points=2 * len(results),
        final_score=0.0,
        rating="test",
        category_scores=[],
        safety_warnings=[],
        total_tokens=0,
    )


def _levels(*outcomes: ModelScoreSummary | BaseException) -> Any:
    """A fake ``run_all_scenarios`` returning (or raising) one outcome per level."""
    queue = list(outcomes)

    async def run_all(adapter: Any, **kwargs: Any) -> ModelScoreSummary:
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return run_all


def _sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario_ids: list[str],
    run_all: Any,
    *,
    estimated: bool = False,
    **overrides: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Run ``run_pressure_sweep`` with only the network faked.

    Returns the persisted runs and the console output.
    """
    from tool_eval_bench.adapters import factory
    from tool_eval_bench.application import run_queries
    from tool_eval_bench.cli import pressure
    from tool_eval_bench.runner import context_pressure, orchestrator

    async def calibrate(
        messages: Any, target: int, *args: Any, on_estimated: Any = None, **kwargs: Any
    ) -> tuple[Any, int]:
        if estimated and on_estimated is not None:
            on_estimated()
        return messages, target

    persisted: list[dict[str, Any]] = []
    monkeypatch.setattr(context_pressure, "calibrate_pressure_messages", calibrate)
    monkeypatch.setattr(orchestrator, "run_all_scenarios", run_all)
    monkeypatch.setattr(factory, "build_adapter", lambda *a, **k: AsyncMock())
    scenarios = [_scenario(sid) for sid in scenario_ids]
    monkeypatch.setattr(pressure, "resolve_scenarios", lambda _args: scenarios)
    monkeypatch.setattr(run_queries, "persist_run", persisted.append)
    settings: dict[str, Any] = {
        "context_pressure_sweep": "0.5-1.0",
        "sweep_steps": 3,
        "context_size": 65536,
        "seed": None,
        "system_prompt": None,
        "output_dir": str(tmp_path),
        "timeout": 60.0,
    }
    settings.update(overrides)
    out = io.StringIO()
    pressure.run_pressure_sweep(
        Console(file=out, force_terminal=False, width=200),
        "m",
        "m",
        "vllm",
        "http://test/v1",
        None,
        argparse.Namespace(**settings),
    )
    return persisted, out.getvalue()


def _report(run: dict[str, Any]) -> str:
    return Path(run["report_path"]).read_text(encoding="utf-8")


# -- H08-01: infrastructure failures are not model failures ------------------------


@pytest.mark.parametrize(
    "kind",
    [FailureKind.SERVER_ERROR, FailureKind.CONNECTION_ERROR, FailureKind.TIMEOUT],
)
def test_infrastructure_failures_leave_the_breaking_point_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    # TC-45 on an endpoint that ignores tool_choice='required' is reported as
    # FAIL + SERVER_ERROR at every level; the model passes everything else.
    level = _summary(_result("TC-01", "pass"), _result("TC-45", "fail", kind))
    persisted, out = _sweep(monkeypatch, tmp_path, ["TC-01", "TC-45"], _levels(level, level, level))

    scores = persisted[0]["scores"]
    assert scores["breaking_point"] == 1.0
    assert scores["first_degradation"] is None
    assert [lv["score_pct"] for lv in scores["level_results"]] == [100.0, 100.0, 100.0]
    assert [lv["excluded_count"] for lv in scores["level_results"]] == [1, 1, 1]
    assert "(1 excluded)" in out
    assert "- **Excluded**: 1 (infrastructure failures" in _report(persisted[0])


def test_a_model_failure_still_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    level = _summary(_result("TC-01", "pass"), _result("TC-45", "fail", FailureKind.WRONG_TOOL))
    persisted, _ = _sweep(monkeypatch, tmp_path, ["TC-01", "TC-45"], _levels(level, level, level))

    scores = persisted[0]["scores"]
    assert scores["breaking_point"] is None
    assert scores["first_degradation"] == 0.5
    assert scores["level_results"][0]["score_pct"] == 50.0
    assert scores["level_results"][0]["excluded_count"] == 0
    assert "**Excluded**" not in _report(persisted[0])


def test_a_level_error_is_neither_pass_nor_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passing = _summary(_result("TC-01", "pass"))
    persisted, _ = _sweep(
        monkeypatch,
        tmp_path,
        ["TC-01"],
        _levels(passing, RuntimeError("upstream reset"), passing),
    )

    scores = persisted[0]["scores"]
    assert scores["breaking_point"] == 1.0
    assert scores["first_degradation"] is None
    errored = scores["level_results"][1]
    assert (errored["score_pct"], errored["excluded_count"]) == (None, 1)
    assert "- **Pass Rate**: n/a (no scenario was scored)" in _report(persisted[0])


def test_unscored_levels_do_not_break_an_all_fail_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failing = _summary(_result("TC-01", "fail", FailureKind.WRONG_TOOL))
    unscored = _summary(_result("TC-01", "fail", FailureKind.TIMEOUT))
    persisted, out = _sweep(
        monkeypatch,
        tmp_path,
        ["TC-01"],
        _levels(failing, unscored, failing, failing),
        sweep_steps=4,
    )

    # fail, (unscored), fail: two all-fail levels with nothing scored between.
    assert persisted[0]["scores"]["levels"] == 3
    assert "stopped (2 consecutive all-fail levels)" in out


def test_two_unscored_levels_stop_the_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unscored = _summary(_result("TC-01", "fail", FailureKind.CONNECTION_ERROR))
    persisted, out = _sweep(
        monkeypatch, tmp_path, ["TC-01"], _levels(unscored, unscored, unscored), sweep_steps=3
    )

    scores = persisted[0]["scores"]
    assert scores["levels"] == 2
    assert scores["breaking_point"] is None
    assert scores["first_degradation"] is None
    assert "stopped (2 consecutive levels with nothing scored)" in out


# -- H08-02: a window too small for real pressure is refused ------------------------


@pytest.mark.parametrize(
    "context_size", [4096, 8192, 16384, RESERVE + _TOKENS_PER_FILLER_CHUNK - 1]
)
def test_sweep_refuses_a_window_without_room_for_one_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, context_size: int
) -> None:
    run_all = AsyncMock()
    with pytest.raises(SystemExit) as exc:
        _sweep(monkeypatch, tmp_path, ["TC-01"], run_all, context_size=context_size)

    assert exc.value.code == 1
    run_all.assert_not_called()


def test_sweep_accepts_the_smallest_window_with_one_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passing = _summary(_result("TC-01", "pass"))
    persisted, _ = _sweep(
        monkeypatch,
        tmp_path,
        ["TC-01"],
        _levels(passing, passing),
        sweep_steps=2,
        context_size=RESERVE + _TOKENS_PER_FILLER_CHUNK,
    )

    assert persisted[0]["scores"]["level_results"][-1]["fill_tokens"] == _TOKENS_PER_FILLER_CHUNK


def test_sweep_checks_the_top_of_the_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 32K holds plenty at 100%, but a sweep that tops out at 10% fills ~1.6K.
    with pytest.raises(SystemExit):
        _sweep(
            monkeypatch,
            tmp_path,
            ["TC-01"],
            AsyncMock(),
            context_pressure_sweep="0.05-0.1",
            context_size=32768,
        )


def _scored_pressure_run(
    monkeypatch: pytest.MonkeyPatch, ratio: str, fill_tokens: int, *, estimated: bool = False
) -> list[dict[str, Any]]:
    """Run ``main`` with ``--context-pressure`` and a faked fill.

    Returns the keyword arguments the benchmark service received.
    """
    from tool_eval_bench.cli import dispatch, plugin_runners
    from tool_eval_bench.runner import context_pressure
    from tool_eval_bench.runner.context_pressure import ContextPressureConfig
    from tool_eval_bench.utils import metadata

    async def no_context(**kwargs: Any) -> None:
        return None

    async def prepare(*args: Any, ratio: float, **kwargs: Any) -> ContextPressureConfig:
        return ContextPressureConfig(ratio=ratio, fill_tokens=fill_tokens, detected_context=8192)

    async def calibrate(
        messages: Any, target: int, *args: Any, on_estimated: Any = None, **kwargs: Any
    ) -> tuple[Any, int]:
        if estimated and on_estimated is not None:
            on_estimated()
        return messages, target + 37

    calls: list[dict[str, Any]] = []

    class Service:
        async def run_benchmark(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"run_id": "r", "scores": {"scenario_results": []}}

    monkeypatch.setattr(dispatch, "_load_dotenv", lambda: None)
    monkeypatch.setattr(dispatch, "_preflight_model_check", lambda *a, **k: None)
    monkeypatch.setattr(metadata, "collect_run_context", no_context)
    monkeypatch.setattr(plugin_runners, "run_selected_plugins", lambda *a, **k: False)
    monkeypatch.setattr(context_pressure, "prepare_context_pressure", prepare)
    monkeypatch.setattr(context_pressure, "calibrate_pressure_messages", calibrate)
    monkeypatch.setattr(dispatch, "BenchmarkService", lambda **kwargs: Service())
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
            "http://localhost:8000",
            "--scenarios",
            "TC-01",
            "--context-pressure",
            ratio,
            "--no-warmup",
            "--no-live",
        ],
    )
    dispatch.main()
    return calls


def test_single_pressure_run_refuses_a_zero_fill(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _scored_pressure_run(monkeypatch, "0.75", 0)

    assert exc.value.code == 1
    assert "too small for --context-pressure" in capsys.readouterr().out


def test_single_pressure_run_accepts_a_small_positive_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A light ratio on a small window: under one chunk, still real pressure.
    (call,) = _scored_pressure_run(monkeypatch, "0.1", 500)

    assert call["context_pressure_config"]["fill_tokens_target"] == 500


def test_zero_ratio_with_zero_fill_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    (call,) = _scored_pressure_run(monkeypatch, "0", 0)

    assert call["context_pressure_config"]["ratio"] == 0.0


# -- H08-06: an estimated fill is labelled as one ----------------------------------


def test_scored_run_marks_an_estimated_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    (estimated,) = _scored_pressure_run(monkeypatch, "0.5", 4096, estimated=True)
    (measured,) = _scored_pressure_run(monkeypatch, "0.5", 4096)

    assert estimated["context_pressure_config"]["fill_tokens_estimated"] is True
    assert "fill_tokens_estimated" not in measured["context_pressure_config"]


@pytest.mark.parametrize(("flag", "labelled"), [(True, True), (False, False), (None, False)])
def test_scored_report_labels_an_estimated_fill(
    tmp_path: Path, flag: bool | None, labelled: bool
) -> None:
    from tool_eval_bench.storage.reports.scenario import write_scenario_report

    config: dict[str, Any] = {"ratio": 0.5, "fill_tokens": 4133, "context_size": 8192}
    if flag is not None:
        config["fill_tokens_estimated"] = flag
    path = write_scenario_report(
        tmp_path,
        "r",
        "m",
        _summary(_result("TC-01", "pass")),
        context_pressure_config=config,
    )
    markdown = path.read_text(encoding="utf-8")

    assert "(~4,133 tokens prefilled of 8,192 context" in markdown
    assert ("estimated, the server has no compatible /tokenize" in markdown) is labelled


def test_sweep_marks_estimated_levels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    passing = _summary(_result("TC-01", "pass"))
    estimated, _ = _sweep(
        monkeypatch,
        tmp_path / "e",
        ["TC-01"],
        _levels(passing, passing),
        sweep_steps=2,
        estimated=True,
    )
    measured, _ = _sweep(
        monkeypatch, tmp_path / "m", ["TC-01"], _levels(passing, passing), sweep_steps=2
    )

    assert all(lv["fill_tokens_estimated"] for lv in estimated[0]["scores"]["level_results"])
    assert not any("fill_tokens_estimated" in lv for lv in measured[0]["scores"]["level_results"])
    assert "(target; not measured" in _report(estimated[0])
    assert "(target; not measured" not in _report(measured[0])


@pytest.mark.asyncio
async def test_calibration_reports_when_it_falls_back_to_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_eval_bench.runner import context_pressure

    messages = [{"role": "user", "content": "x" * 4000}, {"role": "assistant", "content": "ok"}]

    async def calibrate(count: int | None) -> tuple[int, list[bool]]:
        async def counter(*args: Any, **kwargs: Any) -> int | None:
            return count

        monkeypatch.setattr(context_pressure, "count_messages_tokens", counter)
        calls: list[bool] = []
        _, tokens = await context_pressure.calibrate_pressure_messages(
            list(messages),
            1000,
            "http://x",
            "m",
            client_factory=AsyncMock(),
            on_estimated=lambda: calls.append(True),
        )
        return tokens, calls

    assert await calibrate(None) == (1000, [True])
    assert await calibrate(1000) == (1000, [])


# -- H08-04: the stored sweep config pins the effective window and seed -------------


def test_sweep_config_records_context_size_and_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passing = _summary(_result("TC-01", "pass"))

    def run(**overrides: Any) -> dict[str, Any]:
        persisted, _ = _sweep(
            monkeypatch,
            tmp_path,
            ["TC-01"],
            _levels(passing, passing),
            sweep_steps=2,
            **overrides,
        )
        return persisted[0]["config"]

    small = run(context_size=32768, seed=7)
    large = run(context_size=262144, seed=7)
    reseeded = run(context_size=32768, seed=8)
    again = run(context_size=32768, seed=7)

    assert (small["context_size"], small["seed"]) == (32768, 7)
    assert small["config_fingerprint"] != large["config_fingerprint"]
    assert small["config_fingerprint"] != reseeded["config_fingerprint"]
    assert small["config_fingerprint"] == again["config_fingerprint"]


def test_sweep_config_records_the_kv_capped_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tool_eval_bench.runner import context_pressure
    from tool_eval_bench.runner.context_pressure import KvCapacityInfo

    async def detect(*args: Any, **kwargs: Any) -> int:
        return 131072

    async def kv(*args: Any, **kwargs: Any) -> KvCapacityInfo:
        return KvCapacityInfo(num_blocks=2048, block_size=16, capacity=32768, is_hybrid=False)

    monkeypatch.setattr(context_pressure, "detect_context_size", detect)
    monkeypatch.setattr(context_pressure, "detect_kv_capacity", kv)
    passing = _summary(_result("TC-01", "pass"))
    persisted, _ = _sweep(
        monkeypatch,
        tmp_path,
        ["TC-01"],
        _levels(passing, passing),
        sweep_steps=2,
        context_size=None,
    )

    assert persisted[0]["config"]["context_size"] == 32768


# -- H08-07: an interrupted sweep withholds its breaking point ----------------------


def test_interrupted_sweep_withholds_the_breaking_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    partial = _summary(_result("TC-01", "pass"), _result("TC-02", "partial"))
    passing = _summary(_result("TC-01", "pass"), _result("TC-02", "pass"))
    persisted, out = _sweep(
        monkeypatch,
        tmp_path,
        ["TC-01", "TC-02"],
        _levels(passing, partial, KeyboardInterrupt()),
        sweep_steps=5,
    )

    run = persisted[0]
    # "completed" on purpose: history and resume read anything else as a
    # resumable scored run.
    assert run["status"] == "completed"
    scores = run["scores"]
    assert (scores["interrupted"], scores["levels"], scores["planned_levels"]) == (True, 2, 5)
    assert scores["breaking_point"] is None
    assert scores["first_degradation"] == 0.625
    assert "withheld (interrupted after 2 of 5 levels)" in out
    assert "- **Breaking Point**: withheld (interrupted after 2 of 5 levels)" in _report(run)


def test_a_finished_sweep_is_not_marked_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    passing = _summary(_result("TC-01", "pass"))
    persisted, _ = _sweep(
        monkeypatch, tmp_path, ["TC-01"], _levels(passing, passing), sweep_steps=2
    )

    scores = persisted[0]["scores"]
    assert (scores["interrupted"], scores["planned_levels"], scores["breaking_point"]) == (
        False,
        2,
        1.0,
    )


# -- R08-M1: calibration noise does not split scored pressure cohorts ---------------


def _pressure_config(**pressure: Any) -> dict[str, Any]:
    from tool_eval_bench.application.run_config import RunSettings, build_run_config

    block = {
        "ratio": 0.75,
        "fill_tokens": 84_210,
        "fill_tokens_target": 84_720,
        "context_size": 131_072,
        **pressure,
    }
    settings = RunSettings(
        model="m",
        backend="vllm",
        base_url="http://localhost:8000",
        temperature=0.0,
        timeout_seconds=120.0,
        max_turns=8,
        seed=None,
        reference_date=None,
        concurrency=1,
        error_rate=0.0,
        alpha=0.7,
        extra_params=None,
        context_pressure_config=block,
        weight_by_difficulty=False,
    )
    return build_run_config(settings, scenarios=[_scenario("TC-01")], metadata={})


def test_calibrated_fill_does_not_change_the_fingerprint() -> None:
    base = _pressure_config()
    jittered = _pressure_config(fill_tokens=84_655)

    assert base["config_fingerprint"] == jittered["config_fingerprint"]
    # The stored config keeps the measured count for diagnosis.
    assert jittered["context_pressure"]["fill_tokens"] == 84_655


@pytest.mark.parametrize(
    "change",
    [
        {"ratio": 0.5},
        {"fill_tokens_target": 82_652},
        {"context_size": 65_536},
        {"fill_tokens_estimated": True},
    ],
)
def test_conditions_the_model_sees_still_change_the_fingerprint(change: dict[str, Any]) -> None:
    assert (
        _pressure_config()["config_fingerprint"] != _pressure_config(**change)["config_fingerprint"]
    )


def test_resume_ignores_calibration_details() -> None:
    from tool_eval_bench.application.run_config import resume_mismatches

    stored = _pressure_config()
    current = _pressure_config(fill_tokens=80_001, fill_tokens_estimated=True)

    assert resume_mismatches(stored, current) == []
    assert resume_mismatches(stored, _pressure_config(ratio=0.5)) != []


# -- H08-03: held-out packs stay out of the sweep report ----------------------------


@pytest.mark.parametrize(
    ("flags", "rejected"),
    [
        (["--context-pressure-sweep", "0.5-1.0"], True),
        (["--context-pressure", "0.5"], False),
        ([], False),
    ],
    ids=["sweep", "single-pressure", "plain"],
)
def test_only_the_sweep_refuses_a_held_out_pack(
    tmp_path: Path, flags: list[str], rejected: bool
) -> None:
    from tests.test_scenario_packs import _write_pack
    from tool_eval_bench.cli.dispatch import _validate_scenario_selection
    from tool_eval_bench.cli.legacy_parser import make_parser

    pack = _write_pack(tmp_path / "pack", "HO-1")
    parser = make_parser()
    args = parser.parse_args([*flags, "--scenario-pack", str(pack), "--json"])

    if rejected:
        with pytest.raises(SystemExit) as exc:
            _validate_scenario_selection(args, parser, Console(file=io.StringIO()))
        assert exc.value.code == 2
    else:
        _validate_scenario_selection(args, parser, Console(file=io.StringIO()))
