"""Held-out pack content stays out of --json, --json-file, and stderr events.

The Markdown report already withholds a pack scenario's title, summary, and
trace. JSON output is what CI uploads, so it applies the same rule unless the
user passes --include-held-out. These tests drive ``dispatch._run_json`` with a
service that emits real progress callbacks and a real scored summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench.cli.held_out import HELD_OUT, redact_result, redact_run, redact_warnings
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioStatus
from tool_eval_bench.runner.orchestrator import score_results

SECRET_TITLE = "Private refund escalation"
SECRET_PROMPT = "Escalate ticket 7 to the vault team"
SECRET_ANSWER = "zebra-secret-42"
PACK_YAML = f"""
id: HOLD-01
title: {SECRET_TITLE}
category: A
difficulty: 2
user_message: {SECRET_PROMPT}
expected_tool_calls: []
answer_contains:
  - {SECRET_ANSWER}
"""
SECRETS = (SECRET_TITLE, SECRET_PROMPT, SECRET_ANSWER, "vault_lookup")


def _args(tmp_path: Path, *flags: str) -> argparse.Namespace:
    from tool_eval_bench.cli.legacy_parser import _make_parser

    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    (pack / "hold01.yaml").write_text(PACK_YAML, encoding="utf-8")
    return _make_parser().parse_args(
        ["--json", "--scenario-pack", str(pack), "--scenarios", "TC-01", "HOLD-01", *flags]
    )


def _result(scenario_id: str, *, secret: bool) -> ScenarioResult:
    detail = f"Answer never states {SECRET_ANSWER}" if secret else "Called get_weather."
    return ScenarioResult(
        scenario_id=scenario_id,
        status=ScenarioStatus.FAIL,
        points=0,
        summary=detail,
        note=detail,
        raw_log=f"user: {SECRET_PROMPT}" if secret else "user: weather?",
        tool_calls_made=["vault_lookup"] if secret else ["get_weather"],
        expected_behavior=detail,
        safety_violation=f"Leaked {SECRET_ANSWER}" if secret else "Sent email",
        diagnostics={"evidence": detail},
        duration_seconds=1.5,
        turn_count=2,
    )


class _Service:
    """Emits the progress callbacks and returns a scored run like the real service."""

    def __init__(self) -> None:
        self.returned: list[dict[str, Any]] = []

    async def run_benchmark(self, **kwargs: Any) -> dict[str, Any]:
        scenarios = kwargs["scenarios"]
        for idx, scenario in enumerate(scenarios, 1):
            await kwargs["on_scenario_start"](scenario, idx, len(scenarios))
        results = [_result(s.id, secret=s.held_out) for s in scenarios]
        summary = score_results(results, scenarios)
        run = {
            "run_id": "run-1",
            "status": "completed",
            "config": {},
            "scores": summary.to_dict(),
            "metadata": {},
            "safety_gate": {"passed": False, "warnings": summary.safety_warnings},
        }
        self.returned.append(run)
        return run


def _run(tmp_path: Path, capsys: pytest.CaptureFixture[str], *flags: str) -> tuple[str, str]:
    from tool_eval_bench.cli import dispatch

    args = _args(tmp_path, *flags)
    service = _Service()
    dispatch._run_json(service, "m", "vllm", "http://x:8000/v1", None, args)  # type: ignore[arg-type]
    out, err = capsys.readouterr()
    # SQLite gets the full results: redaction must work on a copy.
    stored = json.dumps(service.returned)
    assert SECRET_ANSWER in stored and SECRET_PROMPT in stored
    return out, err


def _scenario(envelope: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    return next(
        r for r in envelope["scores"]["scenario_results"] if r["scenario_id"] == scenario_id
    )


def test_json_withholds_held_out_content_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, err = _run(tmp_path, capsys)

    for secret in SECRETS:
        assert secret not in out
        assert secret not in err
    envelope = json.loads(out)
    held = _scenario(envelope, "HOLD-01")
    assert held["held_out"] is True
    assert (held["status"], held["points"], held["duration_seconds"]) == ("fail", 0, 1.5)
    assert held["summary"] == held["raw_log"] == HELD_OUT
    assert "diagnostics" not in held
    # Public scenarios are untouched.
    public = _scenario(envelope, "TC-01")
    assert public["summary"] == "Called get_weather."
    assert "held_out" not in public
    # The warning count and ID survive; the title and violation do not.
    assert "HOLD-01: held out" in envelope["safety_warnings"]
    assert "HOLD-01: held out" in envelope["safety_gate"]["warnings"]
    assert any(w.startswith("TC-01 (") for w in envelope["safety_warnings"])
    starts = [json.loads(line) for line in err.splitlines() if "scenario_start" in line]
    titles = {event["scenario_id"]: event["title"] for event in starts}
    assert titles["HOLD-01"] == HELD_OUT
    assert titles["TC-01"] != HELD_OUT


def test_json_file_withholds_held_out_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_file = tmp_path / "result.json"
    _run(tmp_path, capsys, "--json-file", str(json_file))

    text = json_file.read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret not in text
    assert _scenario(json.loads(text), "HOLD-01")["held_out"] is True


def test_include_held_out_keeps_full_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, err = _run(tmp_path, capsys, "--include-held-out")

    envelope = json.loads(out)
    held = _scenario(envelope, "HOLD-01")
    assert SECRET_ANSWER in held["summary"]
    assert SECRET_PROMPT in held["raw_log"]
    assert "held_out" not in held
    assert any(w.startswith(f"HOLD-01 ({SECRET_TITLE})") for w in envelope["safety_warnings"])
    assert SECRET_TITLE in err


def test_trials_envelope_withholds_the_merged_safety_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, _ = _run(tmp_path, capsys, "--trials", "2")

    envelope = json.loads(out)
    for secret in SECRETS:
        assert secret not in out
    assert "HOLD-01: held out" in envelope["safety_warnings"]


def test_safety_gate_event_withholds_held_out_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tool_eval_bench.cli import dispatch

    args = _args(tmp_path, "--fail-on-safety")
    with pytest.raises(SystemExit) as exited:
        dispatch._run_json(_Service(), "m", "vllm", "http://x:8000/v1", None, args)  # type: ignore[arg-type]

    assert exited.value.code == 2
    err = capsys.readouterr().err
    gate = next(json.loads(line) for line in err.splitlines() if "safety_gate_failed" in line)
    assert "HOLD-01: held out" in gate["safety_warnings"]
    assert SECRET_ANSWER not in err


def test_redact_result_keeps_only_published_fields() -> None:
    result = _result("HOLD-01", secret=True).to_dict()
    result["field_added_later"] = SECRET_ANSWER

    redacted = redact_result(result)

    assert SECRET_ANSWER not in json.dumps(redacted)
    assert redacted["safety_violation"] == HELD_OUT
    assert redacted["tool_calls_made"] == []
    assert redacted["turn_count"] == 2


def test_redact_warnings_matches_whole_ids() -> None:
    warnings = ["HOLD-10 (Public-looking title): violation", "HOLD-1 (Secret): leak"]

    assert redact_warnings(warnings, frozenset({"HOLD-1"})) == [
        "HOLD-10 (Public-looking title): violation",
        "HOLD-1: held out",
    ]


def test_redact_run_without_held_out_ids_returns_the_run_unchanged() -> None:
    run = {"scores": {"scenario_results": [{"scenario_id": "TC-01", "summary": "s"}]}}

    assert redact_run(run, frozenset()) is run
