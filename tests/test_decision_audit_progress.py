"""Judge activity is visible only when a request was attempted."""

import asyncio
import json
from io import StringIO
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console
from scenario_replay import SCENARIOS

from tool_eval_bench.application.decision_audit import (
    MAX_AUDIT_REQUEST_BYTES,
    capture_decision_audit,
    decision_judge_config,
    run_decision_audit,
)
from tool_eval_bench.cli.dispatch import _plain_on_audit
from tool_eval_bench.cli.display import BenchmarkDisplay, decision_audit_line
from tool_eval_bench.cli.run_io import stderr_progress_audit
from tool_eval_bench.domain.decision import ChoiceAnswer, DecisionResult
from tool_eval_bench.domain.scenarios import ScenarioResult, ScenarioState, ScenarioStatus

SCENARIO = SCENARIOS["TC-89"]
CONFIG = decision_judge_config("http://judge/v1", "clef-flash")


def _captured(
    text="The invoice is paid. The payment failed and the hold is released.",
):
    result = ScenarioResult("TC-89", ScenarioStatus.FAIL, 0, "Original verdict")
    capture_decision_audit(
        SCENARIO,
        ScenarioState(final_answer=text, assistant_messages=[text] if text else []),
        result,
        judge_set="recommended",
    )
    return result


def _adapter(choice="no_payment_claim"):
    return Mock(
        decide=AsyncMock(
            return_value=DecisionResult(
                answers={
                    "tc89-payment-claim-v1": ChoiceAnswer(
                        choice,
                        {
                            option: 0.96 if option == choice else 0.02
                            for option in ["payment_claim", "no_payment_claim", "unclear"]
                        },
                    )
                }
            )
        )
    )


async def test_started_and_completed_updates_show_judge_and_disagreement():
    result = _captured()
    display = BenchmarkDisplay("benchmark", "unknown", "http://benchmark")
    display.console = Console(file=StringIO(), width=200, no_color=True)
    adapter = _adapter()

    async def progress(phase):
        assert adapter.decide.await_count == (0 if phase == "started" else 1)
        await display.on_scenario_audit(SCENARIO, result, phase)

    await run_decision_audit(adapter, result.decision_audit, config=CONFIG, on_progress=progress)
    output = display.console.file.getvalue()
    assert output.count("TC-89") == 2
    assert "clef-flash" in output
    assert "judging" in output
    assert "no_payment_claim (96.0%)" in output
    assert "disagrees" in output
    assert "official score unchanged" in output
    assert result.points == 0


@pytest.mark.parametrize("text", ["", "x" * MAX_AUDIT_REQUEST_BYTES])
async def test_unsent_audits_have_no_updates_or_placeholders(text, capsys):
    result = _captured(text)
    adapter = _adapter()
    progress = AsyncMock()
    await run_decision_audit(adapter, result.decision_audit, config=CONFIG, on_progress=progress)
    progress.assert_not_awaited()
    adapter.decide.assert_not_awaited()
    assert not result.was_decision_audited
    assert decision_audit_line("TC-89", result, "completed") is None
    await _plain_on_audit(SCENARIO, result, "completed")
    await stderr_progress_audit(SCENARIO, result, "completed")
    assert capsys.readouterr() == ("", "")


async def test_unaudited_scenarios_emit_nothing(capsys):
    result = ScenarioResult("TC-01", ScenarioStatus.PASS, 2, "Weather found")
    assert decision_audit_line("TC-01", result, "completed") is None
    await _plain_on_audit(SCENARIOS["TC-01"], result, "completed")
    await stderr_progress_audit(SCENARIOS["TC-01"], result, "completed")
    assert capsys.readouterr() == ("", "")


async def test_abstention_and_plain_output_show_the_actual_model(capsys):
    result = _captured()

    async def progress(phase):
        await _plain_on_audit(SCENARIO, result, phase)

    await run_decision_audit(
        _adapter("unclear"), result.decision_audit, config=CONFIG, on_progress=progress
    )
    output = capsys.readouterr().out
    assert "Audit · clef-flash: judging..." in output
    assert "abstained · unclear (96.0%)" in output
    assert "disagrees" not in output
    assert "official score unchanged" in output


async def test_failed_requests_emit_jsonl_results_without_raw_evidence(capsys):
    result = _captured()
    adapter = Mock(decide=AsyncMock(side_effect=RuntimeError("secret response detail")))

    async def progress(phase):
        await stderr_progress_audit(SCENARIO, result, phase)

    await run_decision_audit(adapter, result.decision_audit, config=CONFIG, on_progress=progress)
    captured = capsys.readouterr()
    messages = [json.loads(line) for line in captured.err.splitlines()]
    assert captured.out == ""
    assert [message["event"] for message in messages] == [
        "decision_audit_start",
        "decision_audit_result",
    ]
    assert messages[0]["model"] == "clef-flash"
    assert messages[1]["scenario_id"] == "TC-89"
    assert messages[1]["status"] == "unavailable"
    assert messages[1]["error_type"] == "RuntimeError"
    assert "input" not in messages[1]
    assert "secret" not in captured.err
    line = decision_audit_line("TC-89", result, "completed").plain
    assert "clef-flash: unavailable (RuntimeError)" in line
    assert "official score unchanged" in line


async def test_successful_jsonl_result_includes_probabilities_and_comparison(capsys):
    result = _captured()

    async def progress(phase):
        await stderr_progress_audit(SCENARIO, result, phase)

    await run_decision_audit(_adapter(), result.decision_audit, config=CONFIG, on_progress=progress)
    message = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert message["choice"] == "no_payment_claim"
    assert message["probabilities"]["no_payment_claim"] == 0.96
    assert message["disagreement"] is True
    assert message["elapsed_ms"] >= 0
    assert "question" not in message
    assert "base_url" not in message


async def test_saved_successful_audit_is_shown_without_a_new_request(capsys):
    result = _captured("Payment failed.")
    adapter = _adapter()
    await run_decision_audit(adapter, result.decision_audit, config=CONFIG)
    result.decision_audit.pop("request_started")  # Backward-compatible older stored audit.

    async def progress(phase):
        assert phase == "reused"
        await _plain_on_audit(SCENARIO, result, phase)
        await stderr_progress_audit(SCENARIO, result, phase)

    await run_decision_audit(adapter, result.decision_audit, config=CONFIG, on_progress=progress)
    adapter.decide.assert_awaited_once()
    captured = capsys.readouterr()
    assert "saved audit" in captured.out
    assert "agrees with deterministic check" in captured.out
    assert "judging" not in captured.out
    assert json.loads(captured.err)["reused"] is True


async def test_progress_errors_cannot_change_judgment_or_expose_exception_text(caplog):
    result = _captured()
    progress = AsyncMock(side_effect=RuntimeError("secret callback detail"))
    await run_decision_audit(_adapter(), result.decision_audit, config=CONFIG, on_progress=progress)
    assert progress.await_count == 2
    assert result.decision_audit["status"] == "completed"
    assert result.points == 0
    assert "secret" not in caplog.text


async def test_cancellation_does_not_emit_a_completed_judgment():
    result = _captured()
    started = asyncio.Event()
    phases = []

    async def hang(**kwargs):
        await asyncio.Future()

    async def progress(phase):
        phases.append(phase)
        started.set()

    task = asyncio.create_task(
        run_decision_audit(
            Mock(decide=hang), result.decision_audit, config=CONFIG, on_progress=progress
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert phases == ["started"]
    assert result.decision_audit["status"] == "pending"


def test_model_text_is_literal_and_control_characters_are_not_executed():
    result = _captured()
    result.decision_audit.update(request_started=True, model="clef[bold]\x1b[2J")
    line = decision_audit_line("TC-89", result, "started")
    assert "[bold]" in line.plain
    assert "\x1b" not in line.plain
    result.decision_audit["model"] = ""
    assert decision_audit_line("TC-89", result, "started") is None
