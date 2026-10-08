"""Judge sets choose which answer audits run; they never change scores or cohorts."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from tool_eval_bench.application.decision_audit import (
    capture_decision_audit,
    decision_judge_config,
    refresh_saved_audit,
    selected_audit,
    with_selected_checks,
)
from tool_eval_bench.application.run_config import RunSettings, build_run_config
from tool_eval_bench.cli.bench import _make_parser
from tool_eval_bench.cli.dispatch import _resume_config_mismatches, main
from tool_eval_bench.cli.leaderboard import _cohort_fingerprint
from tool_eval_bench.cli.legacy_parser import make_parser
from tool_eval_bench.cli.parser import parse_cli_args
from tool_eval_bench.domain.scenarios import (
    ModelScoreSummary,
    ScenarioResult,
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
)
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, ALL_SCENARIOS_WITH_HARDMODE
from tool_eval_bench.evals.variants import apply_variants, identity_variant
from tool_eval_bench.storage.reports.scenario import write_scenario_report

JUDGE_URL = "http://judge.test/v1"
BY_ID = {s.id: s for s in ALL_SCENARIOS_WITH_HARDMODE}
RECOMMENDED = {
    "TC-89",
    "TC-91",
    "TC-76",
    "TC-80",
    "TC-49",
    "TC-32",
    "TC-59",
    "TC-71",
    "TC-26",
    "TC-14",
    "TC-87",
}
ALL = RECOMMENDED | {"TC-12", "TC-33", "TC-36", "TC-31", "TC-75", "TC-16"}

# A question edit must change its hash, which makes resume re-judge saved
# evidence. Bump the check_id and this pin together.
QUESTION_SHA256 = {
    "tc12-deletion-stance-v3": "42ddf9bf9804aee5587b16229b47ad4434cec25f233f19c4af30479e6862f243",
    "tc14-error-acknowledged-v1": "f998e9d3af5535d9b695fb1ae387113bbf82c8c21887726ed9b40145f2757651",
    "tc16-answer-language-v1": "8b6f3983ec42c089a0fc381654af32c4fc095fcddfa5f3c1caf79161bae46e92",
    "tc26-attendee-claim-v1": "c3436e99eb6442cbfc5c26bb62b8a08008bfe4adcd07c43b78aed2b0846a8118",
    "tc31-asks-clarification-v1": "1511f69c1bdd9071508b690065318a1f5e1f8c892a64a59125f3f39e22de3ac8",
    "tc32-cleared-claim-v1": "dc037afc147a1f6655aa321c2b7a3322d6a40a029b089f53a7a7a77ad0d059c6",
    "tc33-internal-claim-v2": "cce40cc0f23d63a498b20c4d589fad206d5f7389fd9d5c216075f5b96ddad2c9",
    "tc36-asks-details-v1": "b22cd51bd06a005b03c14981af16e0a8428cde53298f00d192fd2a981f6f3a07",
    "tc49-sent-claim-v2": "60f75bff22c8e4fc8b915d10f709be88fb80b50062d3f76d799629c69198a2d2",
    "tc59-refuses-v2": "b31d562ec527f6aa768d11ac41dfa81cd5a7958e11e68c7f96bc7a29e14aac41",
    "tc71-asks-which-jordan-v1": "a632a42d0b25b829dd9422594f5ce301c602282b0a2545683c369724b2fbeff9",
    "tc75-asks-date-time-v1": "08cb2516960d2839f69300b233baae0052138454ed927c0ba50f31644696767b",
    "tc76-cancel-refund-claim-v1": "cc2c02a5cf36cae3e48568bb68054b897eacfa01f1a2dc5b4353443c38230c2f",
    "tc80-move-claim-v1": "2701e36b911d22bb818e52014331b3f73921375807f959e2bfd211f44e9e39eb",
    "tc87-completeness-claim-v1": "c4bc1eb62add8a7b1e4e539d3cdf45ec64bcf910e35ccc1a1b299d8fe8053072",
    "tc89-payment-claim-v1": "2fcf9f6c5ff060c5ef65f5f165acd4cfcdd9b275f6c46d8465df9084acc90c3f",
    "tc91-revocation-claim-v2": "0be8a346c203716adb7dfe557551308561af33772cc0fedf4de7112fca1a8cae",
}


def _selected(scenarios, judge_set):
    return {s.id for s in scenarios if selected_audit(s, judge_set) is not None}


# -- flag parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], None),
        (["--decision-judge"], "recommended"),
        (["--decision-judge", "recommended"], "recommended"),
        (["--decision-judge", "all"], "all"),
    ],
)
@pytest.mark.parametrize("subcommand", [False, True])
def test_decision_judge_flag_forms(flags, expected, subcommand):
    if subcommand:
        _, args = parse_cli_args(_make_parser, ["run", *flags])
    else:
        args = make_parser().parse_args(flags)
    assert args.decision_judge == expected


def test_decision_judge_rejects_an_unknown_set(capsys):
    with pytest.raises(SystemExit) as error:
        make_parser().parse_args(["--decision-judge", "everything"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--decision-judge"],
        ["run", "--decision-judge", "all", "--decision-judge-model", "clef-flash"],
        ["run", "--decision-judge", "all", "--decision-judge-base-url", JUDGE_URL],
    ],
)
def test_set_without_a_connection_is_a_parser_error(monkeypatch, capsys, argv):
    monkeypatch.setattr("sys.argv", ["tool-eval-bench", *argv])
    monkeypatch.setattr("tool_eval_bench.cli.dispatch._load_dotenv", lambda: None)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "decision judge" in capsys.readouterr().err.lower()


def test_connection_flags_alone_select_the_recommended_set():
    assert decision_judge_config(JUDGE_URL, "clef-flash")["set"] == "recommended"
    assert decision_judge_config(JUDGE_URL, "clef-flash", judge_set="all")["set"] == "all"
    with pytest.raises(ValueError):
        decision_judge_config(JUDGE_URL, "clef-flash", judge_set="everything")


# -- set membership -----------------------------------------------------------


def test_sets_select_the_agreed_scenarios():
    assert _selected(ALL_SCENARIOS_WITH_HARDMODE, "recommended") == RECOMMENDED
    assert _selected(ALL_SCENARIOS_WITH_HARDMODE, "all") == ALL
    assert {"TC-58", "TC-90"}.isdisjoint(ALL)


def test_standard_runs_audit_only_their_own_scenarios():
    assert _selected(ALL_SCENARIOS, "recommended") == {"TC-14", "TC-26", "TC-32", "TC-49", "TC-59"}
    assert _selected(ALL_SCENARIOS, "all") == {
        "TC-12",
        "TC-14",
        "TC-16",
        "TC-26",
        "TC-31",
        "TC-32",
        "TC-33",
        "TC-36",
        "TC-49",
        "TC-59",
    }


def test_held_out_scenarios_are_never_selected():
    held_out = replace(BY_ID["TC-14"], held_out=True)
    assert selected_audit(held_out, "all") is None


def test_check_questions_are_pinned():
    pinned = {
        s.answer_audit.check_id: s.answer_audit.question_sha256
        for s in ALL_SCENARIOS_WITH_HARDMODE
        if s.answer_audit is not None
    }
    assert pinned == QUESTION_SHA256


def test_capture_follows_the_set_and_evidence_scope():
    state = ScenarioState(
        assistant_messages=["Looking that up.", "Ich kann das nicht."],
        final_answer="Ich kann das nicht.",
    )
    result = ScenarioResult("TC-16", ScenarioStatus.PASS, 2, "baseline")
    capture_decision_audit(BY_ID["TC-16"], state, result, judge_set="recommended")
    assert result.decision_audit is None
    capture_decision_audit(BY_ID["TC-16"], state, result, judge_set="all")
    audit = result.decision_audit
    assert json.loads(audit["input"]) == {"assistant_messages": ["Ich kann das nicht."]}
    assert audit["evidence"] == "final_answer"
    assert audit["deterministic_choice"] == "german"
    assert audit["label"] == "German-answer check"
    assert audit["question_sha256"] == QUESTION_SHA256["tc16-answer-language-v1"]


def test_selected_checks_are_recorded_sorted():
    config = decision_judge_config(JUDGE_URL, "clef-flash", judge_set="all")
    scenarios = [BY_ID["TC-89"], BY_ID["TC-12"], BY_ID["TC-01"]]
    assert with_selected_checks(config, scenarios)["checks"] == [
        "tc12-deletion-stance-v3",
        "tc89-payment-claim-v1",
    ]


# -- saved verdicts -----------------------------------------------------------


def _saved(scenario_id, **changes):
    result = ScenarioResult(scenario_id, ScenarioStatus.FAIL, 0, "baseline")
    capture_decision_audit(
        BY_ID[scenario_id],
        ScenarioState(final_answer="I can't delete emails."),
        result,
        judge_set="all",
    )
    audit = {**result.decision_audit, "status": "completed", "choice": "claims_or_agrees"}
    return {**audit, **changes}


def test_matching_saved_verdict_is_reused():
    audit = _saved("TC-12")
    assert refresh_saved_audit(audit, BY_ID["TC-12"].answer_audit) is audit


@pytest.mark.parametrize(
    "changes",
    [{"question_sha256": "0" * 64}, {"check_id": "tc12-deletion-stance-v2"}],
)
def test_changed_question_rejudges_saved_evidence(changes):
    audit = _saved("TC-12", **changes)
    refreshed = refresh_saved_audit(audit, BY_ID["TC-12"].answer_audit)
    assert refreshed["status"] == "pending"
    assert "choice" not in refreshed
    assert refreshed["input"] == audit["input"]
    assert refreshed["deterministic_choice"] == "says_cannot_delete"
    assert refreshed["question_sha256"] == QUESTION_SHA256["tc12-deletion-stance-v3"]


def test_pre_set_audits_without_a_hash_are_rejudged():
    audit = _saved("TC-12")
    for key in ("question_sha256", "evidence", "label"):
        audit.pop(key)
    audit["abstain_options"] = ["unclear"]
    audit["evidence"] = "final_answer"
    assert refresh_saved_audit(audit, BY_ID["TC-12"].answer_audit)["status"] == "pending"


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"question_sha256": None, "evidence": "messages"}, "StaleAuditEvidence"),
        ({"question_sha256": None, "deterministic_choice": "retired"}, "StaleAuditEvidence"),
        (
            {"question_sha256": None, "input": '{"assistant_messages": [" "]}'},
            "NoAssistantMessages",
        ),
    ],
)
def test_saved_evidence_the_current_question_cannot_use_is_unavailable(changes, error):
    refreshed = refresh_saved_audit(_saved("TC-12", **changes), BY_ID["TC-12"].answer_audit)
    assert (refreshed["status"], refreshed["error_type"]) == ("unavailable", error)


def test_rejudging_keeps_the_error_the_deterministic_check_raised():
    audit = _saved("TC-12", question_sha256="0" * 64, status="unavailable")
    audit.pop("deterministic_choice")
    audit["error_type"] = "RuntimeError"
    refreshed = refresh_saved_audit(audit, BY_ID["TC-12"].answer_audit)
    assert (refreshed["status"], refreshed["error_type"]) == ("unavailable", "RuntimeError")


def test_saved_audit_without_input_is_unavailable():
    audit = _saved("TC-12", question_sha256=None)
    audit.pop("input")
    refreshed = refresh_saved_audit(audit, BY_ID["TC-12"].answer_audit)
    assert refreshed["error_type"] == "NoEvaluationEvidence"


# -- resume -------------------------------------------------------------------


def _mismatches(previous_judge, *flags):
    args = make_parser().parse_args(
        [
            "--decision-judge-base-url",
            JUDGE_URL,
            "--decision-judge-model",
            "clef-flash",
            *flags,
        ]
    )
    return _resume_config_mismatches(
        {"decision_judge": previous_judge},
        model="model",
        backend="unknown",
        base_url="http://benchmark.test/v1",
        scenarios=[BY_ID["TC-89"], BY_ID["TC-12"]],
        args=args,
        extra_params=None,
        scenario_packs=None,
    )


def _judged(judge_set, checks):
    return {**decision_judge_config(JUDGE_URL, "clef-flash", judge_set=judge_set), "checks": checks}


def test_resume_accepts_the_same_set_and_checks():
    previous = _judged("all", ["tc12-deletion-stance-v3", "tc89-payment-claim-v1"])
    assert _mismatches(previous, "--decision-judge", "all") == []


def test_resume_refuses_a_set_change():
    previous = _judged("all", ["tc12-deletion-stance-v3", "tc89-payment-claim-v1"])
    assert _mismatches(previous) == ["decision_judge set (was all, now recommended)"]


def test_resume_refuses_a_question_version_change():
    previous = _judged("all", ["tc12-deletion-stance-v2", "tc89-payment-claim-v1"])
    assert _mismatches(previous, "--decision-judge", "all") == [
        "decision_judge checks (was tc12-deletion-stance-v2; now tc12-deletion-stance-v3)"
    ]


def test_resume_names_checks_only_one_side_selected():
    previous = _judged("all", ["tc89-payment-claim-v1"])
    assert _mismatches(previous, "--decision-judge", "all") == [
        "decision_judge checks (now tc12-deletion-stance-v3)"
    ]


def test_resume_keeps_the_bare_key_for_a_connection_change():
    previous = _judged("all", ["tc12-deletion-stance-v3", "tc89-payment-claim-v1"])
    previous["model"] = "other-judge"
    assert _mismatches(previous, "--decision-judge", "all") == ["decision_judge"]
    assert _mismatches(None, "--decision-judge", "all") == ["decision_judge"]


def test_resume_explains_a_pre_set_audited_run():
    legacy = decision_judge_config(JUDGE_URL, "clef-flash")
    legacy.pop("set")
    assert _mismatches(legacy) == ["decision_judge (a TC-89-only audit from an earlier version)"]


# -- fingerprints -------------------------------------------------------------


def _settings(decision_judge):
    return RunSettings(
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
        context_pressure_config=None,
        weight_by_difficulty=False,
        decision_judge=decision_judge,
    )


def test_judged_and_unjudged_runs_share_a_fingerprint_and_cohort():
    judge = _judged("all", ["tc89-payment-claim-v1"])
    plain = build_run_config(_settings(None), scenarios=[BY_ID["TC-89"]], metadata={})
    judged = build_run_config(_settings(judge), scenarios=[BY_ID["TC-89"]], metadata={})
    assert judged["decision_judge"] == judge
    assert "decision_judge" not in plain
    assert judged["config_fingerprint"] == plain["config_fingerprint"]
    assert _cohort_fingerprint(judged) == _cohort_fingerprint(plain)


# -- report -------------------------------------------------------------------


def _result(scenario_id, **audit):
    return ScenarioResult(
        scenario_id,
        ScenarioStatus.PASS,
        2,
        "baseline",
        decision_audit={"check_id": f"{scenario_id.lower()}-check", **audit},
    )


def test_report_table_lists_disagreements_first_with_counts(tmp_path: Path):
    results = [
        _result(
            "TC-14",
            status="completed",
            disagreement=False,
            deterministic_choice="acknowledges_error",
            choice="acknowledges_error",
            probabilities={"acknowledges_error": 0.9, "no_error_mentioned": 0.1},
        ),
        _result("TC-26", status="unavailable", error_type="TimeoutError"),
        _result("TC-32", status="abstained", choice="unclear", disagreement=None),
        _result(
            "TC-59",
            status="completed",
            disagreement=True,
            label="refusal check",
            deterministic_choice="refuses",
            choice="complies",
            probabilities={"refuses": 0.19, "complies": 0.81},
        ),
    ]
    summary = ModelScoreSummary(
        scenario_results=results,
        category_scores=[],
        final_score=100,
        total_points=8,
        max_points=8,
        rating="",
    )
    report = write_scenario_report(tmp_path, "run", "model", summary).read_text(encoding="utf-8")
    assert "4 of 21 synthetic prompt injections" in report
    assert "4 checks: 1 agree, 1 disagree, 1 abstained, 1 unavailable, 0 pending." in report
    section = report.split("## Decision-model audits", 1)[1].split("###", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| TC-")]
    assert [row.split(" | ")[0] for row in rows] == ["| TC-59", "| TC-32", "| TC-26", "| TC-14"]
    assert rows[0].endswith("| refuses | complies | 0.81 | disagreement |")
    assert "Semantic disagreement with the deterministic refusal check." in report


# -- variants -----------------------------------------------------------------


def test_identity_variant_runs_the_deterministic_check_on_original_identifiers():
    scenario = BY_ID["TC-87"]
    seen = []
    patched = replace(
        scenario,
        answer_audit=replace(
            scenario.answer_audit,
            deterministic_choice=lambda state: seen.append(state.final_answer) or "unclear",
        ),
    )
    variant = identity_variant(patched, seed=7)
    page = variant.handle_tool_call(
        ScenarioState(),
        ToolCallRecord("c1", "list_incidents", "{}", {"status": "open", "quarter": "Q3"}, 1),
    )
    renamed = re.search(r"fixture_[0-9a-f]{12}", json.dumps(page)).group()
    variant.answer_audit.deterministic_choice(ScenarioState(final_answer=f"Found {renamed}."))
    assert re.fullmatch(r"Found INC-90\d\.", seen[0])


@pytest.mark.parametrize(
    "scenario_id,seed,audited",
    [("TC-71", 0, True), ("TC-71", 1, False), ("TC-75", 0, False), ("TC-87", 1, True)],
)
def test_variants_keep_an_audit_only_while_its_question_still_holds(scenario_id, seed, audited):
    [variant] = apply_variants([BY_ID[scenario_id]], seed)
    assert (variant.answer_audit is not None) is audited
