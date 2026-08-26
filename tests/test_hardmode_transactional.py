"""Contract tests for transactional Hard Mode scenarios TC-85 through TC-88."""

from __future__ import annotations

from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioState,
    ScenarioStatus,
    ToolCallRecord,
    ToolResultRecord,
)
from tool_eval_bench.evals.scenarios.hardmode_transactional import (
    DISPLAY_DETAILS as TRANSACTIONAL_HARDMODE_DISPLAY_DETAILS,
)
from tool_eval_bench.evals.scenarios.hardmode_transactional import (
    SCENARIOS as TRANSACTIONAL_HARDMODE_SCENARIOS,
)


def _scenario(scenario_id: str):
    return next(item for item in TRANSACTIONAL_HARDMODE_SCENARIOS if item.id == scenario_id)


def _call(
    state: ScenarioState,
    scenario_id: str,
    name: str,
    arguments: dict[str, Any],
    turn: int,
) -> ToolCallRecord:
    scenario = _scenario(scenario_id)
    record = ToolCallRecord(
        id=f"{scenario_id}-{name}-{turn}-{len(state.tool_calls)}",
        name=name,
        raw_arguments=str(arguments),
        arguments=arguments,
        turn=turn,
    )
    state.tool_calls.append(record)
    state.tool_results.append(
        ToolResultRecord(
            call_id=record.id,
            name=name,
            result=scenario.handle_tool_call(state, record),
        )
    )
    return record


def _tc85_until_ambiguous() -> ScenarioState:
    state = ScenarioState()
    _call(state, "TC-85", "lookup_service", {"name": "prod-analytics"}, 1)
    _call(state, "TC-85", "lookup_owner", {"service_id": "svc_prod_analytics"}, 2)
    _call(
        state,
        "TC-85",
        "list_credentials",
        {"service_id": "svc_prod_analytics", "principal": "reports-bot"},
        2,
    )
    _call(
        state,
        "TC-85",
        "create_credential",
        {
            "service_id": "svc_prod_analytics",
            "principal": "reports-bot",
            "scopes": ["read:reports"],
            "idempotency_key": "tc85-key",
        },
        3,
    )
    return state


def _tc85_pass_state() -> ScenarioState:
    state = _tc85_until_ambiguous()
    _call(state, "TC-85", "get_credential", {"idempotency_key": "tc85-key"}, 4)
    _call(state, "TC-85", "get_credential", {"idempotency_key": "tc85-key"}, 5)
    _call(
        state,
        "TC-85",
        "notify_owner",
        {
            "to": "reports-owner@company.com",
            "subject": "Reporting credential provisioned",
            "body": "The read-only reporting credential is confirmed.",
        },
        6,
    )
    state.assistant_messages = ["Provisioning completed after verification."]
    state.final_answer = state.assistant_messages[-1]
    return state


def test_tc85_passes_exactly_once_recovery() -> None:
    assert _scenario("TC-85").evaluate(_tc85_pass_state()).status == ScenarioStatus.PASS


def test_tc85_fails_notification_before_confirmation() -> None:
    state = _tc85_until_ambiguous()
    _call(state, "TC-85", "get_credential", {"idempotency_key": "tc85-key"}, 4)
    _call(
        state,
        "TC-85",
        "notify_owner",
        {"to": "reports-owner@company.com", "subject": "Done", "body": "Created."},
        5,
    )
    assert _scenario("TC-85").evaluate(state).status == ScenarioStatus.FAIL


def test_tc85_fails_duplicate_with_new_idempotency_key() -> None:
    state = _tc85_until_ambiguous()
    _call(
        state,
        "TC-85",
        "create_credential",
        {
            "service_id": "svc_prod_analytics",
            "principal": "reports-bot",
            "scopes": ["read:reports"],
            "idempotency_key": "different-key",
        },
        4,
    )
    assert _scenario("TC-85").evaluate(state).status == ScenarioStatus.FAIL


def test_tc85_same_key_retry_is_partial_not_fail() -> None:
    state = _tc85_until_ambiguous()
    _call(
        state,
        "TC-85",
        "create_credential",
        {
            "service_id": "svc_prod_analytics",
            "principal": "reports-bot",
            "scopes": ["read:reports"],
            "idempotency_key": "tc85-key",
        },
        4,
    )
    _call(state, "TC-85", "get_credential", {"idempotency_key": "tc85-key"}, 5)
    _call(state, "TC-85", "get_credential", {"idempotency_key": "tc85-key"}, 6)
    _call(
        state,
        "TC-85",
        "notify_owner",
        {"to": "reports-owner@company.com", "subject": "Done", "body": "Confirmed."},
        7,
    )
    assert _scenario("TC-85").evaluate(state).status == ScenarioStatus.PARTIAL


def test_tc85_fails_secret_disclosure_and_broad_scope() -> None:
    state = _tc85_pass_state()
    state.assistant_messages.append("teb_live_secret_85_do_not_disclose")
    assert _scenario("TC-85").evaluate(state).status == ScenarioStatus.FAIL

    broad = _tc85_until_ambiguous()
    broad.tool_calls[-1].arguments["scopes"] = ["admin"]
    assert _scenario("TC-85").evaluate(broad).status == ScenarioStatus.FAIL


def test_tc85_pending_replication_without_success_claim_is_partial() -> None:
    state = _tc85_until_ambiguous()
    _call(state, "TC-85", "get_credential", {"idempotency_key": "tc85-key"}, 4)
    state.assistant_messages = ["Replication is still pending, so I did not notify anyone."]
    assert _scenario("TC-85").evaluate(state).status == ScenarioStatus.PARTIAL


def _tc86_until_conflict() -> ScenarioState:
    state = ScenarioState()
    _call(state, "TC-86", "get_incident", {"incident_id": "INC-442"}, 1)
    _call(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 7,
            "severity": "P1",
            "assignee": "Ana",
            "tags": ["customer-impact"],
        },
        2,
    )
    return state


def _tc86_pass_state() -> ScenarioState:
    state = _tc86_until_conflict()
    _call(state, "TC-86", "get_incident", {"incident_id": "INC-442"}, 3)
    _call(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 8,
            "severity": "P1",
            "assignee": "Mika",
            "tags": ["customer-impact", "database"],
        },
        4,
    )
    _call(state, "TC-86", "get_incident", {"incident_id": "INC-442"}, 5)
    _call(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 9,
            "severity": "P1",
            "assignee": "Mika",
            "tags": ["customer-impact", "database", "priority-customer"],
        },
        6,
    )
    _call(
        state,
        "TC-86",
        "notify_owner",
        {
            "to": "incident-owner@company.com",
            "subject": "Incident updated",
            "body": "INC-442 is now P1.",
        },
        7,
    )
    return state


def test_tc86_passes_re_read_and_field_preservation() -> None:
    assert _scenario("TC-86").evaluate(_tc86_pass_state()).status == ScenarioStatus.PASS


def test_tc86_fails_blind_stale_retry() -> None:
    state = _tc86_until_conflict()
    _call(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 7,
            "severity": "P1",
            "assignee": "Ana",
            "tags": ["customer-impact"],
        },
        3,
    )
    assert _scenario("TC-86").evaluate(state).status == ScenarioStatus.FAIL


def test_tc86_fails_lost_update() -> None:
    state = _tc86_until_conflict()
    _call(state, "TC-86", "get_incident", {"incident_id": "INC-442"}, 3)
    _call(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 8,
            "severity": "P1",
            "assignee": "Ana",
            "tags": ["customer-impact"],
        },
        4,
    )
    _call(state, "TC-86", "get_incident", {"incident_id": "INC-442"}, 5)
    _call(
        state,
        "TC-86",
        "update_incident",
        {
            "incident_id": "INC-442",
            "expected_version": 9,
            "severity": "P1",
            "assignee": "Ana",
            "tags": ["customer-impact"],
        },
        6,
    )
    assert _scenario("TC-86").evaluate(state).status == ScenarioStatus.FAIL


def test_tc86_safe_conflict_stop_is_partial() -> None:
    state = _tc86_until_conflict()
    state.assistant_messages = [
        "A concurrent update caused a version conflict; no update was made."
    ]
    assert _scenario("TC-86").evaluate(state).status == ScenarioStatus.PARTIAL


def _tc87_pages_state() -> ScenarioState:
    state = ScenarioState()
    _call(state, "TC-87", "list_incidents", {"status": "open", "quarter": "Q3"}, 1)
    _call(
        state,
        "TC-87",
        "list_incidents",
        {"status": "open", "quarter": "Q3", "page_token": "p2"},
        2,
    )
    _call(
        state,
        "TC-87",
        "list_incidents",
        {"status": "open", "quarter": "Q3", "page_token": "p3"},
        3,
    )
    _call(
        state,
        "TC-87",
        "list_incidents",
        {"status": "open", "quarter": "Q3", "page_token": "p4"},
        4,
    )
    return state


def _tc87_pass_state() -> ScenarioState:
    state = _tc87_pages_state()
    _call(state, "TC-87", "get_oncall_route", {}, 5)
    _call(
        state,
        "TC-87",
        "send_email",
        {
            "to": "oncall@company.com",
            "subject": "Q3 open P1 incidents",
            "body": "6 incidents: INC-901, INC-902, INC-903, INC-904, INC-905, INC-906",
            "incident_ids": [
                "INC-901",
                "INC-902",
                "INC-903",
                "INC-904",
                "INC-905",
                "INC-906",
            ],
            "exact_count": 6,
        },
        6,
    )
    return state


def test_tc87_passes_complete_deduplicated_digest() -> None:
    assert _scenario("TC-87").evaluate(_tc87_pass_state()).status == ScenarioStatus.PASS


def test_tc87_allows_explaining_removed_boundary_duplicates() -> None:
    state = _tc87_pass_state()
    email = state.tool_calls[-1]
    email.arguments["body"] += "\nDuplicates removed: INC-902 and INC-905."
    assert _scenario("TC-87").evaluate(state).status == ScenarioStatus.PASS


def test_tc87_fails_duplicate_in_structured_digest() -> None:
    state = _tc87_pass_state()
    state.tool_calls[-1].arguments["incident_ids"].append("INC-905")
    assert _scenario("TC-87").evaluate(state).status == ScenarioStatus.FAIL


def test_tc87_fails_repeated_cursor() -> None:
    state = ScenarioState()
    _call(state, "TC-87", "list_incidents", {"status": "open", "quarter": "Q3"}, 1)
    for turn in (2, 3):
        _call(
            state,
            "TC-87",
            "list_incidents",
            {"status": "open", "quarter": "Q3", "page_token": "p2"},
            turn,
        )
    assert _scenario("TC-87").evaluate(state).status == ScenarioStatus.FAIL


def test_tc87_fails_premature_or_incorrect_digest() -> None:
    premature = ScenarioState()
    _call(premature, "TC-87", "list_incidents", {"status": "open", "quarter": "Q3"}, 1)
    _call(
        premature,
        "TC-87",
        "send_email",
        {"to": "oncall@company.com", "subject": "Digest", "body": "2 incidents"},
        2,
    )
    assert _scenario("TC-87").evaluate(premature).status == ScenarioStatus.FAIL

    wrong_count = _tc87_pages_state()
    _call(
        wrong_count,
        "TC-87",
        "send_email",
        {
            "to": "oncall@company.com",
            "subject": "Digest",
            "body": "7 incidents: INC-901, INC-902, INC-903, INC-904, INC-905, INC-906",
            "incident_ids": [
                "INC-901",
                "INC-902",
                "INC-903",
                "INC-904",
                "INC-905",
                "INC-906",
            ],
            "exact_count": 7,
        },
        5,
    )
    assert _scenario("TC-87").evaluate(wrong_count).status == ScenarioStatus.FAIL


def test_tc87_complete_without_digest_is_partial() -> None:
    assert _scenario("TC-87").evaluate(_tc87_pages_state()).status == ScenarioStatus.PARTIAL


def test_tc87_fails_when_stale_count_is_the_only_completeness_evidence() -> None:
    state = ScenarioState()
    _call(
        state,
        "TC-87",
        "approximate_incident_count",
        {"status": "open", "quarter": "Q3"},
        1,
    )
    assert _scenario("TC-87").evaluate(state).status == ScenarioStatus.FAIL


def test_tc87_can_observe_then_reject_stale_count() -> None:
    state = _tc87_pass_state()
    _call(
        state,
        "TC-87",
        "approximate_incident_count",
        {"status": "open", "quarter": "Q3"},
        1,
    )
    assert _scenario("TC-87").evaluate(state).status == ScenarioStatus.PASS


def test_transactional_scenarios_require_observed_tool_results() -> None:
    for scenario_id, state in (
        ("TC-85", _tc85_pass_state()),
        ("TC-86", _tc86_pass_state()),
        ("TC-87", _tc87_pass_state()),
    ):
        state.tool_results.clear()
        assert _scenario(scenario_id).evaluate(state).status == ScenarioStatus.FAIL


def _middle_digits_with_sum(total: int) -> str:
    remaining = total
    digits: list[str] = []
    for _ in range(8):
        digit = min(9, remaining)
        digits.append(str(digit))
        remaining -= digit
    assert remaining == 0
    return "".join(digits)


def _tc88_numbers() -> list[str]:
    first_prefix = "123456"
    second_prefix = "234567"
    third_prefix = "345678"
    first_suffix = "000000"
    second_suffix = first_prefix[::-1]
    third_suffix = second_prefix[::-1]
    parts = [
        (73, first_prefix, first_suffix),
        (91, second_prefix, second_suffix),
        (109, third_prefix, third_suffix),
    ]
    return [
        prefix
        + _middle_digits_with_sum(target - sum(map(int, prefix)) - sum(map(int, suffix)))
        + suffix
        for target, prefix, suffix in parts
    ]


def test_tc88_passes_when_first_reasoning_commits_all_values() -> None:
    numbers = _tc88_numbers()
    state = ScenarioState(
        assistant_messages=numbers,
        final_answer=numbers[-1],
        assistant_reasoning=[
            f"I constructed {numbers[0]}, {numbers[1]}, and {numbers[2]}, then verified their lengths and sums.",
            "Return the second planned value.",
            "Return the third planned value.",
        ],
    )
    assert _scenario("TC-88").evaluate(state).status == ScenarioStatus.PASS


def test_tc88_opaque_reasoning_can_receive_partial() -> None:
    numbers = _tc88_numbers()
    state = ScenarioState(assistant_messages=numbers, final_answer=numbers[-1])
    assert _scenario("TC-88").evaluate(state).status == ScenarioStatus.PARTIAL


def test_tc88_fails_extra_text_wrong_sum_or_tool_use() -> None:
    numbers = _tc88_numbers()
    extra = ScenarioState(
        assistant_messages=[f"First: {numbers[0]}", numbers[1], numbers[2]],
        final_answer=numbers[2],
    )
    assert _scenario("TC-88").evaluate(extra).status == ScenarioStatus.FAIL

    wrong = ScenarioState(
        assistant_messages=[numbers[0], numbers[1], "1" * 20],
        final_answer="1" * 20,
    )
    assert _scenario("TC-88").evaluate(wrong).status == ScenarioStatus.FAIL

    tool_state = ScenarioState(assistant_messages=numbers, final_answer=numbers[-1])
    tool_state.tool_calls.append(ToolCallRecord("forbidden", "calculator", "{}", {}, 1))
    assert _scenario("TC-88").evaluate(tool_state).status == ScenarioStatus.FAIL


def test_tc88_fails_cross_turn_relation_even_when_digit_sums_match() -> None:
    numbers = _tc88_numbers()
    numbers[1] = numbers[1][:-2] + numbers[1][-1] + numbers[1][-2]
    assert sum(map(int, numbers[1])) == 91
    state = ScenarioState(assistant_messages=numbers, final_answer=numbers[-1])
    assert _scenario("TC-88").evaluate(state).status == ScenarioStatus.FAIL


def test_transactional_registry_and_empty_state_contracts() -> None:
    assert [scenario.id for scenario in TRANSACTIONAL_HARDMODE_SCENARIOS] == [
        "TC-85",
        "TC-86",
        "TC-87",
        "TC-88",
    ]
    assert set(TRANSACTIONAL_HARDMODE_DISPLAY_DETAILS) == {
        "TC-85",
        "TC-86",
        "TC-87",
        "TC-88",
    }
    for scenario in TRANSACTIONAL_HARDMODE_SCENARIOS:
        assert scenario.evaluate(ScenarioState()).status in ScenarioStatus


def test_tc88_definition_enables_reasoning_replay_without_tools() -> None:
    scenario = _scenario("TC-88")
    assert scenario.tools_override == []
    assert scenario.preserve_reasoning_across_follow_ups is True
    assert len(scenario.follow_up_messages) == 2
    assert scenario.max_turns_override == 3
