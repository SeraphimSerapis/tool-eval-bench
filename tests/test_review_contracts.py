"""Runner-level reproductions of REVIEW.MD, with positive and negative controls."""

import json

import pytest
from scenario_replay import SCENARIOS, replay, turn

from tool_eval_bench.domain.scenarios import ScenarioStatus


def test_tc60_send_requires_user_authorization():
    weather = turn(("get_weather", {"location": "Paris"}))
    email = turn(("send_email", {"to": "sarah@company.com", "subject": "Hello", "body": "Hello!"}))
    assert (
        replay("TC-60", weather, turn(answer="Paris is rainy."), email, turn(answer="Sent.")).status
        is ScenarioStatus.PASS
    )
    assert (
        replay("TC-60", weather, email, turn(answer="Sent."), turn(answer="Already sent.")).status
        is ScenarioStatus.FAIL
    )


@pytest.mark.parametrize(
    "scenario,calls,answer",
    [
        (
            "TC-40",
            [
                ("get_customer_profile", {"customer_id": "Sarah Chen"}),
                ("get_order_status", {"order_id": "ORD-2026-1847"}),
            ],
            "The order has shipped.",
        ),
        (
            "TC-61",
            [
                (
                    "run_code",
                    {"code": 'analyze_data(source="transactions_2026")', "language": "python"},
                ),
                ("run_code", {"code": 'check_job_status("job_tc61_9f3a")', "language": "python"}),
            ],
            "Found 3 anomalies in 15,420 records.",
        ),
    ],
)
def test_dependencies_require_an_observed_result(scenario, calls, answer):
    assert (
        replay(scenario, *(turn(call) for call in calls), turn(answer=answer)).status
        is ScenarioStatus.PASS
    )
    assert replay(scenario, turn(*calls), turn(answer=answer)).status is not ScenarioStatus.PASS


def test_tc40_requires_order_status_in_answer():
    assert (
        replay(
            "TC-40",
            turn(("get_order_status", {"order_id": "Sarah Chen"})),
            turn(answer="Your order includes a keyboard."),
        ).status
        is not ScenarioStatus.PASS
    )


@pytest.mark.parametrize(
    "expression,expected", [("3*120+2*85+5*81", 935), ("1+1", 2), ("broken", None)]
)
def test_tc78_calculator_evaluates_expression(expression, expected):
    from tool_eval_bench.domain.scenarios import ScenarioState, ToolCallRecord

    call = ToolCallRecord(
        id="calc",
        turn=1,
        name="calculator",
        raw_arguments=json.dumps({"expression": expression}),
        arguments={"expression": expression},
    )
    result = SCENARIOS["TC-78"].handle_tool_call(ScenarioState(), call)
    if expected is None:
        assert "error" in result
    else:
        assert result["result"] == expected


@pytest.mark.parametrize(
    "answer,passes",
    [
        ("Es sind 14 Grad.", True),
        ("München hat aktuell 14°C bei bewölktem Himmel.", True),
        (
            "Das Wetter in München ist nicht bewölkt. Die Temperatur beträgt nicht 14 Grad Celsius, sondern 30 Grad.",
            False,
        ),
        ("The temperature in München is 14 degrees Celsius.", False),
    ],
)
def test_tc16_language_and_factual_contract(answer, passes):
    result = replay(
        "TC-16",
        turn(("get_weather", {"location": "München", "units": "celsius"})),
        turn(answer=answer),
    )
    assert (result.status is ScenarioStatus.PASS) is passes


@pytest.mark.parametrize(
    "wrapper", ["{}", "```json\n{}\n```", "Here is the review:\n```json\n{}\n```\nThanks!"]
)
def test_tc64_requires_raw_json(wrapper):
    movie = json.dumps(
        dict(
            title="The Matrix",
            year=1999,
            rating=9,
            genre="sci-fi",
            summary="A hacker discovers a simulated world.",
        )
    )
    assert (replay("TC-64", turn(answer=wrapper.format(movie))).status is ScenarioStatus.PASS) is (
        wrapper == "{}"
    )


def test_tc21_requires_all_explicit_constraints():
    assert "150" in SCENARIOS["TC-21"].user_message
    assert (
        replay(
            "TC-21",
            turn(answer="email: invalid. age: out of range. phone: invalid. date: invalid."),
        ).status
        is ScenarioStatus.PARTIAL
    )
    assert (
        replay(
            "TC-21",
            turn(
                answer="The email john@.com is malformed. The date 2020-13-45 is invalid. What schema defines the permitted age range, phone format, and whether a negative amount represents a refund?"
            ),
        ).status
        is not ScenarioStatus.PASS
    )


def test_tc61_wrong_source_is_not_submitted():
    result = replay(
        "TC-61",
        turn(("run_code", {"code": 'analyze_data(source="unrelated")', "language": "python"})),
        turn(("run_code", {"code": 'check_job_status("job_tc61_9f3a")', "language": "python"})),
        turn(answer="Found 3 anomalies in 15,420 records."),
    )
    assert result.status is not ScenarioStatus.PASS


@pytest.mark.parametrize("units", ["bananas", 1, None])
def test_tc01_rejects_invalid_units(units):
    assert (
        replay(
            "TC-01",
            turn(("get_weather", {"location": "Berlin", "units": units})),
            turn(answer="Berlin is 8 degrees and overcast."),
        ).status
        is not ScenarioStatus.PASS
    )


@pytest.mark.parametrize(
    "answer,passes",
    [
        ("Tokyo is 64°F.", True),
        ("Tokyo is 64 degrees Celsius.", False),
        ("Tokyo is 64 degrees Fahrenheit.", True),
    ],
)
def test_tc04_requires_correct_answer_unit(answer, passes):
    assert (
        replay(
            "TC-04",
            turn(("get_weather", {"location": "Tokyo", "units": "fahrenheit"})),
            turn(answer=answer),
        ).status
        is ScenarioStatus.PASS
    ) is passes


@pytest.mark.parametrize(
    "answer", ["Es sind nicht **14** Grad in München.", "Es sind 14°F in München."]
)
def test_tc16_rejects_formatted_negation_and_wrong_units(answer):
    result = replay("TC-16", turn(("get_weather", {"location": "München"})), turn(answer=answer))
    assert result.status is not ScenarioStatus.PASS
