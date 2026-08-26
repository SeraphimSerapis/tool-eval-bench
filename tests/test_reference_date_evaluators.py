"""``--reference-date`` must reach the evaluators, not just the prompt.

The system prompt tells the model what "today" is, and the orchestrator puts the
same date in ``ScenarioState.meta``.  Every scenario whose prompt says "tomorrow"
or "next Tuesday" has to derive its expected date from there.  When it hard-codes
one instead, the flag grades a correct answer against a date the model was never
given, which is a deterministic false negative rather than a flaky one.

Each scenario below is run against two reference dates: the benchmark default and
a mid-August date that lands on a different weekday and inside EU summer time.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_state
from tool_eval_bench.domain.scenarios import ScenarioState, ScenarioStatus, ToolCallRecord
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE

# (reference date, expected target date per scenario).  Written out rather than
# derived so a wrong helper cannot agree with a wrong test.
MARCH = "2026-03-20"  # Friday, winter time
AUGUST = "2026-08-19"  # Wednesday, summer time

EXPECTED: dict[str, dict[str, str]] = {
    "TC-05": {MARCH: "2026-03-23", AUGUST: "2026-08-24"},  # next Monday
    "TC-08": {MARCH: "2026-03-21", AUGUST: "2026-08-20"},  # tomorrow
    "TC-17": {MARCH: "2026-03-24", AUGUST: "2026-08-25"},  # next Tuesday
    "TC-74": {MARCH: "2026-03-25", AUGUST: "2026-08-26"},  # next Tuesday, moved to Wednesday
    "TC-79": {MARCH: "2026-03-21", AUGUST: "2026-08-20"},  # tomorrow
    "TC-84": {MARCH: "2026-03-25", AUGUST: "2026-08-26"},  # next Wednesday
}

REFERENCES = [MARCH, AUGUST]


def _scenario(sid: str):
    return next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == sid)


def _other_date(sid: str, reference: str) -> str:
    """The date this scenario expects under the *other* reference date.

    A plausible-but-wrong date, which is what a hard-coded evaluator produces.
    """
    return next(v for k, v in EXPECTED[sid].items() if k != reference)


def _simulated(sid: str, reference: str, calls: list[tuple[str, dict, int]]) -> ScenarioState:
    """Replay ``calls`` through the scenario's own tool simulator."""
    scenario = _scenario(sid)
    state = ScenarioState()
    state.meta["reference_date"] = reference
    for name, args, turn in calls:
        call = ToolCallRecord(f"{name}_{turn}", name, str(args), args, turn)
        scenario.handle_tool_call(state, call)
        state.tool_calls.append(call)
    return state


# ---------------------------------------------------------------------------
# Single-call scenarios
# ---------------------------------------------------------------------------


def _tc05_calls(date: str) -> list[dict]:
    return [
        {
            "name": "create_calendar_event",
            "arguments": {
                "title": "Team Standup",
                "date": date,
                "time": "09:30",
                "duration_minutes": 30,
                "attendees": ["alex@company.com", "jamie@company.com"],
            },
            "turn": 1,
        }
    ]


def _tc08_calls(date: str) -> list[dict]:
    return [
        {"name": "get_weather", "arguments": {"location": "Paris"}, "turn": 1},
        {
            "name": "set_reminder",
            "arguments": {"message": "Bring an umbrella", "datetime": f"{date}T08:00:00"},
            "turn": 2,
        },
    ]


def _tc17_calls(date: str, timezone: str = "Europe/Berlin") -> list[dict]:
    return [
        {
            "name": "create_calendar_event",
            "arguments": {
                "title": "Team Standup",
                "date": date,
                "time": "14:00",
                "timezone": timezone,
            },
            "turn": 1,
        }
    ]


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize(("sid", "builder"), [("TC-05", _tc05_calls), ("TC-08", _tc08_calls)])
def test_relative_date_scenarios_follow_the_reference_date(sid, builder, reference):
    expected = EXPECTED[sid][reference]
    scenario = _scenario(sid)

    right = make_state(tool_calls=builder(expected), meta={"reference_date": reference})
    assert scenario.evaluate(right).status == ScenarioStatus.PASS

    wrong = make_state(
        tool_calls=builder(_other_date(sid, reference)), meta={"reference_date": reference}
    )
    assert scenario.evaluate(wrong).status != ScenarioStatus.PASS


@pytest.mark.parametrize("reference", REFERENCES)
def test_tc17_follows_the_reference_date(reference):
    expected = EXPECTED["TC-17"][reference]
    scenario = _scenario("TC-17")

    right = make_state(tool_calls=_tc17_calls(expected), meta={"reference_date": reference})
    assert scenario.evaluate(right).status == ScenarioStatus.PASS

    wrong = make_state(
        tool_calls=_tc17_calls(_other_date("TC-17", reference)),
        meta={"reference_date": reference},
    )
    assert scenario.evaluate(wrong).status != ScenarioStatus.PASS


@pytest.mark.parametrize(
    ("reference", "accepted", "rejected"),
    [(MARCH, "CET", "CEST"), (AUGUST, "CEST", "CET")],
)
def test_tc17_offset_alias_tracks_summer_time(reference, accepted, rejected):
    """ "CET" is only a synonym for Europe/Berlin outside summer time."""
    scenario = _scenario("TC-17")
    expected = EXPECTED["TC-17"][reference]

    ok = make_state(tool_calls=_tc17_calls(expected, accepted), meta={"reference_date": reference})
    assert scenario.evaluate(ok).status == ScenarioStatus.PASS

    off = make_state(tool_calls=_tc17_calls(expected, rejected), meta={"reference_date": reference})
    assert scenario.evaluate(off).status == ScenarioStatus.PARTIAL


# ---------------------------------------------------------------------------
# Multi-step scenarios
# ---------------------------------------------------------------------------


def _tc74_calls(date: str) -> list[dict]:
    return [
        {"name": "get_contacts", "arguments": {"query": "Sarah"}, "turn": 1},
        {
            "name": "create_calendar_event",
            "arguments": {
                "title": "Product Review",
                "date": date,
                "time": "14:00",
                "duration_minutes": 45,
                "attendees": ["mark.chen@company.com", "sarah.jones@company.com"],
            },
            "turn": 2,
        },
        {
            "name": "send_email",
            "arguments": {
                "to": "mark.chen@company.com,sarah.jones@company.com",
                "subject": "Confirmed",
                "body": "Product Review.",
            },
            "turn": 3,
        },
    ]


@pytest.mark.parametrize("reference", REFERENCES)
def test_tc74_correction_date_follows_the_reference_date(reference):
    """The third follow-up moves the event to the day after "next Tuesday"."""
    scenario = _scenario("TC-74")
    expected = EXPECTED["TC-74"][reference]

    def evaluate(date: str):
        calls = _tc74_calls(date)
        for call in calls:
            call["user_phase"] = 4
        return scenario.evaluate(make_state(tool_calls=calls, meta={"reference_date": reference}))

    assert evaluate(expected).status == ScenarioStatus.PASS
    assert evaluate(_other_date("TC-74", reference)).status != ScenarioStatus.PASS


def _tc79_calls(date: str) -> list[tuple[str, dict, int]]:
    return [
        ("get_weather", {"location": "Lisbon"}, 1),
        ("get_contacts", {"query": "Priya Shah"}, 2),
        (
            "create_calendar_event",
            {
                "title": "Outdoor review",
                "date": date,
                "time": "09:00",
                "timezone": "Europe/Lisbon",
                "duration_minutes": 30,
                "attendees": ["priya.shah@company.com"],
            },
            3,
        ),
    ]


@pytest.mark.parametrize("reference", REFERENCES)
def test_tc79_follows_the_reference_date(reference):
    scenario = _scenario("TC-79")
    expected = EXPECTED["TC-79"][reference]

    right = _simulated("TC-79", reference, _tc79_calls(expected))
    assert scenario.evaluate(right).status == ScenarioStatus.PASS

    wrong = _simulated("TC-79", reference, _tc79_calls(_other_date("TC-79", reference)))
    assert scenario.evaluate(wrong).status != ScenarioStatus.PASS


def _tc84_calls(date: str) -> list[tuple[str, dict, int]]:
    attendees = ["elena@company.com", "ravi@company.com"]
    booking = {"date": date, "time": "14:00", "duration_minutes": 45, "attendees": attendees}
    return [
        ("get_contacts", {"query": "Elena and Ravi"}, 1),
        ("search_slots", {"date": date, "period": "afternoon", "duration_minutes": 45}, 2),
        ("search_rooms", {"office": "Berlin", "minimum_capacity": 3}, 3),
        ("search_files", {"query": "agenda"}, 4),
        ("book_room", {"room_id": "berlin_3a", **booking}, 5),
        ("book_room", {"room_id": "berlin_5b", **booking}, 6),
        (
            "send_email",
            {
                "to": "elena@company.com,ravi@company.com",
                "subject": "Review booked",
                "body": "Booked.",
                "attachments": ["agenda_q2"],
            },
            7,
        ),
    ]


@pytest.mark.parametrize("reference", REFERENCES)
def test_tc84_follows_the_reference_date(reference):
    scenario = _scenario("TC-84")
    expected = EXPECTED["TC-84"][reference]

    right = _simulated("TC-84", reference, _tc84_calls(expected))
    assert scenario.evaluate(right).status == ScenarioStatus.PASS

    wrong = _simulated("TC-84", reference, _tc84_calls(_other_date("TC-84", reference)))
    assert scenario.evaluate(wrong).status != ScenarioStatus.PASS


@pytest.mark.parametrize("reference", REFERENCES)
def test_tc84_simulator_offers_the_date_the_prompt_asked_for(reference):
    """The slot the simulator returns has to match the date the evaluator wants.

    Before the fix the simulator always offered 2026-03-25, so under any other
    reference date it handed the model a slot that could not score.
    """
    scenario = _scenario("TC-84")
    state = ScenarioState()
    state.meta["reference_date"] = reference
    call = ToolCallRecord("search_slots_1", "search_slots", "{}", {}, 1)
    payload = scenario.handle_tool_call(state, call)

    assert payload["slots"][0]["date"] == EXPECTED["TC-84"][reference]


def test_offset_aliases_need_the_timezone_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without tzdata the check silently accepts both DST spellings.

    Windows ships no IANA database, so `ZoneInfo` raises there unless the
    `tzdata` package is installed. The fallback then treats "cet" and "cest" as
    equally correct year-round, which scores TC-17 PASS where it should score
    PARTIAL. `pyproject.toml` declares `tzdata` on win32 for exactly this
    reason; this test says what breaks if that declaration is dropped.
    """
    import zoneinfo

    from tool_eval_bench.evals import helpers

    summer_only = helpers.utc_offset_aliases("2026-08-19", "Europe/Berlin")
    assert summer_only == {"cest", "utc+2"}, "with a database, only summer spellings are correct"

    def no_database(key: str) -> None:
        raise zoneinfo.ZoneInfoNotFoundError(key)

    monkeypatch.setattr(helpers, "ZoneInfo", no_database)
    without = helpers.utc_offset_aliases("2026-08-19", "Europe/Berlin")

    assert without == {"cet", "cest", "utc+1", "utc+2"}
    assert without > summer_only, "the fallback is strictly more permissive"
