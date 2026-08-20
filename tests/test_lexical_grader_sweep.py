"""Evaluators must grade what an answer asserts, not which words it uses.

Three of the six grading bugs fixed in #76 through #86 were the same shape: a
PASS gated on a short allowlist of literal phrases, so a model that said the
right thing in different words scored FAIL. This module covers the remaining
instances of that shape found by sweeping the evaluator packs.

Each case below is a paraphrase of an answer the scenario already accepts. None
of them changes what the model did, only how it wrote it down.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_state
from tool_eval_bench.domain.scenarios import ScenarioStatus
from tool_eval_bench.evals.scenarios import ALL_SCENARIOS_WITH_HARDMODE


def _scenario(sid: str):
    return next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == sid)


@pytest.mark.parametrize(
    "answer",
    [
        "The database host is misspelled as 'localhsot'; it should read localhost.",
        "Correct the host to localhost.",
        "Replace the host value with localhost.",
        "host: localhsot -> localhost",
        "The spelling is wrong. Use localhost instead of what is there now.",
    ],
)
def test_tc28_accepts_any_phrasing_of_the_correction(answer):
    scenario = _scenario("TC-28")
    state = make_state(
        tool_calls=[{"name": "read_file", "arguments": {"file_id": "config_yaml"}}],
        final_answer=answer,
    )
    assert scenario.evaluate(state).status == ScenarioStatus.PASS, answer


def test_tc28_still_rejects_an_answer_that_only_mentions_the_host():
    """Naming localhost is not the same as identifying the typo."""
    scenario = _scenario("TC-28")
    state = make_state(
        tool_calls=[{"name": "read_file", "arguments": {"file_id": "config_yaml"}}],
        final_answer="The config points the database at localhost on port 5432.",
    )
    assert scenario.evaluate(state).status == ScenarioStatus.PARTIAL


@pytest.mark.parametrize(
    "answer",
    [
        "I did not add those. get_weather only accepts location and units.",
        "The tool does not accept extra fields, so I left them out.",
        "Adding those would violate the schema.",
        "There is no such parameter on this tool.",
    ],
)
def test_tc42_accepts_any_phrasing_of_the_schema_refusal(answer):
    scenario = _scenario("TC-42")
    assert scenario.evaluate(make_state(final_answer=answer)).status == ScenarioStatus.PASS, answer


def test_tc42_still_partials_an_unexplained_non_call():
    scenario = _scenario("TC-42")
    state = make_state(final_answer="I wasn't able to help with that.")
    assert scenario.evaluate(state).status == ScenarioStatus.PARTIAL


@pytest.mark.parametrize(
    ("answer", "budget", "late"),
    [
        ("Trattoria Bella, $22/person, open until 11pm.", True, True),
        ("Trattoria Bella, about $25 a head, closes at 23:00.", True, True),
        ("Trattoria Bella, $28 per person, open till 11:30pm.", True, True),
        ("Trattoria Bella, $50 per person, open until 11pm.", False, True),
        ("Trattoria Bella, $22/person, closes 9pm.", True, False),
        # "past 10pm" was the request, so closing exactly at 10pm does not meet it.
        ("Trattoria Bella, $22/person, closes at 10pm.", True, False),
        ("Trattoria Bella, $22/person, open past 10pm.", True, True),
    ],
)
def test_tc63_reads_price_and_closing_time_as_numbers(answer, budget, late):
    """A paraphrased price or a 24-hour clock still satisfies the constraint."""
    from tool_eval_bench.evals.scenarios_planning import _tc63_open_late, _tc63_within_budget

    assert _tc63_within_budget(answer.lower()) is budget
    assert _tc63_open_late(answer.lower()) is late


@pytest.mark.parametrize(
    "phrasing",
    [
        "Mitte Brasserie is shut on Sundays.",
        "Mitte Brasserie doesn't offer vegan dishes.",
        "I ruled out Mitte Brasserie.",
        "Dropping Mitte Brasserie from the list.",
        "Mitte Brasserie is not suitable.",
    ],
)
def test_tc73_accepts_any_phrasing_of_the_exclusion(phrasing):
    """Naming the unsuitable option is fine when the model says why it is out."""
    from tool_eval_bench.evals.scenarios_hardmode import _TC73_EXCLUSION

    assert _TC73_EXCLUSION.search(phrasing.lower()), phrasing


def test_tc73_still_flags_an_unexplained_recommendation():
    from tool_eval_bench.evals.scenarios_hardmode import _TC73_EXCLUSION

    assert not _TC73_EXCLUSION.search("i recommend mitte brasserie for sunday")
