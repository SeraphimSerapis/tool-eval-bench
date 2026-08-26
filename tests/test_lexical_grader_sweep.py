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
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_open_late, _tc63_within_budget

    assert _tc63_within_budget(answer.lower()) is budget
    assert _tc63_open_late(answer.lower()) is late


@pytest.mark.parametrize(
    "closing",
    ["11pm", "11 PM", "11 p.m.", "11 P.M.", "11:30 p.m.", "11:30pm", "23:00"],
)
def test_tc63_reads_a_late_closing_time_in_every_spelling(closing):
    """The periods in "11 p.m." are punctuation, not a different time.

    `_TC63_PAST_CUTOFF` and TC-03's clock reader both accept them already, so an
    answer that named a qualifying closing time in the most ordinary English
    spelling lost the constraint in this evaluator alone.
    """
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_open_late

    answer = f"Trattoria Bella, downtown, $22 per person, open until {closing}."
    assert _tc63_open_late(answer.lower()) is True, closing


@pytest.mark.parametrize("closing", ["closes at 10pm", "closes at 10 p.m.", "closes at 22:00"])
def test_tc63_still_rejects_a_closing_time_that_is_not_past_the_cutoff(closing):
    """The request was for somewhere open *past* 10pm; 22:00 exactly is not."""
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_open_late

    answer = f"Trattoria Bella, downtown, $22 per person, {closing}."
    assert _tc63_open_late(answer.lower()) is False, closing


@pytest.mark.parametrize(
    "phrase",
    [
        "trattoria bella has 2 pms on staff",
        "the 1 amount shown is per person",
        "the dining room is 40 sq.m. and seats 20",
        "table for 12 among the regulars",
    ],
)
def test_tc63_does_not_read_ordinary_prose_as_a_closing_time(phrase):
    """A looser meridiem must not turn a stray letter after a number into a clock."""
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_open_late

    assert _tc63_open_late(phrase) is False, phrase


@pytest.mark.parametrize("closing", ["11pm", "11 p.m.", "23:00"])
def test_tc63_scores_the_same_recommendation_the_same_way(closing):
    """Spelling the closing hour differently is not dropping a constraint."""
    scenario = _scenario("TC-63")
    state = make_state(
        tool_calls=[
            {
                "name": "web_search",
                "arguments": {"query": "italian restaurant downtown under $30 open late"},
            }
        ],
        final_answer=f"Trattoria Bella - Italian, downtown, $22 per person, open until {closing}.",
    )
    assert scenario.evaluate(state).status == ScenarioStatus.PASS, closing


@pytest.mark.parametrize("guess", ["3pm", "15:00", "3:00 p.m.", "3 p.m.", "3 P.M."])
def test_tc75_spots_a_guessed_time_in_every_spelling(guess):
    """Asking for the time while naming one is a guess in any spelling.

    `3 p.m.` matched no concrete-value form, so the answer scored PASS under a
    verdict that reads "without guessing" — about an answer that guessed.
    """
    scenario = _scenario("TC-75")
    state = make_state(
        final_answer=(
            f"I have pencilled the panel in for {guess}. "
            "Could you please provide the date and time you want?"
        )
    )
    assert scenario.evaluate(state).status == ScenarioStatus.PARTIAL, guess


def test_tc75_still_passes_an_answer_that_names_no_time_at_all():
    """The guard must not fire on the clean clarification it is meant to pass."""
    scenario = _scenario("TC-75")
    state = make_state(
        final_answer="Could you please provide the date and time you want for the panel?"
    )
    assert scenario.evaluate(state).status == ScenarioStatus.PASS


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
    from tool_eval_bench.evals.scenarios.hardmode.tc73 import _TC73_EXCLUSION

    assert _TC73_EXCLUSION.search(phrasing.lower()), phrasing


def test_tc73_still_flags_an_unexplained_recommendation():
    from tool_eval_bench.evals.scenarios.hardmode.tc73 import _TC73_EXCLUSION

    assert not _TC73_EXCLUSION.search("i recommend mitte brasserie for sunday")
