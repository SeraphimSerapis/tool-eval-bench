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


# Every case here FAILS at the parent revision. Cases already correct upstream are
# regression guards and live in their own `still_`-named tests below, so this list
# stays what its name says: the demonstration set.
@pytest.mark.parametrize(
    "price",
    # `$30.001` is why the cents group is `\d+` and not `\d{1,2}`: capped at two
    # digits it reads as `$30.00` and sits exactly on the ceiling.
    [
        "$1,200",
        "$1,200.00",
        "$1,000.50",
        "$30.99",
        "$30.01",
        "$30.001",
        "$30.99,4.5 stars",
    ],
)
def test_tc63_reads_the_whole_amount_not_its_first_digits(price):
    """An over-budget price must not clear the ceiling by being truncated.

    `_TC63_PRICE` stopped its number body at the first non-digit, so `$1,200`
    was read as `$1` and `$30.99` as `$30`. Both cleared a $30 ceiling, and
    since `answer_affirms_number` collapses digit grouping, the stray `1` in
    "table for 1" was what affirmed the truncated figure.
    """
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_within_budget

    answer = f"Trattoria Bella,Italian,downtown,{price},table for 1,open until 11pm."
    assert _tc63_within_budget(answer.lower()) is False, price


# REGRESSION GUARDS — every case in the two lists that follow is already green at
# the parent revision.
# They are here because the plausible fixes to the above break them.
@pytest.mark.parametrize(
    "price",
    [
        "$25",
        "$ 25",
        "US$25",
        "CA$25",
        "AU$25",
        "$30",
        "$30.00",
        "$29.99",
        "$28.50",
        "$22/person",
        "$25.500",
        "$30.000",
        "$22.50,4.5 stars",
        "$28.50,11pm",
        "$25.50,2 blocks away",
        "$22.50, 4.5 stars",
    ],
)
def test_tc63_still_accepts_an_amount_at_or_under_the_ceiling(price):
    """The plausible false positives of the change above.

    `US$25` works today, and a lookbehind rejecting a preceding word character
    would silently break it along with `CA$` and `AU$`. `$30.00` is the ceiling
    exactly. The last four are the cases that killed two earlier drafts: reading
    the amount is not enough if what gets LOOKED UP in the answer changes with
    it, because `answer_affirms_number` strips digit-grouping commas from the
    answer and can pull a following digit against a wider needle.
    """
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_within_budget

    answer = f"Trattoria Bella,Italian,downtown,{price},open until 11pm."
    assert _tc63_within_budget(answer.lower()) is True, price


@pytest.mark.parametrize("price", ["$25,4.5 stars", "$28,11pm", "$007", "$٢٥"])
def test_tc63_still_does_not_read_these_and_this_change_does_not_either(price):
    """Known limitations, genuinely unchanged — recorded, not fixed.

    `answer_affirms_number` collapses digit-grouping commas across the whole
    answer, so in `$25,4.5` the amount is not affirmed by that occurrence, and
    unless the answer names it again elsewhere the price is not read at all.
    `$007` is looked up as `7`, which the lookbehind refuses inside `007`
    itself. `$٢٥` is looked up as ASCII `25`, a string that does not occur in
    the answer at all. For all four the value looked up is byte-identical
    before and after, because the old three-digit cap did not change any of
    these amounts. Named so the guard above is not read as covering a class it
    only half covers.
    """
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_within_budget

    answer = f"Trattoria Bella,Italian,downtown,{price},open until 11pm."
    assert _tc63_within_budget(answer.lower()) is False, price


# ⚠ THE CLASS WHERE THIS CHANGE MOVES SCORES IN BOTH DIRECTIONS, pinned in both.
# The old lookup was built from the TRUNCATED capture, so where leading zeros push
# the significant digits past the old three-digit cap -- `$0025`, read as `$2` -- the
# two revisions look up different numbers and the sentence decides which one affirms.
# Shorter amounts are unaffected: `$007`, `$025` and `$030` look up identically on
# both. Keeping the truncated capture for the lookup would avoid the movement
# entirely; the wider lookup is a choice, made so the affirmation half stops reading
# a prefix too.
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # gains the constraint: `25` is affirmed, the old `2` was not
        (
            "Trattoria Bella - Italian, downtown, $0025 per person - that is 25 "
            "dollars, open until 11pm.",
            True,
        ),
        (
            "Trattoria Bella - Italian, downtown, $0030 per person, 30 total for "
            "two, open until 11pm.",
            True,
        ),
        # loses it: the old `2` was affirmed by "table for 2", `25` is not
        (
            "Trattoria Bella - Italian, downtown, $0025 per person, table for 2, open until 11pm.",
            False,
        ),
        (
            "Trattoria Bella - Italian, downtown, $0025, 4.5 stars, table for 2, open until 11pm.",
            False,
        ),
        # over budget AND leading-zero: base reads `003`, affirmed by "table for 3"
        (
            "Trattoria Bella - Italian, downtown, $0035 per person, table for 3, open until 11pm.",
            False,
        ),
        # `$0007`: the old lookup was `0`, from the truncated `000`; the new one is `7`
        (
            "Trattoria Bella - Italian, downtown, $0007 per person - that is 7 "
            "dollars, open until 11pm.",
            True,
        ),
    ],
)
def test_tc63_reads_a_leading_zero_amount_by_its_value_in_both_directions(answer, expected):
    """Both directions of the leading-zero class, asserted rather than left to be found."""
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_within_budget

    assert _tc63_within_budget(answer.lower()) is expected, answer


def test_tc63_still_returns_a_verdict_on_a_long_run_of_zeros():
    """Regression guard against a defect THIS change could introduce.

    Green before and after: upstream returns a verdict here only because its
    three-digit cap truncated the string before the conversion saw it.

    An answer can carry an arbitrarily long run of leading zeros whose float
    value is still under the ceiling, so the conversion has to strip them. A
    long NON-zero run cannot reach it: `float()` makes it `inf` and the ceiling
    rejects it first.
    """
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_within_budget

    for zero, body in (("0", "25"), ("\u0660", "\u0662\u0665"), ("\u0966", "\u0968\u096b")):
        answer = f"trattoria bella - italian, downtown, ${zero * 5000}{body} per person, 11pm."
        assert _tc63_within_budget(answer) is False, zero
    # a long NON-zero run cannot reach the conversion: it is `inf`, rejected by the ceiling
    long_nonzero = "trattoria bella - italian, downtown, $" + "1" * 5000 + " per person, 11pm."
    assert _tc63_within_budget(long_nonzero) is False


@pytest.mark.parametrize("price", ["$2,500", "$5,000"])
def test_tc63_still_rejects_a_grouped_amount_that_never_affirmed(price):
    """Regression guard, green before and after — recorded so it is not miscounted.

    These already scored correctly upstream, by luck rather than design: `$2,500`
    truncated to `$2`, and no standalone `2` appeared to affirm it. Reading the
    whole amount reaches the same verdict for the right reason.
    """
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_within_budget

    answer = f"Trattoria Bella,Italian,downtown,{price},table for 1,open until 11pm."
    assert _tc63_within_budget(answer.lower()) is False, price


def test_tc63_still_reads_a_negated_price_as_not_affirmed():
    """Regression guard, green before and after: the negation window must survive."""
    from tool_eval_bench.evals.scenarios.planning.tc63 import _tc63_within_budget

    answer = "Trattoria Bella, downtown, not $25.50 per person, open until 11pm."
    assert _tc63_within_budget(answer.lower()) is False


def test_tc63_does_not_award_full_credit_to_a_forty_times_over_budget_answer():
    """The scoring consequence, at the scenario boundary rather than the helper.

    CONTRIBUTING: "Ensure a plausible hallucination or unsafe action cannot
    receive full credit." A `$1,200 per person` recommendation scored PASS
    against a `$30` ceiling, under a verdict reading "Maintained all
    accumulated constraints".
    """
    scenario = next(s for s in ALL_SCENARIOS_WITH_HARDMODE if s.id == "TC-63")
    state = make_state(
        tool_calls=[{"name": "web_search", "arguments": {"query": "italian restaurant downtown"}}],
        final_answer=(
            "Trattoria Bella - Italian, downtown, $1,200 per person, table for 1, open until 11pm."
        ),
    )
    assert scenario.evaluate(state).status is ScenarioStatus.PARTIAL
