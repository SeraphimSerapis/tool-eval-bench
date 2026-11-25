"""TC-15 — Conflicting Information."""

from __future__ import annotations

import re
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    first_call_without_error as _first_call_without_error,
)
from tool_eval_bench.evals.helpers import forbid_unrequested_side_effects
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    has_explicit_tool_error as _has_explicit_tool_error,
)
from tool_eval_bench.evals.helpers import (
    parse_math_expression as _parse_math_expression,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    tool_calls_by_name as _tool_calls_by_name,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _answer_affirms_exact_value,
    _answer_affirms_rounded_value,
    _numeric_value,
    _positive_argument_contains,
    _result_matches_if_present,
)

_TC15_POPULATION_TERMS = re.compile(
    r"\b(?:population|people|inhabitants|residents|citizens|demographics|how\s+many)\b",
    re.IGNORECASE,
)
_TC15_EXPR_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def _tc15_normalized_expression(call: ToolCallRecord) -> str:
    return re.sub(r"(?<=\d)_(?=\d)", "", _as_str(call.arguments.get("expression")).replace(",", ""))


def _tc15_used_rounded_population(call: ToolCallRecord) -> bool:
    """Return True when a calculator call uses a large number other than 372520."""
    expr = _tc15_normalized_expression(call)
    for token in _TC15_EXPR_NUMBER.findall(expr):
        val = float(token)
        if val >= 10_000 and abs(val - 372520.0) > 1e-9:
            return True
    return False


def _tc15_search_is_iceland(
    state: ScenarioState, search: ToolCallRecord, *, allow_handler_resolved: bool = False
) -> bool:
    query = search.arguments.get("query")
    if not _positive_argument_contains(query, "iceland"):
        return False
    if _positive_argument_contains(query, "population") or bool(
        _TC15_POPULATION_TERMS.search(_as_str(query))
    ):
        return True
    return (
        allow_handler_resolved
        and bool(state.tool_results)
        and _result_matches_if_present(state, search, _tc15_search_result_has_population)
    )


def _tc15_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        return _noise(
            {
                "results": [
                    {"snippet": "Iceland has a population of approximately 372,520 as of 2025."}
                ]
            },
            "web_search",
        )
    if call.name == "calculator":
        result = _parse_math_expression(_as_str(call.arguments.get("expression", "")))
        payload = {"error": "Invalid expression."} if result is None else {"result": result}
        return _noise(payload, "calculator")
    return _generic_tool_fallback(call)


def _tc15_search_result_has_population(payload: Any) -> bool:
    if (
        not isinstance(payload, dict)
        or "error" in payload
        or not isinstance(payload.get("results"), list)
    ):
        return False
    return any(
        isinstance(item, dict)
        and (
            bool(re.search(r"\b372[,.]?520\b", _as_str(item.get("snippet"))))
            or any(
                _numeric_value(item.get(field)) == 372520
                for field in ("population", "population_estimate", "value")
            )
        )
        for item in payload["results"]
    )


def _tc15_calculator_result_is_percentage(payload: Any) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    value = _numeric_value(payload.get("result"))
    return value is not None and abs(value - 7450.4) < 1e-9


@forbid_unrequested_side_effects({"run_code": None})
def _tc15_eval(state: ScenarioState) -> ScenarioEvaluation:
    # Grade the first search attempt that returned something. An errored call
    # gave the model no number to carry or misuse, so a retry after it is
    # recovery.
    search = _first_call_without_error(state, "web_search")
    all_calcs = _tool_calls_by_name(state, "calculator")
    usable_calcs = [c for c in all_calcs if not _has_explicit_tool_error(state, c)]

    # Only when no search succeeded did the model lack the searched value.
    search_got_error = bool(search and _has_explicit_tool_error(state, search))

    if search_got_error:
        # Search tool failed — model can't use a value it never received.
        # Credit it for attempting the search and providing a reasonable answer.
        answer = state.final_answer.replace(",", "")
        has_reasonable_answer = any(
            num in answer for num in ("7450", "7450.4", "7500", "7504", "7451")
        )
        if (
            search
            and has_reasonable_answer
            and _positive_argument_contains(search.arguments.get("query"), "iceland")
        ):
            return _partial("Search tool failed — used background knowledge as fallback.")
        if search and _positive_argument_contains(search.arguments.get("query"), "iceland"):
            return _partial("Attempted search but tool failed — answer was incomplete.")
        return _fail("Did not attempt to search for the population.")

    # Multi-step calculations (e.g. calculator("2 / 100") then
    # calculator("372520 * 0.02")) are valid as long as no calculator call ran
    # before the search and no non-error call plugged in a memorized/rounded
    # population value.
    no_pre_search_calc = bool(search) and all(
        search is not None and search.turn < c.turn for c in usable_calcs
    )
    no_rounded_pop_calc = not any(_tc15_used_rounded_population(c) for c in usable_calcs)
    matching_calc = (
        next(
            (
                c
                for c in usable_calcs
                if "372520" in _tc15_normalized_expression(c)
                and _result_matches_if_present(state, c, _tc15_calculator_result_is_percentage)
            ),
            None,
        )
        if (no_pre_search_calc and no_rounded_pop_calc)
        else None
    )
    if (
        search
        and matching_calc
        and _tc15_search_is_iceland(state, search, allow_handler_resolved=True)
        and _result_matches_if_present(state, search, _tc15_search_result_has_population)
    ):
        # Verify the model surfaced the actual computed result.
        has_result = _answer_affirms_rounded_value(state.final_answer, "7450.4")
        if has_result:
            return _pass("Used the searched population value in the calculator.")
        return _partial(
            "Correctly piped the population into the calculator but did not surface "
            "the computed result in the answer.",
        )
    if (
        search
        and usable_calcs
        and any(search.turn == c.turn for c in usable_calcs)
        and _tc15_search_is_iceland(state, search, allow_handler_resolved=True)
    ):
        return _partial(
            "Issued dependent calls in the same turn, so the calculation could not use "
            "the search result."
        )
    if (
        not usable_calcs
        and search
        and _tc15_search_is_iceland(state, search, allow_handler_resolved=True)
        and _result_matches_if_present(state, search, _tc15_search_result_has_population)
        and _answer_affirms_exact_value(state.final_answer, "7450.4")
    ):
        return _partial("Computed the right answer mentally after searching.")
    return _fail("Did not preserve the exact searched value across tool calls.")


SCENARIO = ScenarioDefinition(
    id="TC-15",
    title="Conflicting Information",
    category=Category.E,
    user_message=(
        "Search for the population of Iceland, then use the calculator to compute 2% "
        "of the exact population value you find."
    ),
    description="Carry the exact searched value into the calculator.",
    handle_tool_call=_tc15_handle,
    evaluate=_tc15_eval,
    difficulty=3,
    dependencies=(("web_search", "calculator"),),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it searches first, then calculates 2% using the exact searched population value.",
    "Fail if it skips the search or uses a memorized rounded number.",
)
