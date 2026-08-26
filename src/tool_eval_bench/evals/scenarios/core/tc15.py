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
    answer_contains_number as _answer_contains_number,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    first_call as _first_call,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback as _generic_tool_fallback,
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
    with_noise as _noise,
)
from tool_eval_bench.evals.scenarios.core._shared import (
    _numeric_value,
    _positive_argument_contains,
    _result_matches_if_present,
    _tc14_result_is_error,
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


def _tc15_eval(state: ScenarioState) -> ScenarioEvaluation:
    search = _first_call(state, "web_search")
    calc = _first_call(state, "calculator")

    # Check if the search tool actually returned data or was error-injected.
    search_results = [r for r in state.tool_results if r.call_id == search.id] if search else []
    search_got_error = bool(
        search and search_results and any(_tc14_result_is_error(r.result) for r in search_results)
    )

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
        if search:
            return _partial("Attempted search but tool failed — answer was incomplete.")
        return _fail("Did not attempt to search for the population.")

    if (
        search
        and calc
        and search.turn < calc.turn
        and _positive_argument_contains(search.arguments.get("query"), "population")
        and _positive_argument_contains(search.arguments.get("query"), "iceland")
        and "372520" in _as_str(calc.arguments.get("expression")).replace(",", "")
        and _result_matches_if_present(state, search, _tc15_search_result_has_population)
        and _result_matches_if_present(state, calc, _tc15_calculator_result_is_percentage)
    ):
        # Verify the model surfaced the actual computed result.
        has_result = _answer_contains_number(state.final_answer, "7450")
        if has_result:
            return _pass("Used the searched population value in the calculator.")
        return _partial(
            "Correctly piped the population into the calculator but did not surface "
            "the computed result in the answer.",
        )
    if (
        search
        and calc
        and search.turn == calc.turn
        and _positive_argument_contains(search.arguments.get("query"), "population")
        and _positive_argument_contains(search.arguments.get("query"), "iceland")
    ):
        return _partial(
            "Issued dependent calls in the same turn, so the calculation could not use "
            "the search result."
        )
    if not calc and search and _answer_contains_number(state.final_answer, "7450.4"):
        return _partial("Computed the right answer mentally after searching.")
    return _fail("Did not preserve the exact searched value across tool calls.")


SCENARIO = ScenarioDefinition(
    id="TC-15",
    title="Conflicting Information",
    category=Category.E,
    user_message="Search for the population of Iceland and calculate what 2% of it would be.",
    description="Carry the exact searched value into the calculator.",
    handle_tool_call=_tc15_handle,
    evaluate=_tc15_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it searches first, then calculates 2% using the exact searched population value.",
    "Fail if it skips the search or uses a memorized rounded number.",
)
