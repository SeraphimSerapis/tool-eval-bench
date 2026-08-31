"""TC-28 — Read-Before-Write."""

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
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    matching_tool_results as _matching_tool_results,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)
from tool_eval_bench.evals.helpers import (
    result_is_usable_if_present as _result_is_usable_if_present,
)
from tool_eval_bench.evals.helpers import (
    with_noise as _noise,
)


def _tc28_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "read_file":
        return _noise(
            {
                "content": (
                    "# Database Configuration\n"
                    "db:\n"
                    "  host: localhsot\n"
                    "  port: 5432\n"
                    "  name: myapp_prod\n"
                    "  pool_size: 10\n"
                ),
            },
            "read_file",
        )
    if call.name == "search_files":
        return _noise(
            {"results": [{"file_id": "config_yaml", "name": "config.yaml"}]}, "search_files"
        )
    return _generic_tool_fallback(call)


_TC28_CORRECTION = re.compile(
    r"\b(?:typo|misspell\w*|mis-?typed|spelling|correct\w*|fix\w*|replace\w*|"
    r"change\w*|should\s+(?:be|read)|instead\s+of|rather\s+than)\b|->|\u2192"
)


# Tools whose whole purpose is to find the file the read then consumes.
_TC28_DISCOVERY_TOOLS = frozenset({"search_files"})


# One lookup for *this* file earns its keep. A second identical search, or a
# search for something else entirely, is the waste this check exists to catch.
_TC28_DISCOVERY_QUERY = re.compile(r"config", re.IGNORECASE)


def _tc28_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User: 'The file config.yaml has a typo in the database host. Find and describe the fix.'

    Model MUST read_file first to see the actual contents before suggesting a fix.
    The typo is 'localhsot' → 'localhost'.
    """
    read_calls = [c for c in state.tool_calls if c.name == "read_file"]
    answer = state.final_answer.lower()

    # Quoting the misspelling is unambiguous. Otherwise the answer has to name
    # the corrected value *and* frame it as a correction, in any of the ways a
    # person would write one.
    found_typo = "localhsot" in answer or (
        "localhost" in answer and bool(_TC28_CORRECTION.search(answer))
    )

    if len(read_calls) == 0:
        if found_typo:
            return _partial("Identified the typo but guessed without reading the file first.")
        return _fail("Did not read the file before suggesting a fix.")

    correct_reads = [
        call
        for call in read_calls
        if _as_str(call.arguments.get("file_id")).lower() in ("config_yaml", "config.yaml")
    ]
    if found_typo and correct_reads:
        correct_read = correct_reads[-1]
        if not _result_is_usable_if_present(state, correct_read):
            return _partial("The file read failed, so the proposed correction was not grounded.")
        recorded_results = _matching_tool_results(state, correct_read)
        if recorded_results and not any(
            isinstance(result.result, dict)
            and "localhsot" in _as_str(result.result.get("content")).lower()
            for result in recorded_results
        ):
            return _partial("The proposed correction was not present in the recorded file content.")
        # `read_file` takes a `file_id`, and the only way to learn it is to
        # search for the file first. That one lookup is part of
        # read-before-write, not a detour. Everything else is waste: a repeat
        # search, a search for an unrelated file, a search issued after the
        # read, or any other tool.
        first_read = next(
            index for index, call in enumerate(state.tool_calls) if call.name == "read_file"
        )
        stray_calls = []
        allowed_discovery = 1
        for index, call in enumerate(state.tool_calls):
            if call.name == "read_file":
                continue
            if (
                allowed_discovery
                and index < first_read
                and call.name in _TC28_DISCOVERY_TOOLS
                and _TC28_DISCOVERY_QUERY.search(_as_str(call.arguments.get("query")))
            ):
                allowed_discovery -= 1
                continue
            stray_calls.append(call)
        if stray_calls:
            return _partial("Found the typo, but made an unnecessary additional tool call.")
        return _pass("Read the file first, correctly identified 'localhsot' → 'localhost'.")

    if found_typo:
        return _partial("Identified the typo, but read the wrong file.")

    return _partial("Read the file but didn't identify the typo correctly.")


SCENARIO = ScenarioDefinition(
    id="TC-28",
    title="Read-Before-Write",
    category=Category.J,
    user_message="The file config.yaml has a typo in the database host. Find and describe the fix.",
    description="Must read_file first to see actual contents before suggesting fix.",
    handle_tool_call=_tc28_handle,
    evaluate=_tc28_eval,
    difficulty=3,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it reads config.yaml first, then identifies 'localhsot' → 'localhost'.",
    "Fail if it guesses the fix without reading the file.",
)
