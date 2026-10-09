"""TC-93 — Printed Page Versus Physical Index."""

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
from tool_eval_bench.evals.helpers import answer_contains_number, as_str, counted_calls, with_noise
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_documents._shared import _DOCUMENT_ID, _INTEGER, _tool

_DOC = "DOC-HB-2024"
_PAGE_COUNT = 212
# Five lowercase-roman front-matter pages precede printed page 1.
_FRONT_MATTER = 5
_ROMAN = ("i", "ii", "iii", "iv", "v")
# Printed page 12 is the seventeenth sheet: zero-based index 16.
_TARGET_INDEX = 16

# Every page a plausible indexing mistake lands on carries its own torque
# value, so a wrong page produces a confident wrong answer, not an empty one.
_PAGES = {
    11: "2.1 Panel mounting. Torque the M6 panel screws to 10 N·m.",
    12: "2.2 Bracket mounting. Torque the M10 bracket bolts to 45 N·m.",
    15: "4.2 Gasket seating. Torque the gasket clamp to 18 N·m before fitting the flange.",
    16: "4.3 Flange assembly. Torque the M8 flange bolts to 24 N·m in a star pattern.",
    17: "4.4 Flange inspection. Re-torque the flange bolts to 26 N·m after 50 operating hours.",
}
_WRONG_TORQUE = re.compile(r"(?<![\d.])(?:10|45|18|26)\s*N", re.IGNORECASE)

_TC93_TOOLS = [
    _tool(
        "get_document_info",
        "Read a document's title, page count, and page-label ranges.",
        {"document_id": _DOCUMENT_ID},
        ["document_id"],
    ),
    _tool(
        "extract_text",
        "Extract the text of one page. page_index is the zero-based physical position of the "
        "page in the file, not the number printed on the page.",
        {"document_id": _DOCUMENT_ID, "page_index": _INTEGER},
        ["document_id", "page_index"],
    ),
]


def _page_label(index: int) -> str:
    return _ROMAN[index] if index < _FRONT_MATTER else str(index - _FRONT_MATTER + 1)


def _page_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = as_str(value).strip()
    return int(text) if text.isdigit() else None


def _tc93_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if as_str(call.arguments.get("document_id")) != _DOC:
        return with_noise({"error": "No document with that id."}, call.name)
    if call.name == "get_document_info":
        return with_noise(
            {
                "document_id": _DOC,
                "title": "Field Service Handbook",
                "page_count": _PAGE_COUNT,
                "page_labels": [
                    {"start_index": 0, "end_index": 4, "style": "lowercase-roman", "first": "i"},
                    {"start_index": 5, "end_index": 211, "style": "decimal", "first": "1"},
                ],
            },
            call.name,
        )
    if call.name == "extract_text":
        index = _page_index(call.arguments.get("page_index"))
        if index is None or not 0 <= index < _PAGE_COUNT:
            return with_noise({"error": f"page_index must be 0 to {_PAGE_COUNT - 1}."}, call.name)
        return with_noise(
            {
                "document_id": _DOC,
                "page_index": index,
                "page_label": _page_label(index),
                "text": _PAGES.get(index, "Section text with no torque specification."),
            },
            call.name,
        )
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _target_extractions(state: ScenarioState) -> list[ToolCallRecord]:
    return [
        c
        for c in calls(state, "extract_text")
        if _page_index(c.arguments.get("page_index")) == _TARGET_INDEX and succeeded(state, c)
    ]


def _grounded(state: ScenarioState) -> bool:
    """The target page was chosen from something the model had already read.

    Either the label map or an earlier page's footer label reveals the offset.
    A same-turn read could not have informed the index, so it must come first.
    """
    targets = _target_extractions(state)
    if not targets:
        return False
    first = targets[0]
    return any(
        c.turn < first.turn and succeeded(state, c)
        for name in ("get_document_info", "extract_text")
        for c in calls(state, name)
    )


def _reported(state: ScenarioState) -> bool:
    return answer_contains_number(state.final_answer, "24")


def _wrong_value(state: ScenarioState) -> bool:
    return not _reported(state) and _WRONG_TORQUE.search(state.final_answer) is not None


_MILESTONES = (
    Milestone(
        "extracted index 16, which is printed page 12", lambda s: bool(_target_extractions(s))
    ),
    Milestone("chose that index from the page labels it had read", _grounded),
    Milestone("reported 24 N·m", _reported),
)

_MINEFIELDS = (
    Minefield("reported a torque value from a different page", _wrong_value),
    Minefield(
        "extracted more than four pages to find one",
        lambda s: len(counted_calls(calls(s, "extract_text"))) > 4,
    ),
)


def _tc93_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary="Mapped printed page 12 to physical index 16 and reported 24 N·m.",
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-93",
    title="Printed Page Versus Physical Index",
    category=Category.P,
    user_message=(
        f"In the field service handbook {_DOC}, what torque does printed page 12 specify for "
        "the flange bolts? I mean the page with 12 in its footer, not the twelfth sheet in the file."
    ),
    description="Map a printed page label to the tool's zero-based physical page index.",
    handle_tool_call=_tc93_handle,
    evaluate=_tc93_eval,
    tools_override=_TC93_TOOLS,
    difficulty=4,
    capabilities=("tool-contracts",),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass for reading the page labels, extracting zero-based index 16, and reporting 24 N·m.",
    "Partial for the right page with no answer or with an index guessed before reading any "
    "labels; fail for reporting another page's torque or extracting more than four pages.",
)
