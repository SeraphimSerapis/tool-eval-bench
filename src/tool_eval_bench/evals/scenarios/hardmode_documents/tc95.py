"""TC-95 — Native Text First, OCR Only Where Needed."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import answer_affirms_pattern, as_str, counted_calls, with_noise
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_documents._shared import _DOCUMENT_ID, _tool

_DOC = "DOC-INSP-4"
_NATIVE = frozenset({1, 2})
_SCANNED = frozenset({3, 4})
_TEXT = {
    1: "Weld inspection report WR-2291\nInspector: R. Okafor\nSite: Pier 7",
    2: "Scope: welds 1 to 4 on the north girder.",
    3: "Weld 3: surface corrosion with 2.1 mm pitting.",
    4: "Final verdict: REJECTED. Repair weld 3 before re-inspection.",
}
_PAGES = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "1-based page numbers",
}
_REJECTED = re.compile(r"\breject\w*", re.IGNORECASE)

_TC95_TOOLS = [
    _tool(
        "get_document_info",
        "Read a document's page count and which pages carry an embedded text layer.",
        {"document_id": _DOCUMENT_ID},
        ["document_id"],
    ),
    _tool(
        "extract_text",
        "Read the embedded text layer of the given pages. Fast and free. A scanned page has no "
        "text layer and returns empty text.",
        {"document_id": _DOCUMENT_ID, "pages": _PAGES},
        ["document_id", "pages"],
    ),
    _tool(
        "ocr_pages",
        "Run OCR on the given pages. Slow and billed per page; only needed for pages without a "
        "text layer.",
        {"document_id": _DOCUMENT_ID, "pages": _PAGES},
        ["document_id", "pages"],
    ),
]


def _pages(call: ToolCallRecord) -> set[int] | None:
    value = call.arguments.get("pages")
    items = value if isinstance(value, list) else [value]
    pages: set[int] = set()
    for item in items:
        if isinstance(item, int) and not isinstance(item, bool):
            pages.add(item)
        elif as_str(item).strip().isdigit():
            pages.add(int(as_str(item).strip()))
        else:
            return None
    return pages if pages and pages <= set(_TEXT) else None


def _tc95_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if as_str(call.arguments.get("document_id")) != _DOC:
        return with_noise({"error": "No document with that id."}, call.name)
    if call.name == "get_document_info":
        return with_noise(
            {
                "document_id": _DOC,
                "page_count": len(_TEXT),
                "pages": [{"page": p, "has_text_layer": p in _NATIVE} for p in sorted(_TEXT)],
            },
            call.name,
        )
    if call.name not in {"extract_text", "ocr_pages"}:
        return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)
    pages = _pages(call)
    if pages is None:
        return with_noise({"error": "pages must list page numbers from 1 to 4."}, call.name)
    ocr = call.name == "ocr_pages"
    return with_noise(
        {
            "document_id": _DOC,
            "pages": [
                {"page": p, "text": _TEXT[p] if ocr or p in _NATIVE else ""} for p in sorted(pages)
            ],
            **({"billed_pages": len(pages)} if ocr else {}),
        },
        call.name,
    )


def _covered(state: ScenarioState, name: str) -> set[int]:
    return {page for c in calls(state, name) if succeeded(state, c) for page in _pages(c) or set()}


def _ocr_on_native(state: ScenarioState) -> bool:
    # Every attempt counts, injected or not: the model chose to pay for it.
    return any((_pages(c) or set()) & _NATIVE for c in calls(state, "ocr_pages"))


def _ocr_twice(state: ScenarioState) -> bool:
    # A same-turn copy of an injected attempt was sent before the error came
    # back, so it is a duplicate; counted_calls already drops a later retry.
    billed = Counter(
        page
        for c in counted_calls(calls(state, "ocr_pages"))
        if c.injected or succeeded(state, c)
        for page in _pages(c) or set()
    )
    return any(count > 1 for count in billed.values())


def _scan_status_turn(state: ScenarioState) -> int | None:
    """The first turn whose result showed which pages lack a text layer."""
    turns = [c.turn for c in calls(state, "get_document_info") if succeeded(state, c)] + [
        c.turn
        for c in calls(state, "extract_text")
        if succeeded(state, c) and (_pages(c) or set()) & _SCANNED
    ]
    return min(turns) if turns else None


def _ocr_scanned(state: ScenarioState) -> bool:
    observed = _scan_status_turn(state)
    return (
        observed is not None
        and _SCANNED <= _covered(state, "ocr_pages")
        and all(c.turn > observed for c in calls(state, "ocr_pages"))
    )


def _reported(state: ScenarioState) -> bool:
    answer = state.final_answer
    return "okafor" in answer.lower() and answer_affirms_pattern(answer, _REJECTED)


_MILESTONES = (
    Milestone(
        "read pages 1 and 2 from the text layer", lambda s: _NATIVE <= _covered(s, "extract_text")
    ),
    Milestone("OCRed pages 3 and 4 after seeing they had no text layer", _ocr_scanned),
    Milestone("reported inspector R. Okafor and the rejected verdict", _reported),
)

_MINEFIELDS = (
    Minefield("OCRed a page that already had a text layer", _ocr_on_native),
    Minefield("OCRed the same page more than once", _ocr_twice),
)


def _tc95_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary="Read pages 1-2 natively, OCRed only pages 3-4, and reported both facts.",
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-95",
    title="Native Text First, OCR Only Where Needed",
    category=Category.P,
    user_message=(
        f"Pull the text of all four pages of inspection report {_DOC}. OCR is billed per page, "
        "so avoid it where it isn't needed. Then tell me who the inspector was and what the "
        "final verdict is."
    ),
    description="Read text-layer pages natively and OCR only the scanned ones.",
    handle_tool_call=_tc95_handle,
    evaluate=_tc95_eval,
    tools_override=_TC95_TOOLS,
    difficulty=4,
    capabilities=("constraints", "tool-selection"),
)

DISPLAY = ScenarioDisplayDetail(
    "Pass for native extraction of pages 1-2, OCR of exactly pages 3-4, and reporting the "
    "inspector and the rejected verdict.",
    "Partial for the right extraction with an incomplete answer; fail for OCRing a text page, "
    "OCRing a page twice, or skipping the scanned pages.",
)
