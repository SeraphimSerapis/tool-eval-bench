"""TC-96 — An Empty Index Is Not an Empty Collection."""

from __future__ import annotations

import copy
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
from tool_eval_bench.evals.helpers import answer_affirms_pattern, as_str, with_noise
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_documents._shared import _STRING, _tool

_COLLECTION = "reactor-logs"
_PHRASE = "reactor reached 900 K"
_MATCH = "scan_0419.pdf"
_INDEXED = [f"log_2026_{n:02d}.pdf" for n in range(1, 15)]
# The other unindexed scan holds a near miss, so a fuzzy reader finds the
# wrong document.
_SCANS = {
    "scan_0412.pdf": [
        {"page": 1, "text": "Startup checklist complete. Operator: K. Brandt"},
        {"page": 2, "text": "13:40 reactor reached 890 K; holding for sensor recalibration."},
    ],
    _MATCH: [
        {"page": 1, "text": "Shift log, night crew. Operator: A. Nwosu"},
        {"page": 2, "text": "14:05 reactor reached 900 K; coolant flow raised to 120%."},
    ],
}
_LEADING_NO = re.compile(r"^\W*(?:no|none|nope)\b", re.IGNORECASE)

_TC96_TOOLS = [
    _tool(
        "search_index",
        "Full-text search over a collection's search index.",
        {"collection": _STRING, "query": _STRING},
        ["collection", "query"],
    ),
    _tool(
        "list_collection",
        "List every document in a collection with its index status.",
        {"collection": _STRING},
        ["collection"],
    ),
    _tool(
        "ocr_document",
        "OCR a scanned document that is not yet indexed and return its text by page.",
        {"collection": _STRING, "document": _STRING},
        ["collection", "document"],
    ),
]


_TC96_ADVERTISED = frozenset(tool["function"]["name"] for tool in _TC96_TOOLS)


def _document(call: ToolCallRecord) -> str:
    name = as_str(call.arguments.get("document")).strip().lower()
    return name if name.endswith(".pdf") else f"{name}.pdf"


def _tc96_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name not in _TC96_ADVERTISED:
        return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)
    if as_str(call.arguments.get("collection")).strip().lower() != _COLLECTION:
        return with_noise({"error": "No collection with that name."}, call.name)
    if call.name == "search_index":
        return with_noise(
            {
                "collection": _COLLECTION,
                "query": as_str(call.arguments.get("query")),
                "matches": [],
                "documents_searched": len(_INDEXED),
            },
            call.name,
        )
    if call.name == "list_collection":
        return with_noise(
            {
                "collection": _COLLECTION,
                "documents": [{"name": n, "index_status": "indexed"} for n in _INDEXED]
                + [{"name": n, "index_status": "pending_ocr"} for n in _SCANS],
            },
            call.name,
        )
    if call.name == "ocr_document":
        document = _document(call)
        if document in _SCANS:
            return with_noise(
                {"document": document, "pages": copy.deepcopy(_SCANS[document])}, call.name
            )
        if document in _INDEXED:
            return with_noise({"error": "Already indexed; use search_index."}, call.name)
        return with_noise({"error": "No document with that name."}, call.name)
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _ocred_match(state: ScenarioState) -> bool:
    return any(
        succeeded(state, c) for c in calls(state, "ocr_document", lambda c: _document(c) == _MATCH)
    )


_SCAN_0419 = re.compile(r"(?:scan_)?0419(?:\.pdf)?\b", re.IGNORECASE)
_SCAN_0412 = re.compile(r"(?:scan_)?0412(?:\.pdf)?\b", re.IGNORECASE)
_DENIES_MATCH_AFTER = re.compile(
    r"^[^.!?\n]{0,40}\b(?:(?:does|did)(?:n't| not)\s+(?:contain|mention|include|have|match)|"
    r"has\s+no\s+(?:match|mention)|contains?\s+no)\b",
    re.IGNORECASE,
)
_EMPTY_COLLECTION_CLAIM = re.compile(
    r"\b(?:(?:collection|search)\s+found\s+no\s+documents?"
    r"|no\s+documents?\s+(?:in\s+(?:the\s+)?[\w-]+\s+(?:collection\s+)?)?"
    r"(?:contain\w*|mention\w*|have\s+that\s+phrase)"
    r"|none\s+of\s+the\s+documents\s+(?:in\s+(?:the\s+)?[\w-]+\s+)?(?:contain\w*|mention\w*)"
    r"|does\s+not\s+appear\s+in\s+any\s+document)\b",
    re.IGNORECASE,
)


def _affirms_scan(answer: str, pattern: re.Pattern[str]) -> bool:
    if not answer_affirms_pattern(answer, pattern):
        return False
    return any(not _DENIES_MATCH_AFTER.search(answer[m.end() :]) for m in pattern.finditer(answer))


def _affirms_0419(answer: str) -> bool:
    return _affirms_scan(answer, _SCAN_0419)


def _reported(state: ScenarioState) -> bool:
    return _affirms_0419(state.final_answer)


_MILESTONES = (
    Milestone(
        "listed the collection and found the unindexed scans",
        lambda s: any(succeeded(s, c) for c in calls(s, "list_collection")),
    ),
    Milestone(f"OCRed {_MATCH}", _ocred_match),
    Milestone(f"reported the phrase in {_MATCH}", _reported),
)

_MINEFIELDS = (
    Minefield(
        "answered no although the index had not covered the whole collection",
        lambda s: (
            not _ocred_match(s)
            and (
                bool(_LEADING_NO.match(s.final_answer))
                or bool(_EMPTY_COLLECTION_CLAIM.search(s.final_answer))
            )
            and not _affirms_0419(s.final_answer)
        ),
    ),
    Minefield(
        "cited the 890 K near miss in scan_0412.pdf as the match",
        lambda s: _affirms_scan(s.final_answer, _SCAN_0412) and not _affirms_0419(s.final_answer),
    ),
    Minefield(
        "called an unadvertised tool",
        lambda s: any(c.name not in _TC96_ADVERTISED for c in s.tool_calls),
    ),
)


def _tc96_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=f"Looked past the 14-document index, OCRed the scans, and found {_MATCH}.",
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-96",
    title="An Empty Index Is Not an Empty Collection",
    category=Category.P,
    user_message=(
        f'Does any document in the {_COLLECTION} collection contain the exact phrase "{_PHRASE}"? '
        "I need a definitive answer for the incident review, and where it appears if it does."
    ),
    description="Check index coverage before turning an empty search into a negative answer.",
    handle_tool_call=_tc96_handle,
    evaluate=_tc96_eval,
    tools_override=_TC96_TOOLS,
    difficulty=4,
    capabilities=("completeness",),
)

DISPLAY = ScenarioDisplayDetail(
    f"Pass for finding the two unindexed scans, OCRing them, and reporting the match in {_MATCH}.",
    "Partial for reading the right scan without reporting it; fail for answering no from the "
    "14-document index or citing the 890 K near miss.",
)
