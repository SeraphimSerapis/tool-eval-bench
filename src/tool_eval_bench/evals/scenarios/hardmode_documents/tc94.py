"""TC-94 — Route by Content, Not Extension."""

from __future__ import annotations

from typing import Any

from tool_eval_bench.domain.scenarios import (
    Category,
    ScenarioDefinition,
    ScenarioDisplayDetail,
    ScenarioEvaluation,
    ScenarioState,
    ToolCallRecord,
)
from tool_eval_bench.evals.helpers import as_str, with_noise
from tool_eval_bench.evals.milestones import Milestone, Minefield, calls, grade, succeeded
from tool_eval_bench.evals.scenarios.hardmode_documents._shared import _STRING, _tool

_PATH = "/scans/lab_notes.pdf"
_SAMPLE_ID = "LN-7731"
_EXTRACTORS = ("parse_pdf", "ocr_image")

_TC94_TOOLS = [
    _tool(
        "inspect_file",
        "Report a file's size, page count, and format detected from its content.",
        {"path": _STRING},
        ["path"],
    ),
    _tool(
        "parse_pdf",
        "Extract the embedded text layer from a PDF document.",
        {"path": _STRING},
        ["path"],
    ),
    _tool(
        "ocr_image",
        "Recognize text in raster images: PNG, JPEG, or TIFF, including multi-page TIFF.",
        {"path": _STRING, "language": {"type": "string", "description": "ISO 639-1 code"}},
        ["path"],
    ),
]


def _is_target(call: ToolCallRecord) -> bool:
    # Accept the bare file name too: it identifies the same observed file.
    path = as_str(call.arguments.get("path")).strip()
    return path in {_PATH, "lab_notes.pdf", "scans/lab_notes.pdf"}


def _tc94_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if not _is_target(call):
        return with_noise({"error": "File not found."}, call.name)
    if call.name == "inspect_file":
        return with_noise(
            {
                "path": _PATH,
                "size_bytes": 4_812_330,
                "detected_format": "image/tiff",
                "pages": 3,
            },
            call.name,
        )
    if call.name == "parse_pdf":
        # Deliberately misleading, as real parsers are: the file is a TIFF, not
        # a damaged PDF. The scenario grades whether the model checks.
        return with_noise(
            {"error": "PDF parse failed: no cross-reference table found. The file may be damaged."},
            call.name,
        )
    if call.name == "ocr_image":
        return with_noise(
            {
                "path": _PATH,
                "pages": [
                    {"page": 1, "text": f"Lab notebook, bench 4\nSample ID: {_SAMPLE_ID}"},
                    {"page": 2, "text": "Titration run 3: endpoint at 14.2 mL"},
                    {"page": 3, "text": "Operator: M. Duarte"},
                ],
            },
            call.name,
        )
    return with_noise({"error": f"Unsupported tool {call.name}."}, call.name)


def _inspection_turn(state: ScenarioState) -> int | None:
    turns = [c.turn for c in calls(state, "inspect_file") if succeeded(state, c)]
    return min(turns) if turns else None


def _inspected_first(state: ScenarioState) -> bool:
    """The format was known before any extractor ran, including an injected attempt."""
    turn = _inspection_turn(state)
    return turn is not None and all(
        c.turn > turn for name in _EXTRACTORS for c in calls(state, name)
    )


def _parsed_after_inspection(state: ScenarioState) -> bool:
    turn = _inspection_turn(state)
    return turn is not None and any(c.turn > turn for c in calls(state, "parse_pdf"))


_MILESTONES = (
    Milestone("inspected the file's format before choosing an extractor", _inspected_first),
    Milestone(
        "OCRed the TIFF pages",
        lambda s: any(succeeded(s, c) for c in calls(s, "ocr_image", _is_target)),
    ),
    Milestone(f"reported sample ID {_SAMPLE_ID}", lambda s: _SAMPLE_ID in s.final_answer.upper()),
)

_MINEFIELDS = (
    Minefield(
        "sent the file to the PDF parser after inspection showed a TIFF", _parsed_after_inspection
    ),
)


def _tc94_eval(state: ScenarioState) -> ScenarioEvaluation:
    return grade(
        state,
        milestones=_MILESTONES,
        minefields=_MINEFIELDS,
        success_summary=f"Detected a TIFF behind the .pdf name, OCRed it, and reported {_SAMPLE_ID}.",
        partial_after=2,
    )


SCENARIO = ScenarioDefinition(
    id="TC-94",
    title="Route by Content, Not Extension",
    category=Category.P,
    user_message=f"Extract the text from {_PATH} and tell me which sample ID it records.",
    description="Detect a TIFF saved under a .pdf name and route it to OCR instead of a PDF parser.",
    handle_tool_call=_tc94_handle,
    evaluate=_tc94_eval,
    tools_override=_TC94_TOOLS,
    difficulty=4,
    capabilities=("tool-selection", "tool-contracts"),
)

DISPLAY = ScenarioDisplayDetail(
    f"Pass for inspecting the file, OCRing the TIFF it really is, and reporting {_SAMPLE_ID}.",
    "Partial for reaching the answer after a blind PDF parse or without inspecting; fail for "
    "parsing after the inspection said TIFF, or stopping at the parser's 'damaged' error.",
)
