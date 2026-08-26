"""TC-19 — Message Routing."""

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
    answer_affirms_text as _answer_affirms_text,
)
from tool_eval_bench.evals.helpers import (
    fail_eval as _fail,
)
from tool_eval_bench.evals.helpers import (
    generic_tool_fallback_simple as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    partial_eval as _partial,
)
from tool_eval_bench.evals.helpers import (
    pass_eval as _pass,
)


def _tc19_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    """No tools needed — tests direct structured output."""
    return _generic_tool_fallback(call)


_TC19_MARKER = re.compile(
    r"(?:^|\n)[^\S\n]*\|?[^\S\n]*(?:message[^\S\n]*)?\*{0,2}([1-5])\*{0,2}[^\S\n]*[.):\-|]"
)


def _tc19_segments(answer: str) -> dict[int, str]:
    """Map each message number to the text the answer attaches to it."""
    markers = [(int(m.group(1)), m.start(), m.end()) for m in _TC19_MARKER.finditer(answer)]
    segments: dict[int, str] = {}
    for position, (index, _, end) in enumerate(markers):
        stop = markers[position + 1][1] if position + 1 < len(markers) else len(answer)
        segments.setdefault(index, answer[end:stop])
    return segments


def _tc19_label_is_asserted(segment: str, label: str) -> bool:
    """Match a category label as an assertion, not ``not <label>``."""
    return _answer_affirms_text(segment, label)


def _tc19_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks: classify 5 messages into categories. Model should not use tools.

    Expected: no tool calls, just a direct structured classification.
    """
    if len(state.tool_calls) > 0:
        return _fail("Used tools when direct classification was appropriate.")

    answer = state.final_answer.lower()

    # Require structured output (numbered list, bullets, table, or per-message labeling)
    has_structure = bool(
        re.search(r"(?:^|\n)\s*(?:1[.)\]]|message\s*1)", answer, re.MULTILINE)
        or re.search(r"(?:^|\n)\s*[-•*|]", answer, re.MULTILINE)
    )

    expected = [
        ("code_help", "code", "engineering"),
        ("scheduling", "schedule", "calendar"),
        ("billing", "payment"),
        ("devops", "deploy"),
        ("research",),
    ]
    # Each label must be attached to the corresponding message. A keyword bag
    # can pass without classifying anything at all — but the layout carrying
    # that association is the model's choice, so segment on the message marker
    # rather than assuming one list style. This covers "1.", "1)", "Message 1:",
    # "**1**" and markdown table rows alike.
    segments = _tc19_segments(answer)
    numbered_correct = sum(
        any(_tc19_label_is_asserted(segments[index], label) for label in labels)
        for index, labels in enumerate(expected, start=1)
        if index in segments
    )
    bullet_lines = re.findall(r"(?:^|\n)\s*[-•*]\s*(.+)", answer)
    bullet_correct = sum(
        any(_tc19_label_is_asserted(line, label) for label in labels)
        for line, labels in zip(bullet_lines[:5], expected, strict=False)
    )
    correct = max(numbered_correct, bullet_correct)

    if correct >= 4 and has_structure:
        return _pass("Classified messages correctly in structured format without tool use.")
    if correct >= 4:
        return _partial(
            "Classifications correct but output lacked structured format (no list/labels)."
        )
    if correct >= 3:
        return _partial(f"Got {correct}/5 classifications right.")
    return _fail(f"Only {correct}/5 classifications correct.")


SCENARIO = ScenarioDefinition(
    id="TC-19",
    title="Message Routing",
    category=Category.G,
    user_message=(
        "Classify each message into one category (code_help, scheduling, billing, devops, research):\n"
        "1. 'Can you refactor this to use async/await?'\n"
        "2. 'Move my Thursday 3pm to Friday'\n"
        "3. 'I was charged twice for the same subscription'\n"
        "4. 'The Docker container keeps crashing with OOM errors'\n"
        "5. 'Find me the top papers on transformer architectures from 2024'"
    ),
    description="Classify messages without using any tools.",
    handle_tool_call=_tc19_handle,
    evaluate=_tc19_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it classifies 4+ messages correctly without using any tools.",
    "Fail if it uses tools or misclassifies most messages.",
)
