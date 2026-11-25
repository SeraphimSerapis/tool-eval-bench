"""TC-19 — Message Routing."""

from __future__ import annotations

import json
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


_TC19_EXPECTED: tuple[tuple[str, ...], ...] = (
    ("code_help", "code", "engineering"),
    ("scheduling", "schedule", "calendar"),
    ("billing", "payment"),
    ("devops", "deploy"),
    ("research",),
)

_TC19_MESSAGE_CUES: tuple[tuple[str, ...], ...] = (
    ("refactor", "async/await"),
    ("thursday 3pm", "move my thursday"),
    ("charged twice", "same subscription"),
    ("docker container", "oom errors", "docker"),
    ("transformer architectures", "top papers", "transformer"),
)

_TC19_MARKER = re.compile(
    r"(?:^|\n)[^\S\n]*(?:[-•*][^\S\n]+|\|[^\S\n]*)?\*{0,2}"
    r"(?:"
    r"(?:message|msg)[^\S\n]*#?[^\S\n]*\*{0,2}([1-5])\*{0,2}[^\S\n]*(?:[.):\-|—–→]|->|=>)"
    r"|\[([1-5])\][^\S\n]*(?:[.):\-|—–→]|->|=>)?"
    r"|#([1-5])[^\S\n]*(?:[.):\-|—–→]|->|=>|\s)"
    r"|\*{0,2}([1-5])\*{0,2}[^\S\n]*(?:[.):\-|—–→]|->|=>)"
    r")",
    re.IGNORECASE,
)

_TC19_FENCED_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TC19_BULLET_LINE = re.compile(r"^\s*(?:[-•]\s*|\*\s+)(.+)$")
_TC19_CATEGORY_LIST_HEADER = re.compile(
    r"\b(?:available|possible|allowed|valid|given|provided|five|5)\s+categor(?:y|ies)\b"
    r"|\bcategories\s+(?:are|include|options)\b",
    re.IGNORECASE,
)
_TC19_REFUSAL = re.compile(
    r"\b(?:cannot|can't|can\s+not|unable\s+to|won't|will\s+not|refuse\s+to|"
    r"could\s+not|couldn't|not\s+able\s+to|"
    r"(?:don't|do\s+not)\s+have\s+(?:a\s+tool|the\s+ability)\s+to)\s+"
    r"(?:classify|categorize|route|label|sort|assign)\b",
    re.IGNORECASE,
)


def _tc19_segments(answer: str) -> dict[int, str]:
    """Map each message number to the text the answer attaches to it.

    When an answer quotes messages 1..5 before giving its 1..5 classifications,
    split markers into non-repeating blocks and grade the last complete block
    (or merged last-wins mapping when fewer than 4 items appear in any block).
    """
    markers: list[tuple[int, int, int]] = []
    for match in _TC19_MARKER.finditer(answer):
        digit = next(group for group in match.groups() if group is not None)
        markers.append((int(digit), match.start(), match.end()))
    if not markers:
        return {}
    blocks: list[dict[int, str]] = [{}]
    merged: dict[int, str] = {}
    for position, (index, _, end) in enumerate(markers):
        stop = markers[position + 1][1] if position + 1 < len(markers) else len(answer)
        text = answer[end:stop]
        if index in blocks[-1]:
            blocks.append({})
        blocks[-1][index] = text
        merged[index] = text
    complete_blocks = [block for block in blocks if len(block) >= 4]
    if complete_blocks:
        return complete_blocks[-1]
    return merged


def _tc19_message_index_from_text(text: str) -> int | None:
    low = text.lower()
    matches = [
        idx
        for idx, cues in enumerate(_TC19_MESSAGE_CUES, start=1)
        if any(cue in low for cue in cues)
    ]
    return matches[0] if len(matches) == 1 else None


def _tc19_json_value_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for field in ("category", "label", "classification", "class", "route"):
            if field in value and value[field] is not None:
                return _tc19_json_value_text(value[field])
        return str(value)
    if isinstance(value, list):
        return " ".join(_tc19_json_value_text(item) for item in value if item is not None)
    return str(value)


def _tc19_json_segments(answer: str) -> dict[int, str]:
    """Map message numbers to their classification in a JSON-shaped answer.

    Fenced JSON blocks are parsed before scanning raw ``{``/``[`` spans via
    ``JSONDecoder.raw_decode``, so prose brackets such as ``[1-5]`` before a
    JSON block never break extraction.
    """
    candidates: list[dict[int, str]] = []
    for block in _TC19_FENCED_BLOCK.findall(answer):
        try:
            payload = json.loads(block.strip())
        except (ValueError, RecursionError):
            continue
        parsed = _tc19_flatten(payload)
        if parsed:
            candidates.append(parsed)
    if candidates:
        return candidates[-1]

    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(answer):
        next_obj = answer.find("{", pos)
        next_arr = answer.find("[", pos)
        starts = [idx for idx in (next_obj, next_arr) if idx != -1]
        if not starts:
            break
        start = min(starts)
        try:
            payload, end = decoder.raw_decode(answer, start)
        except (ValueError, RecursionError):
            pos = start + 1
            continue
        parsed = _tc19_flatten(payload)
        if parsed:
            candidates.append(parsed)
        pos = max(end, start + 1)
    return candidates[-1] if candidates else {}


def _tc19_flatten(payload: object) -> dict[int, str]:
    """Pull ``{number: classification}`` out of any plausible JSON layout."""
    if isinstance(payload, list):
        segments: dict[int, str] = {}
        for pos, item in enumerate(payload, start=1):
            if item is None:
                continue
            if isinstance(item, dict):
                idx: int | None = None
                for key_name in ("id", "index", "number", "message_id", "message"):
                    raw_id = item.get(key_name)
                    if isinstance(raw_id, int) and 1 <= raw_id <= 5:
                        idx = raw_id
                        break
                    if isinstance(raw_id, str):
                        digits = re.fullmatch(
                            r"(?:(?:message|msg)[\s_#-]*)?([1-5])", raw_id.strip(), re.IGNORECASE
                        )
                        if digits:
                            idx = int(digits.group(1))
                            break
                        inferred = _tc19_message_index_from_text(raw_id)
                        if inferred is not None:
                            idx = inferred
                            break
                target = idx if idx is not None else pos
                if 1 <= target <= 5:
                    segments[target] = _tc19_json_value_text(item)
            elif 1 <= pos <= 5:
                segments[pos] = _tc19_json_value_text(item)
        return segments
    if not isinstance(payload, dict):
        return {}
    segments = {}
    for key, value in payload.items():
        key_str = str(key).strip()
        digits = re.fullmatch(r"(?:(?:message|msg)[\s_#-]*)?([1-5])", key_str, re.IGNORECASE)
        if digits:
            segments[int(digits.group(1))] = _tc19_json_value_text(value)
        elif (msg_idx := _tc19_message_index_from_text(key_str)) is not None:
            segments[msg_idx] = _tc19_json_value_text(value)
        elif isinstance(value, (list, dict)):
            segments.update(_tc19_flatten(value))
    return segments


def _tc19_label_is_asserted(segment: str, label: str) -> bool:
    """Match a category label as an assertion, not ``not <label>``."""
    return _answer_affirms_text(segment, label)


def _tc19_segment_matches(segment: str, expected_idx: int) -> bool:
    """Return True when ``segment`` asserts only the expected category."""
    matched_categories = {
        cat_idx
        for cat_idx, labels in enumerate(_TC19_EXPECTED)
        if any(_tc19_label_is_asserted(segment, label) for label in labels)
    }
    return matched_categories == {expected_idx}


def _tc19_bullet_blocks(answer: str) -> list[list[str]]:
    """Return contiguous bullet blocks excluding category-menu echoes."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    skip_current = False
    prev_non_empty = ""
    for raw_line in answer.splitlines():
        bullet = _TC19_BULLET_LINE.match(raw_line)
        if bullet:
            if current is None:
                skip_current = bool(_TC19_CATEGORY_LIST_HEADER.search(prev_non_empty))
                current = []
            if not skip_current:
                current.append(bullet.group(1))
        else:
            if current:
                blocks.append(current)
            current = None
            skip_current = False
            if raw_line.strip():
                prev_non_empty = raw_line.strip()
    if current:
        blocks.append(current)
    return blocks


def _tc19_eval(state: ScenarioState) -> ScenarioEvaluation:
    """User asks: classify 5 messages into categories. Model should not use tools.

    Expected: no tool calls, just a direct structured classification.
    """
    if len(state.tool_calls) > 0:
        return _fail("Used tools when direct classification was appropriate.")

    answer = state.final_answer.lower()
    json_segments = _tc19_json_segments(answer)
    segments = _tc19_segments(answer)
    refused = bool(_TC19_REFUSAL.search(answer))

    # Require structured output (numbered list, bullets, table, or per-message labeling)
    has_structure = bool(
        re.search(r"(?:^|\n)\s*(?:1[.)\]]|\[1\]|#1\b|message\s*1)", answer, re.MULTILINE)
        or re.search(r"(?:^|\n)\s*[-•*|]", answer, re.MULTILINE)
        or len(segments) >= 4
        or len(json_segments) >= 4
    )

    # Each label must be attached exclusively to the corresponding message.
    numbered_correct = sum(
        _tc19_segment_matches(segments[index], index - 1)
        for index in range(1, len(_TC19_EXPECTED) + 1)
        if index in segments
    )
    json_correct = sum(
        _tc19_segment_matches(json_segments[index], index - 1)
        for index in range(1, len(_TC19_EXPECTED) + 1)
        if index in json_segments
    )
    bullet_correct = 0
    if not refused:
        blocks = _tc19_bullet_blocks(answer)
        if blocks:
            candidate_lines = blocks[-1][:5]
            bullet_correct = sum(
                _tc19_segment_matches(line, idx) for idx, line in enumerate(candidate_lines)
            )
    correct = max(numbered_correct, bullet_correct, json_correct)

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
