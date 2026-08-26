"""TC-06 — Multi-Value Extraction."""

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
    generic_tool_fallback as _generic_tool_fallback,
)
from tool_eval_bench.evals.helpers import (
    includes_text as _includes_text,
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
    _language_alias,
    _positive_argument_contains,
    _result_matches_if_present,
)


def _tc06_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "translate_text":
        target = _language_alias(call.arguments.get("target_language"))
        if target == "spanish":
            return _noise({"translated": "¿Dónde está el hospital más cercano?"}, "translate_text")
        if target == "japanese":
            return _noise({"translated": "最寄りの病院はどこですか？"}, "translate_text")
        raw = _as_str(call.arguments.get("target_language"))
        return _noise({"error": f"Unsupported target language {raw}."}, "translate_text")
    return _generic_tool_fallback(call)


def _tc06_translation_result_is_valid(payload: Any, target: str) -> bool:
    if not isinstance(payload, dict) or "error" in payload:
        return False
    translated = _as_str(payload.get("translated"))
    if not translated:
        return False
    if target == "spanish":
        return _includes_text(translated, "hospital") and (
            _includes_text(translated, "cercano") or _includes_text(translated, "dónde")
        )
    return _includes_text(translated, "病院") and (
        _includes_text(translated, "どこ") or _includes_text(translated, "最寄り")
    )


def _tc06_eval(state: ScenarioState) -> ScenarioEvaluation:
    calls = _tool_calls_by_name(state, "translate_text")
    has_spanish = any(
        _language_alias(c.arguments.get("source_language")) == "english"
        and _language_alias(c.arguments.get("target_language")) == "spanish"
        and _positive_argument_contains(c.arguments.get("text"), "where is the nearest hospital")
        for c in calls
    )
    has_japanese = any(
        _language_alias(c.arguments.get("source_language")) == "english"
        and _language_alias(c.arguments.get("target_language")) == "japanese"
        and _positive_argument_contains(c.arguments.get("text"), "where is the nearest hospital")
        for c in calls
    )
    invalid_bundled = any(
        re.search(
            r"spanish.*japanese|japanese.*spanish",
            _as_str(c.arguments.get("target_language")),
            re.IGNORECASE,
        )
        for c in calls
    )
    # A call is extra/off-target when its target language is not one of the
    # two required languages (Spanish or Japanese by any accepted designator)
    # or when the text is missing/off-topic. Calls targeting Spanish or
    # Japanese by any accepted designator count as translation attempts.
    extraneous = any(
        _language_alias(c.arguments.get("target_language")) not in ("spanish", "japanese")
        or _language_alias(c.arguments.get("source_language")) != "english"
        or not _positive_argument_contains(c.arguments.get("text"), "where is the nearest hospital")
        for c in calls
    )
    spanish_calls = [
        c
        for c in calls
        if _language_alias(c.arguments.get("source_language")) == "english"
        and _language_alias(c.arguments.get("target_language")) == "spanish"
        and _positive_argument_contains(c.arguments.get("text"), "where is the nearest hospital")
    ]
    japanese_calls = [
        c
        for c in calls
        if _language_alias(c.arguments.get("source_language")) == "english"
        and _language_alias(c.arguments.get("target_language")) == "japanese"
        and _positive_argument_contains(c.arguments.get("text"), "where is the nearest hospital")
    ]
    usable_spanish = any(
        _result_matches_if_present(
            state, call, lambda payload: _tc06_translation_result_is_valid(payload, "spanish")
        )
        for call in spanish_calls
    )
    usable_japanese = any(
        _result_matches_if_present(
            state, call, lambda payload: _tc06_translation_result_is_valid(payload, "japanese")
        )
        for call in japanese_calls
    )
    if (
        has_spanish
        and has_japanese
        and usable_spanish
        and usable_japanese
        and not invalid_bundled
        and not extraneous
    ):
        # Verify the model surfaced the actual translations.
        answer = state.final_answer
        has_spanish_text = "Dónde" in answer or (
            "hospital" in answer.lower() and "cercano" in answer
        )
        has_japanese_text = "病院" in answer or "最寄り" in answer or "どこ" in answer
        if has_spanish_text and has_japanese_text:
            return _pass("Issued separate translate_text calls for both languages.")
        return _partial(
            "Called translate_text correctly for both languages but did not surface "
            "the translations in the answer.",
        )
    if has_spanish and has_japanese:
        return _partial("Made the required translations but added an invalid or extra call.")
    return _fail("Did not split the translation request into two valid tool calls.")


SCENARIO = ScenarioDefinition(
    id="TC-06",
    title="Multi-Value Extraction",
    category=Category.B,
    user_message=(
        "Use the translate_text tool to translate 'Where is the nearest hospital?' "
        "from English into both Spanish and Japanese."
    ),
    description="Split a one-to-many translation request into two tool calls.",
    handle_tool_call=_tc06_handle,
    evaluate=_tc06_eval,
    difficulty=2,
)

DISPLAY = ScenarioDisplayDetail(
    "Pass if it makes two translate_text calls, one for Spanish and one for Japanese.",
    "Fail if it answers without the tool, combines both languages into one call, "
    "or only translates one of them.",
)
