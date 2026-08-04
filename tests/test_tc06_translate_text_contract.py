"""Focused contract tests for the TC-06 translate_text language designators.

Pins the behaviour of the TC-06 scenario's mock and evaluator for the
finite language-designator vocabulary introduced by PR #43:

1. the mock accepts any accepted Spanish/Japanese designator as the target;
   English designators are valid *sources* only and error as targets;
2. unsupported designators are rejected by the mock;
3. the evaluator recognises Spanish/Japanese by any accepted designator and
   returns PASS only when both translations are present in the final answer —
   a missing translation fails, and extra/off-target calls or an incomplete
   answer downgrade PASS to PARTIAL;
4. the translate_text tool schema advertises exactly the finite supported
   designator set — the TC-06 alias table plus the German designators carried
   by the shared tool — with no omissions and no extras. This is NOT a full
   ISO 639 / BCP-47 resolver: only the explicitly listed designators are
   valid.

The tests exercise the scenario's public ``handle_tool_call``/``evaluate``
methods with real ``ScenarioState``/``ToolCallRecord`` objects built with the
repo's ``make_state``/``make_tool_call`` helpers (tests/conftest.py), so they
run through the same code path as the main contract suite.
"""

import pytest
from conftest import make_state, make_tool_call

from tool_eval_bench.domain.scenarios import ScenarioStatus
from tool_eval_bench.domain.tools import (
    TRANSLATE_LANGUAGE_DESIGNATORS,
    UNIVERSAL_TOOLS,
)
from tool_eval_bench.evals.scenarios import _LANGUAGE_ALIASES, SCENARIOS

# German designators accepted by the shared translate_text tool schema (used
# by other scenarios' handlers) but NOT part of the TC-06 alias table: the
# TC-06 mock rejects them. The schema advertises the union of both sets.
_EXTENDED_ONLY = {"german", "de", "deutsch"}

TC06 = next(s for s in SCENARIOS if s.id == "TC-06")

_TEXT = "Where is the nearest hospital?"
_ANSWER_SPANISH = "¿Dónde está el hospital más cercano?"
_ANSWER_JAPANESE = "最寄りの病院はどこですか？"


def _call(**arguments):
    return make_tool_call(name="translate_text", arguments=arguments)


def _state(*calls, final_answer=""):
    return make_state(tool_calls=list(calls), final_answer=final_answer)


def _mock(**arguments):
    return TC06.handle_tool_call(_state(), _call(**arguments))


# ---------------------------------------------------------------------------
# Mock: accepted target designators
# ---------------------------------------------------------------------------


def test_mock_accepts_spanish_designators():
    for designator in ("Spanish", "es", "ES", "es-ES", "es-419", "español", "castilian"):
        result = _mock(source_language="English", target_language=designator, text=_TEXT)
        assert "translated" in result, designator


def test_mock_accepts_japanese_designators():
    for designator in ("Japanese", "ja", "ja-JP", "jpn", "日本語"):
        result = _mock(source_language="English", target_language=designator, text=_TEXT)
        assert "translated" in result, designator


# ---------------------------------------------------------------------------
# Mock: rejected targets
# ---------------------------------------------------------------------------


def test_mock_rejects_unsupported_targets():
    for designator in ("German", "de", "French", "fr"):
        result = _mock(source_language="English", target_language=designator, text=_TEXT)
        assert "error" in result, designator


def test_mock_rejects_english_as_target():
    """English designators are valid *sources* only — never targets."""
    for designator in ("English", "en", "en-US"):
        result = _mock(source_language="English", target_language=designator, text=_TEXT)
        assert "error" in result, designator


@pytest.mark.parametrize("designator", sorted(_LANGUAGE_ALIASES))
def test_mock_handles_every_claimed_alias(designator):
    # Every alias the TC-06 handler claims to understand is exercised:
    # Spanish/Japanese designators translate, English ones error as targets.
    result = _mock(source_language="English", target_language=designator, text=_TEXT)
    if _LANGUAGE_ALIASES[designator] == "english":
        assert "error" in result, designator
    else:
        assert "translated" in result, designator


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_evaluator_passes_with_alias_targets():
    state = _state(
        _call(source_language="English", target_language="es", text=_TEXT),
        _call(source_language="en", target_language="Japanese", text=_TEXT),
        final_answer=f"{_ANSWER_SPANISH} {_ANSWER_JAPANESE}",
    )
    result = TC06.evaluate(state)
    assert result.status == ScenarioStatus.PASS
    assert result.points == 2


def test_evaluator_fails_when_spanish_translation_missing():
    state = _state(
        _call(source_language="English", target_language="Japanese", text=_TEXT),
        final_answer=_ANSWER_JAPANESE,
    )
    result = TC06.evaluate(state)
    assert result.status == ScenarioStatus.FAIL


def test_evaluator_fails_when_japanese_translation_missing():
    state = _state(
        _call(source_language="English", target_language="es", text=_TEXT),
        final_answer=_ANSWER_SPANISH,
    )
    result = TC06.evaluate(state)
    assert result.status == ScenarioStatus.FAIL


def test_evaluator_fails_when_source_not_english():
    state = _state(
        _call(source_language="German", target_language="Spanish", text=_TEXT),
        _call(source_language="en", target_language="Japanese", text=_TEXT),
        final_answer=f"{_ANSWER_SPANISH} {_ANSWER_JAPANESE}",
    )
    result = TC06.evaluate(state)
    assert result.status == ScenarioStatus.FAIL


def test_evaluator_fails_when_targets_bundled_in_one_call():
    state = _state(
        _call(
            source_language="English",
            target_language="Spanish and Japanese",
            text=_TEXT,
        ),
        final_answer=f"{_ANSWER_SPANISH} {_ANSWER_JAPANESE}",
    )
    result = TC06.evaluate(state)
    assert result.status == ScenarioStatus.FAIL


def test_evaluator_partial_with_off_target_call():
    state = _state(
        _call(source_language="English", target_language="Spanish", text=_TEXT),
        _call(source_language="en", target_language="ja", text=_TEXT),
        _call(source_language="English", target_language="German", text=_TEXT),
        final_answer=f"{_ANSWER_SPANISH} {_ANSWER_JAPANESE}",
    )
    result = TC06.evaluate(state)
    assert result.status == ScenarioStatus.PARTIAL
    assert result.points == 1


def test_evaluator_partial_when_answer_markers_missing():
    state = _state(
        _call(source_language="English", target_language="Spanish", text=_TEXT),
        _call(source_language="en", target_language="Japanese", text=_TEXT),
        final_answer="Here are the translations.",
    )
    result = TC06.evaluate(state)
    assert result.status == ScenarioStatus.PARTIAL
    assert result.points == 1


# ---------------------------------------------------------------------------
# Schema: translate_text advertises exactly the supported designator set
# ---------------------------------------------------------------------------


def test_translate_text_schema_enum_matches_supported_designators():
    tool = next(t for t in UNIVERSAL_TOOLS if t["function"]["name"] == "translate_text")
    props = tool["function"]["parameters"]["properties"]
    src_enum = props["source_language"]["enum"]
    tgt_enum = props["target_language"]["enum"]
    assert src_enum == TRANSLATE_LANGUAGE_DESIGNATORS
    assert tgt_enum == TRANSLATE_LANGUAGE_DESIGNATORS

    # The schema is exactly the TC-06 alias table plus the German
    # designators carried by the shared tool: no omissions, no extras.
    expected = set(_LANGUAGE_ALIASES) | _EXTENDED_ONLY
    assert set(TRANSLATE_LANGUAGE_DESIGNATORS) == expected
    assert len(TRANSLATE_LANGUAGE_DESIGNATORS) == len(expected)


def test_tc06_registered():
    assert TC06.id == "TC-06"
    assert TC06.handle_tool_call is not None
    assert TC06.evaluate is not None
