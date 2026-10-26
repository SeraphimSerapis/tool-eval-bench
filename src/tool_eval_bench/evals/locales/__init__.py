"""Opt-in multilingual runs: translated prompts and tool text, invariant anchors.

A locale run replaces a scenario's user message, follow-up messages, and the
natural-language descriptions in its tool schemas with a checked-in
translation. Everything a mock handler or evaluator matches stays the same:
identifiers, tickers, file names, code, JSON schemas, and the proper nouns in
``ANCHORS``, which every translation must carry verbatim. Mock tool results
stay in English, so the model reads mixed-language context, as it would
against a real English backend.

Only scenarios whose grading is language-neutral are eligible. ``ANCHORS``
is that list. A scenario stays out when its evaluator reads English prose
(refusals, clarifications, explanations) or when its mock matches an ordinary
English noun in a search query or a written field, which a model would
reasonably translate.

The evaluator sees the answer after ``normalize_answer``: locale number
formats become the English ones the shared helpers parse, and locale negation
words become ``not`` so the negation guard still works.
"""

from __future__ import annotations

import copy
import functools
import json
import re
from dataclasses import replace
from importlib import resources
from typing import Any

from tool_eval_bench.domain.scenarios import (
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioState,
)
from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS

SUPPORTED_LOCALES: tuple[str, ...] = ("de", "es", "zh", "ja")
# Bump when a pack's meaning changes, so runs on different translations land
# in different fingerprint cohorts.
LOCALE_VERSION = 1

# Eligible scenarios and the literals each translation must keep verbatim.
ANCHORS: dict[str, tuple[str, ...]] = {
    "TC-02": ("AAPL",),
    "TC-04": ("Tokyo",),
    "TC-05": ("Team Standup", "Alex", "Jamie"),
    "TC-09": ("London", "MSFT"),
    "TC-11": ("15", "200"),
    "TC-21": ("R1", "R2", "R3", "R4", "R5", "YYYY-MM-DD", "user@example.com"),
    "TC-24": ("Q3",),
    "TC-27": ("London",),
    "TC-28": ("config.yaml",),
    "TC-30": ("print(2+2)", 'print("correct")', 'print("wrong")'),
    "TC-39": ("15", "200"),
    "TC-44": (),
    "TC-45": ("7", "8"),
    "TC-47": ("Sprint Planning",),
    "TC-64": ("The Matrix",),
    "TC-67": ("NVDA",),
    "TC-68": ("PROJ-127",),
    "TC-72": ("Project Alpha", "Dana"),
    "TC-78": ("ACME", "BETA", "CYGN"),
    "TC-88": ("73", "91", "109"),
    "TC-93": ("DOC-HB-2024", "12"),
    "TC-94": ("/scans/lab_notes.pdf",),
    "TC-97": ("DOC-CONTRACT-88", "CUST-7714"),
    "TC-100": ("INC-710", "acc_101", "acc_102", "acc_103", "issue_batch_credit"),
    "TC-102": ("DOS-550", "glacier-deep"),
    "TC-103": ("Berlin",),
}

_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)


def is_eligible(scenario_id: str) -> bool:
    return scenario_id in ANCHORS


@functools.cache
def load_pack(locale: str) -> dict[str, Any]:
    """The checked-in translation pack for ``locale``."""
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(
            f"Unsupported locale {locale!r}; choose one of {', '.join(SUPPORTED_LOCALES)}."
        )
    text = resources.files(__package__).joinpath(f"{locale}.json").read_text(encoding="utf-8")
    pack: dict[str, Any] = json.loads(text)
    return pack


def tool_strings(tools: list[dict[str, Any]]) -> list[str]:
    """Every natural-language description in ``tools``, in schema order."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(tools)
    return found


def localize_tools(tools: list[dict[str, Any]], strings: dict[str, str]) -> list[dict[str, Any]]:
    """A copy of ``tools`` with every description translated.

    Names, parameter keys, types, and enum values are left alone: they are the
    contract the mock handlers check.
    """
    missing = [text for text in tool_strings(tools) if text not in strings]
    if missing:
        raise ValueError(f"Translation pack lacks {len(missing)} tool strings, e.g. {missing[0]!r}")
    localized = copy.deepcopy(tools)
    # Scenario tool lists reuse one parameter dict across tools, and deepcopy
    # keeps that sharing; translate each node once.
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, (dict, list)) or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    node[key] = strings[value]
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(localized)
    return localized


def scenario_tools(scenario: ScenarioDefinition) -> list[dict[str, Any]]:
    return UNIVERSAL_TOOLS if scenario.tools_override is None else scenario.tools_override


# --- answer normalization --------------------------------------------------

_FULLWIDTH = str.maketrans(
    {**{chr(0xFF10 + d): str(d) for d in range(10)}, "％": "%", "＄": "$", "．": "."}
)
# 1.234.567,89 -> 1234567.89 and 1.020 -> 1020 for comma-decimal locales.
_DOT_GROUPED = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+)(?:,(\d+))?(?!\d|[.,]\d)")
# 3,14 -> 3.14. Exactly three digits after the comma stays ambiguous and is
# left for the shared helpers, which read it as a thousands separator.
_DECIMAL_COMMA = re.compile(r"(?<![\d.,])(\d+),(\d{1,2}|\d{4,})(?!\d|[.,]\d)")
# 1020 $ or 1.020 € -> $1020: the shared helpers expect a leading symbol.
_TRAILING_DOLLAR = re.compile(r"(\d[\d.,]*)\s?(?:\$|US-?\$|USD|美元|美金|ドル)(?![A-Za-z])")

_NEGATORS: dict[str, re.Pattern[str]] = {
    "de": re.compile(r"\b(?:nicht|kein(?:e|en|em|er|es)?|nie(?:mals)?|weder)\b", re.IGNORECASE),
    "es": re.compile(r"\b(?:no|nunca|jamás|ningun[oa]?|ningún|ni|tampoco)\b", re.IGNORECASE),
    # Chinese negation precedes what it negates, as English does. Guard against
    # compound words ("无理数", "无限不循环", "不需要", "非幂等", "不再", "不超过", "非常").
    "zh": re.compile(
        r"不是|并不是|并非|不等于|没到|不到"
        r"|没有(?![别其任何问题错])"
        r"|(?<![毫绝并])不(?![同用需要断过错仅但只论管妨久再可相符含超满少低于足能会是限循环难])"
        r"|(?<![毫绝])没(?![有事错问题法])"
    ),
    # Japanese negation follows the value ("30ではありません"), so the guard's
    # look-back cannot see it; only the short look-ahead some helpers use can.
    # docs/locales.md records the limit.
    "ja": re.compile(r"ではありません|じゃない|ではない|ません|ない"),
}
_CONSTITUENT_NEGATORS: dict[str, re.Pattern[str]] = {
    "de": re.compile(r"\bohne\b", re.IGNORECASE),
    "es": re.compile(r"\bsin\b(?!\s+(?:duda|embargo)\b)", re.IGNORECASE),
}
_COMMA_DECIMAL_LOCALES = frozenset({"de", "es"})


def normalize_answer(text: str, locale: str) -> str:
    """Rewrite ``text`` so English-oriented number and negation parsing applies."""
    text = text.translate(_FULLWIDTH)
    if locale in _COMMA_DECIMAL_LOCALES:
        text = _DOT_GROUPED.sub(
            lambda m: m.group(1).replace(".", "") + (f".{m.group(2)}" if m.group(2) else ""),
            text,
        )
        text = _DECIMAL_COMMA.sub(r"\1.\2", text)
    text = _TRAILING_DOLLAR.sub(r"$\1", text)
    constituent = _CONSTITUENT_NEGATORS.get(locale)
    if constituent is not None:
        text = constituent.sub(" without ", text)
    negator = _NEGATORS.get(locale)
    if negator is not None:
        text = negator.sub(" not ", text)
    return text


def localized_state(state: ScenarioState, locale: str) -> ScenarioState:
    result = copy.copy(state)
    result.final_answer = normalize_answer(state.final_answer, locale)
    result.assistant_messages = [normalize_answer(m, locale) for m in state.assistant_messages]
    return result


# --- language detection for the drift diagnostic ---------------------------

_KANA = re.compile(r"[\u3040-\u30ff]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_STRIP = re.compile(
    r"```.*?```|`[^`]*`|https?://\S+|[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+"
    r"|(?<![A-Za-z0-9_])[A-Za-z0-9_./-]*\d[A-Za-z0-9_./-]*(?![A-Za-z0-9_])",
    re.DOTALL,
)
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the and is are of to in it for with that this you your was were be have has on".split()
    ),
    "de": frozenset(
        "der die das und ist sind nicht ich mit für auf den dem ein eine zu von sie wir es".split()
    ),
    "es": frozenset("el la los las y es son de que en para con por una un su se lo del al".split()),
}


def detect_language(text: str) -> str:
    """A coarse guess among en, de, es, zh, ja; ``unknown`` when unsure.

    Identifiers, code, and numbers are removed first so the invariant anchors
    a correct localized answer repeats do not read as English.
    """
    text = _STRIP.sub(" ", text)
    if _KANA.search(text):
        return "ja"
    if len(_HAN.findall(text)) >= 2:
        return "zh"
    words = re.findall(r"[a-zäöüßáéíóúñ]+", text.lower())
    hits = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORDS.items()}
    best = max(hits, key=lambda lang: hits[lang])
    ranked = sorted(hits.values(), reverse=True)
    if ranked[0] >= 2 and ranked[0] > ranked[1]:
        return best
    return "unknown"


def _language_diagnostic(state: ScenarioState, locale: str) -> str | None:
    replies = [m for m in state.assistant_messages if m and m.strip()]
    if not replies:
        return None
    first, last = detect_language(replies[0]), detect_language(replies[-1])
    note = f"requested={locale} first={first} final={last}"
    if first == locale and last not in (locale, "unknown"):
        note += " (drifted)"
    return note


# --- scenario transformation -----------------------------------------------


def localized_variant(scenario: ScenarioDefinition, locale: str) -> ScenarioDefinition:
    if not is_eligible(scenario.id):
        raise ValueError(f"{scenario.id} is not eligible for locale runs.")
    pack = load_pack(locale)
    entry = pack["scenarios"][scenario.id]
    original_evaluate = scenario.evaluate

    def evaluate(state: ScenarioState) -> ScenarioEvaluation:
        evaluation = original_evaluate(localized_state(state, locale))
        diagnostic = _language_diagnostic(state, locale)
        if diagnostic is None:
            return evaluation
        return replace(evaluation, diagnostics={**evaluation.diagnostics, "language": diagnostic})

    return replace(
        scenario,
        user_message=entry["user_message"],
        follow_up_messages=list(entry.get("follow_up_messages", [])),
        tools_override=localize_tools(scenario_tools(scenario), pack["strings"]),
        evaluate=evaluate,
        variant_metadata={
            **scenario.variant_metadata,
            "locale": locale,
            "locale_version": LOCALE_VERSION,
        },
    )


def apply_locale(
    scenarios: list[ScenarioDefinition], locale: str | None
) -> list[ScenarioDefinition]:
    """Localize every eligible scenario and drop the rest.

    Apply before ``apply_variants`` so seeded fixtures rewrite the translated
    prompt; the anchors guarantee the literals they replace are still there.
    """
    if locale is None:
        return scenarios
    load_pack(locale)
    return [localized_variant(s, locale) for s in scenarios if is_eligible(s.id)]


def ineligible_ids(scenarios: list[ScenarioDefinition]) -> list[str]:
    return [s.id for s in scenarios if not is_eligible(s.id)]


def fenced_blocks(text: str) -> list[str]:
    """Fenced code blocks, which translations must reproduce byte for byte."""
    return _FENCED_BLOCK.findall(text)
