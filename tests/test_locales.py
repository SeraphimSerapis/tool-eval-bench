"""Locale runs: translation packs, answer normalization, and end-to-end grading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from scenario_replay import SCENARIOS, replay, turn

from tool_eval_bench.cli.commands import resolve_scenarios
from tool_eval_bench.domain.scenarios import ScenarioState, ScenarioStatus
from tool_eval_bench.evals.helpers import (
    answer_affirms_number,
    answer_affirms_text,
    positive_argument_contains,
)
from tool_eval_bench.evals.locales import (
    ANCHORS,
    SUPPORTED_LOCALES,
    apply_locale,
    detect_language,
    fenced_blocks,
    load_pack,
    localized_variant,
    normalize_answer,
    scenario_tools,
    tool_strings,
)
from tool_eval_bench.evals.variants import prepare_scenarios, substitute

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/scenario_reference_traces.json").read_text(encoding="utf-8")
)


def _required_strings() -> set[str]:
    return {text for sid in ANCHORS for text in tool_strings(scenario_tools(SCENARIOS[sid]))}


# --- pack integrity, one set per locale -------------------------------------


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_pack_covers_exactly_the_eligible_scenarios(locale: str) -> None:
    pack = load_pack(locale)
    assert pack["locale"] == locale
    assert set(pack["scenarios"]) == set(ANCHORS)


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_pack_translates_every_tool_string_and_nothing_stale(locale: str) -> None:
    strings = load_pack(locale)["strings"]
    required = _required_strings()
    assert sorted(required - strings.keys()) == []
    assert sorted(strings.keys() - required) == []
    assert all(isinstance(value, str) and value.strip() for value in strings.values())


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
@pytest.mark.parametrize("sid", sorted(ANCHORS))
def test_translation_keeps_anchors_fences_and_follow_up_count(locale: str, sid: str) -> None:
    english = SCENARIOS[sid]
    entry = load_pack(locale)["scenarios"][sid]
    translated = "\n".join([entry["user_message"], *entry.get("follow_up_messages", [])])
    assert len(entry.get("follow_up_messages", [])) == len(english.follow_up_messages)
    assert entry["user_message"] != english.user_message, "left untranslated"
    missing = [anchor for anchor in ANCHORS[sid] if anchor not in translated]
    assert missing == []
    assert fenced_blocks(entry["user_message"]) == fenced_blocks(english.user_message)


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_translated_prompts_do_not_read_as_english(locale: str) -> None:
    detected = {
        sid: detect_language(entry["user_message"])
        for sid, entry in load_pack(locale)["scenarios"].items()
    }
    assert [sid for sid, lang in detected.items() if lang == "en"] == []


# --- end to end: reference traces still grade under every locale -------------


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
@pytest.mark.parametrize("sid", sorted(ANCHORS))
def test_reference_trace_passes_under_locale(locale: str, sid: str) -> None:
    scenario = localized_variant(SCENARIOS[sid], locale)
    turns = [
        turn(*[(c["name"], c["arguments"]) for c in step["calls"]], answer=step["answer"])
        for step in REFERENCE[sid]
    ]
    result = replay(scenario, *turns)
    assert result.status is ScenarioStatus.PASS, result.summary


def test_localized_scenario_sends_translated_text_and_unchanged_contract() -> None:
    english = SCENARIOS["TC-102"]
    german = localized_variant(english, "de")
    assert german.user_message == load_pack("de")["scenarios"]["TC-102"]["user_message"]
    assert german.tools_override is not None and english.tools_override is not None
    names = [t["function"]["name"] for t in german.tools_override]
    assert names == [t["function"]["name"] for t in english.tools_override]
    params = [t["function"]["parameters"]["properties"].keys() for t in german.tools_override]
    assert params == [
        t["function"]["parameters"]["properties"].keys() for t in english.tools_override
    ]
    assert tool_strings(german.tools_override) != tool_strings(english.tools_override)


def test_universal_tool_scenarios_get_a_translated_copy() -> None:
    german = localized_variant(SCENARIOS["TC-02"], "de")
    assert SCENARIOS["TC-02"].tools_override is None
    assert german.tools_override is not None
    assert all(
        text in load_pack("de")["strings"].values() for text in tool_strings(german.tools_override)
    )


# --- selection and composition ----------------------------------------------


def test_apply_locale_drops_ineligible_scenarios() -> None:
    chosen = [SCENARIOS["TC-02"], SCENARIOS["TC-12"], SCENARIOS["TC-102"]]
    assert [s.id for s in apply_locale(chosen, "es")] == ["TC-02", "TC-102"]
    assert apply_locale(chosen, None) == chosen


def test_unknown_locale_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported locale"):
        apply_locale([SCENARIOS["TC-02"]], "fr")


def test_locale_composes_with_fixture_variants() -> None:
    (varied,) = prepare_scenarios([SCENARIOS["TC-24"]], variant_seed=5, locale="ja")
    assert varied.variant_metadata["locale"] == "ja"
    assert varied.variant_metadata["seed"] == 5
    assert varied.variant_metadata["kind"] == "identifiers"
    assert varied.user_message == load_pack("ja")["scenarios"]["TC-24"]["user_message"]


def test_locale_changes_the_variant_metadata_that_feeds_the_fingerprint() -> None:
    plain = prepare_scenarios([SCENARIOS["TC-02"]], variant_seed=None)
    german = prepare_scenarios([SCENARIOS["TC-02"]], variant_seed=None, locale="de")
    assert plain[0].variant_metadata == {}
    assert german[0].variant_metadata == {"locale": "de", "locale_version": 1}


# --- answer normalization -----------------------------------------------------


@pytest.mark.parametrize(
    ("locale", "answer", "value"),
    [
        ("de", "Der Kurs liegt bei 187,42 $.", "187.42"),
        ("de", "Gesamtumsatz: 4.250.000", "4250000"),
        ("es", "El total final es 1.020 $ con LAB100.", "1020"),
        ("es", "Pi vale aproximadamente 3,14.", "3.14"),
        ("zh", "结果是３０。", "30"),
        ("ja", "合計は935ドルです。", "935"),
    ],
)
def test_locale_numbers_are_read_as_the_english_helpers_expect(
    locale: str, answer: str, value: str
) -> None:
    assert answer_affirms_number(normalize_answer(answer, locale), value)


@pytest.mark.parametrize(
    ("locale", "answer"),
    [
        ("de", "Es sind nicht 30, sondern 40."),
        ("es", "Nunca fue 30."),
        ("es", "El resultado no es 30 sino 40."),
        ("zh", "答案不是30。"),
    ],
)
def test_locale_negation_still_blocks_a_denied_value(locale: str, answer: str) -> None:
    assert not answer_affirms_number(normalize_answer(answer, locale), "30")


@pytest.mark.parametrize(
    ("locale", "answer", "value"),
    [
        ("de", "Ohne Tools ist es 30.", "30"),
        ("de", "Es sind nicht 20, sondern 30.", "30"),
        ("es", "Sin herramientas es 30.", "30"),
        ("es", "El resultado es sin duda 30.", "30"),
        ("es", "El resultado no es 20 sino 30.", "30"),
        ("zh", "不需要使用工具，200的15%是30。", "30"),
        ("zh", "计算结果非常明确：30", "30"),
        ("zh", "购物车里没有其他商品，最终总价为 1020 美元，优惠码是 LAB100。", "1020"),
    ],
)
def test_locale_constituent_and_compound_words_do_not_falsely_negate(
    locale: str, answer: str, value: str
) -> None:
    assert answer_affirms_number(normalize_answer(answer, locale), value)


@pytest.mark.parametrize(
    ("locale", "answer", "identifier"),
    [
        ("ja", "アーカイブ受領IDはARC-2026-8841です。", "ARC-2026-8841"),
        ("ja", "最終合計は1,020ドルで、適用したクーポンコードはLAB100です。", "LAB100"),
        ("ja", "クレジットIDはcrd_9001、crd_9002、crd_9003です。", "crd_9001"),
        ("zh", "归档回执ID为ARC-2026-8841。", "ARC-2026-8841"),
        ("zh", "由于接口非幂等，仅重试了失败账户，赔付ID为crd_9001。", "crd_9001"),
        ("zh", "购物车里已没有别的东西，使用优惠码LAB100后最终总价为1,020美元。", "LAB100"),
    ],
)
def test_ascii_identifiers_adjacent_to_cjk_are_affirmed(
    locale: str, answer: str, identifier: str
) -> None:
    assert answer_affirms_text(normalize_answer(answer, locale), identifier)


def test_cjk_adjacent_arguments_and_variant_substitutions_match() -> None:
    assert positive_argument_contains("Team Standup会议", "standup")
    assert positive_argument_contains("Team Standupミーティング", "standup")
    assert (
        substitute("已读取文件file_405。", {"file_405": "fixture_123"}) == "已读取文件fixture_123。"
    )


def test_three_digit_comma_group_is_left_for_the_thousands_reading() -> None:
    assert normalize_answer("1,050", "de") == "1,050"


def test_dollar_suffix_moves_to_the_front() -> None:
    assert normalize_answer("4250000 $", "de") == "$4250000"
    assert normalize_answer("4,250,000 美元", "zh") == "$4,250,000"
    assert normalize_answer("935ドル", "ja") == "$935"


def test_localized_scenario_evaluators_accept_natural_target_language_answers() -> None:
    tc44_zh = localized_variant(SCENARIOS["TC-44"], "zh")
    assert (
        tc44_zh.evaluate(
            ScenarioState(final_answer="圆周率（π）是一个无理数，约等于3.14159。")
        ).status
        is ScenarioStatus.PASS
    )
    tc44_es = localized_variant(SCENARIOS["TC-44"], "es")
    assert (
        tc44_es.evaluate(
            ScenarioState(final_answer="El valor de pi es aproximadamente 3,142.")
        ).status
        is ScenarioStatus.PASS
    )
    tc64_zh = localized_variant(SCENARIOS["TC-64"], "zh")
    assert (
        tc64_zh.evaluate(
            ScenarioState(
                final_answer='{"title":"黑客帝国","year":1999,"rating":9,"genre":"sci-fi","summary":"经典科幻电影。"}'
            )
        ).status
        is ScenarioStatus.PASS
    )
    for locale, assignee in [("de", "mir"), ("es", "yo"), ("zh", "我"), ("ja", "私")]:
        tc68 = localized_variant(SCENARIOS["TC-68"], locale)
        payload = json.dumps({"task_id": "PROJ-127", "status": "in_progress", "assignee": assignee})
        assert tc68.evaluate(ScenarioState(final_answer=payload)).status is ScenarioStatus.PASS


# --- language detection and the drift diagnostic --------------------------------


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("The weather in London is overcast and the price is shown below.", "en"),
        ("Das Wetter in London ist bewölkt und der Kurs ist nicht gestiegen.", "de"),
        ("El clima en London es nublado y el precio de la acción es alto.", "es"),
        ("伦敦今天多云，微软的股价是多少？", "zh"),
        ("AAPL现在的股价是187.42美元。", "zh"),
        ("ロンドンの天気は曇りです。", "ja"),
        ("Tokyoの気温は華氏64度です。", "ja"),
        ("AAPL 187.42 USD", "unknown"),
    ],
)
def test_detect_language(text: str, language: str) -> None:
    assert detect_language(text) == language


def test_drift_back_to_english_is_recorded_without_changing_the_grade() -> None:
    scenario = localized_variant(SCENARIOS["TC-45"], "de")
    state = ScenarioState(
        assistant_messages=[
            "Das ist eine einfache Rechnung, die ich ohne Werkzeug lösen kann.",
            "The answer is 56, and it is the product of the two numbers.",
        ],
        final_answer="The answer is 56, and it is the product of the two numbers.",
    )
    evaluation = scenario.evaluate(state)
    english = SCENARIOS["TC-45"].evaluate(state)
    assert (evaluation.status, evaluation.points) == (english.status, english.points)
    assert evaluation.diagnostics["language"] == "requested=de first=de final=en (drifted)"


# --- CLI selection ---------------------------------------------------------------


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "scenarios": None,
        "categories": None,
        "short": False,
        "hardmode": False,
        "hardmode_only": False,
        "pack_only": False,
        "scenario_pack": None,
        "locale": None,
    }
    return argparse.Namespace(**{**base, **overrides})


def test_cli_selection_keeps_only_locale_scenarios_so_previews_match_the_run() -> None:
    chosen = resolve_scenarios(_args(hardmode=True, locale="de"))
    assert [s.id for s in chosen] == [s for s in sorted(ANCHORS, key=lambda i: int(i[3:]))]


def test_cli_selection_rejects_an_explicitly_named_ineligible_scenario() -> None:
    with pytest.raises(ValueError, match="does not support TC-12"):
        resolve_scenarios(_args(scenarios=["TC-02", "TC-12"], locale="de"))


def test_cli_selection_rejects_a_selection_with_nothing_to_run() -> None:
    with pytest.raises(ValueError, match="None of the selected scenarios"):
        resolve_scenarios(_args(categories=["K"], locale="zh"))


def test_cli_selection_is_unchanged_without_a_locale() -> None:
    assert len(resolve_scenarios(_args())) == 69


def test_cli_locale_flag_reaches_the_service_and_the_recorded_config() -> None:
    from tool_eval_bench.cli.bench import _make_parser
    from tool_eval_bench.cli.scored_run import ScoredRun

    args = _make_parser().parse_args(["--locale", "ja", "--scenarios", "TC-02"])
    scenarios = resolve_scenarios(args)
    run = ScoredRun.from_args(
        args,
        model="m",
        backend="vllm",
        base_url="http://localhost:1/v1",
        api_key=None,
        wire_format="openai",
        scenarios=scenarios,
        extra_params=None,
        scenario_packs=None,
    )
    assert run.service_kwargs()["locale"] == "ja"
    (recorded,) = run.config_scenarios()
    assert recorded.variant_metadata["locale"] == "ja"


def test_cli_rejects_an_unsupported_locale() -> None:
    from tool_eval_bench.cli.bench import _make_parser

    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--locale", "fr"])
