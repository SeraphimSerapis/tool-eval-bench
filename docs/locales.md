# Locale runs

`--locale` runs the benchmark with the user's side of the conversation in another language:
German (`de`), Spanish (`es`), Simplified Chinese (`zh`), or Japanese (`ja`).

```bash
tool-eval-bench run --locale de
tool-eval-bench run --hardmode --locale ja --variant-seed 1
```

The Python API takes `locale="de"` on `run_benchmark`.

## What changes

- The user message and any follow-up messages come from a checked-in translation pack,
  `src/tool_eval_bench/evals/locales/<locale>.json`.
- Every natural-language description in the tool schemas, for both the tool and its parameters,
  is translated too.
- Everything a mock or an evaluator checks stays the same: tool and parameter names, types, enum
  values, identifiers, tickers, file names, code, and JSON schemas in fenced blocks. Each scenario
  also lists invariant anchors in `ANCHORS` (`evals/locales/__init__.py`), the proper nouns its mock
  matches, such as `Tokyo` or `Project Alpha`. Every translation must contain those verbatim.
- Mock tool results stay in English. The model reads a translated request, translated tool
  documentation, and English data, which is what it sees against a real English backend.
- The system prompt and the benchmark's date line stay in English.

## Which scenarios run

A locale run includes only scenarios whose grading does not depend on English prose. Everything
else is skipped, so locale scores cover a smaller set than English runs and are not comparable with
them. The locale and pack version are part of the scenario variant metadata, so each locale gets
its own `config_fingerprint` cohort, and resume rejects a different locale.

A scenario is left out when:

- its evaluator reads English discourse: refusals, clarification questions, explanations, claims
  of success, or Markdown prose structure;
- its mock or evaluator matches an ordinary English noun inside an argument the model writes, such
  as a search query (`population iceland`), an email body, or a reminder text, which a model writing
  in German would reasonably translate;
- it tests language itself (the translation and German-response scenarios);
- a fixture variant replaces its evaluator in a way that skips the locale wrapper (TC-61).

The current list is the 26 IDs in `ANCHORS`. Adding a scenario means adding its anchors and a
translation to all four packs; `tests/test_locales.py` checks that every pack covers every eligible
scenario and tool string, keeps every anchor and fenced block, and that each scenario's reference
trace still passes under every locale.

## How answers are graded

The scenario's own evaluator grades the run. Before it sees the answer, the locale layer rewrites
two things the shared helpers parse with English rules:

- **Numbers.** Full-width digits become ASCII. For `de` and `es`, dot-grouped thousands
  (`4.250.000`) lose their dots and a decimal comma (`187,42`) becomes a point. A comma followed by
  exactly three digits (`1,050`) is ambiguous and is left alone, which the helpers read as a
  thousands separator. A trailing dollar sign (`1.020 $`) moves to the front.
- **Negation.** Locale negation words become `not`, so "nicht 30" or "不是30" does not count as an
  answer of 30. Japanese puts negation after the value (`30ではありません`), where the helpers'
  look-back does not reach, so a negated Japanese number can still read as affirmed. This is a known
  limit.

## Language drift

Models often start in the user's language and slide back into English after a few turns of English
tool output. Each locale result records the detected language of the first and the last reply under
the `language` diagnostic, for example `requested=de first=de final=en (drifted)`. It appears in the
report's capability diagnostics and does not change the score.

Detection is deliberately coarse: kana means Japanese, Han characters mean Chinese, and Latin text is
scored against short English, German, and Spanish stopword lists after identifiers, code, and numbers
are stripped. Short or mixed replies come back as `unknown` rather than a guess.

## Maintaining the packs

The packs were drafted by a model and need a native-speaker review before their results carry
weight. When an English prompt or tool description changes, the pack test fails with the missing or
stale string; update all four packs in the same change and bump `LOCALE_VERSION` when a translation's
meaning changes, so old and new runs do not share a cohort.
