**Locale runs with `--locale de|es|zh|ja`.** The user messages and tool descriptions of 26
language-neutral scenarios can now be sent in German, Spanish, Simplified Chinese, or Japanese
from checked-in translation packs, while identifiers, schemas, invariant anchors, and English mock
data stay fixed. Answers are normalized for locale number formats and negation before the usual
evaluators run, and each result records the first and final reply language so drift back into
English shows up in the report. Scenarios that grade English prose are skipped. The locale is part
of the run's fingerprint, so locale runs form their own leaderboard cohort and are not comparable
with English runs. The Python API takes `locale=`. Adapted from QwenLM BabelArena's localized task
families and its language-consistency audit. See `docs/locales.md`.
