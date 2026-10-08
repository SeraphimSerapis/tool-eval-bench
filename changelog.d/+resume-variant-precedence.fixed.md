**Resumed `--variant-seed` runs keep their variants.** Scenarios whose outcomes were preserved from
the interrupted half of a run were merged under their unvarianted registry definitions, so the
completed run's stored `scenario_variants` omitted them and its `config_fingerprint` no longer
matched an uninterrupted run with the same seed. Resume now merges each preserved outcome under the
definition that produced it, so a resumed run lands in the same leaderboard cohort as a fresh one.
Scores are unchanged. Runs without `--variant-seed` are unaffected.
