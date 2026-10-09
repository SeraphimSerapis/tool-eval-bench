**An accuracy run that selects nothing now fails instead of saving 0%.** A GSM8K, MMLU, or IFEval
run whose filters left no items used to persist a 0/0 result rated Poor. It now exits with an
error. An unknown `--mmlu-subjects` name, or a list with no names, fails before any download and lists
the valid subjects and categories. Subject and category names match in any case, and the stored
value is normalised, so `stem` and `STEM` share a fingerprint.
