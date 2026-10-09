**`--mmlu-limit` now samples every subject.** The MMLU test split is sorted by subject, so a limited
run took the first N rows and the default 500 covered only the first few subjects alphabetically.
A limited run now takes a proportional stratified sample: the `--mmlu-subjects` filter applies
first, each subject gets a share of the limit proportional to its size, and each contributes its
first questions in dataset order. The selection is deterministic and ignores `--seed`. Runs record
`"sampling": "stratified"`. Limited MMLU scores from earlier versions measured a different,
narrower question set and are not comparable.
