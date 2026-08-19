**TC-55 branch ordering** — in `_tc55_eval`, the broad
`searched and (read_na or read_emea) and has_total` branch shadowed the
both-files case: reading **both** regional files and producing the correct
total without a calculator call was reported as *"only read one of two
files"*. A dedicated `searched and read_na and read_emea and has_total`
branch now precedes the `or`-subset, so the reason reflects the actual
trace. Regression test `test_partial_both_files_total_no_calculator`
covers the case.
