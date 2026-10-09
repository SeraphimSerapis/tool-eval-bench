**Standard scenarios grade retries, number spellings, and array recipients consistently.** A
correct retry after a tool error now passes: TC-15 grades the first `web_search` and `calculator`
calls that did not error, so a calculator syntax error or an `--error-rate` failure followed by a
good retry is no longer FAIL or "used background knowledge". A calculator call that succeeded with
a rounded population still fails. TC-01 and TC-02 pass a retry after an error and give a redundant
repeat of the correct call PARTIAL instead of FAIL. TC-20 passes a sum-then-divide calculator route
and names the real shortfall in its PARTIAL summary. Stated numbers must match the tool value at the
precision written: TC-02 rejects "$187.99" for a $187.42 price, TC-09 accepts "$412.8" and rejects
"$412.99", and TC-61 rejects a wrong anomaly count such as "13 anomalies" even beside the right
record count. TC-07 and TC-38 accept "$4,400,000" for the $4.4M total. TC-24 scores "$4.25 million"
as PARTIAL instead of a wrong value. TC-35 reads "500 kelvins" like "500 kelvin". TC-03, TC-07,
TC-38 and TC-56 accept a one-element array in `to`; a second recipient still fails. TC-03 no longer
treats "The room has not changed" as denying the move. TC-16 ends a German denial at "und", "aber",
"sondern", "bei", "mit" and commas, so "es regnet nicht und es hat 14 °C" passes. TC-18 accepts every
schema English source (`en-us`, `en-gb`, `en-ca`, `en-au`) and no longer reports an unusable
translation as "out of order". TC-49 passes a send offer that waits on a new user instruction ("If
you change your mind, just say so and I'll send it"); "Once you've reviewed it, I'll send it" is
still PARTIAL. TC-56 says the reminder was missing instead of "didn't note the freezing condition".
TC-64 accepts a zero-fraction year such as `1999.0`, which JSON Schema counts as an integer. Scores
on these scenarios can rise for correct recoveries and variants and drop for wrong decimals or
counts.
