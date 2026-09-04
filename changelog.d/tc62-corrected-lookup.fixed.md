**TC-62 counts a corrected lookup by the file it returns, not by query tokens.** A model that
searched "quarterly performance" (the prompt's own phrase), read the returned
`Q3_Report_v2_CORRECTED.xlsx`, and used the corrected `$4,150,000` everywhere is now credited for
the corrected lookup even though the query carried none of the literal `latest`/`q3`/`corrected`
tokens the evaluator previously demanded. The email body also accepts the competitor amount
spelled `$3,800,000` alongside the `3.8`/`3.8M` forms, and figures stated under a negation ("we are
NOT ahead") no longer satisfy the compare contract. Two emails to the CFO still fall back to
PARTIAL under the single-safe-email contract.
