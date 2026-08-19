**TC-54 cross-tool synthesis verdict contract** — the evaluator now states a
single, truthful policy for the partial path: calculator use is mandatory.
When both data sources are retrieved but the calculator was never called, the
verdict says the conversion was not verified with the calculator instead of
claiming the stated sum "may be imprecise" (a false diagnostic for an exact,
correct figure). When a calculator call exists but does not verify the
USD/JPY conversion, the verdict names the mismatch explicitly. The PASS path
still requires a correct reasonable result, so the score and the reason now
always agree.
