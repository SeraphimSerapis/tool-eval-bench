**Spec-bench acceptance on llama.cpp and Strata now comes from the request.** Both servers return
each request's draft counts in the response `timings` (`draft_n`, `draft_n_accepted`), but
spec-bench preferred the server-wide `/metrics` delta whenever one existed, so another client
drafting during a measurement skewed α and set off the cross-traffic warning. The response counts
now win, and the summary and report name the source as per-request response timings. On a quiet
server the numbers are unchanged: a live llama.cpp run read 154 drafted and 56 accepted from both
sources. Timings carry no step count, so llama.cpp τ, draft window, and Steps/s still come from the
`/metrics` step delta, used only when its draft and accepted deltas match the request exactly. When
other traffic breaks the match they show as unknown instead of wrong. llama.cpp omits `draft_n`
exactly when a request drafted nothing, so a llama.cpp response with `timings` but no `draft_n`
now counts as zero drafts instead of falling back to the server-wide delta. With `--spec-runs`
above 1, pooled τ and draft window come only from the runs that got a step count, so a run whose
step count was refused no longer inflates τ. A run in which nothing was drafted no longer names an
acceptance source in the report. Current llama.cpp exports its
draft counters even with no draft model, so detection now reports speculative decoding as active
only once those counters are non-zero. `--spec-live` labels Strata's counters `strata` instead of
`vllm` and shows its method as MTP.
