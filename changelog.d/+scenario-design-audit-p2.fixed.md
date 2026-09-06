Made scenarios that test the same thing agree with each other. A single shared
matcher now compares `location` arguments, so "Berlin, DE" is Berlin in TC-22,
TC-25, TC-27, TC-65, TC-69 and TC-79 as it already was in TC-01. A shared
`time_matches` helper lets TC-17 accept the formats TC-05 accepts, and TC-17 now
names the field that was actually wrong instead of blaming the timezone. TC-38
uses TC-07's number check, so "$4.4 million" scores like "$4.4M". TC-31, TC-33
and TC-50 use the shared clarification and refusal helpers instead of narrower
per-scenario word lists. TC-34 and TC-73 derive provenance from what the search
returned rather than from how the model worded the query, and TC-73 names the
steps it found missing. TC-03 accepts any way of saying the meeting moved, TC-23
any verb describing what a function does, TC-40 an order id resolved from a prior
lookup, and TC-66 any query string naming engineering. TC-63 gained a turn budget
for its five user messages.
