**TC-75 missing-parameter scoring (scoring)** — need-style and without/don't-
know clarification requests ("I need the date and time", "I cannot book
without the date and time") now PASS. Negated markers, quoted echoes, and
meta attributions ("I do not need…", "Just repeating: 'I need…'", "The report
says 'I need…'") stay FAIL. Request detection uses match-local negation/meta/
quote filters on explicit request constructions (not a global date/time veto),
so genuine questions like "The document lists rooms; what date and time…" keep
PASSING. The no-guessing contract (any tool call before clarification fails)
is unchanged.
