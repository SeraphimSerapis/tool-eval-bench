`_TC63_PRICE` stopped its number body at the first non-digit, so TC-63 compared a PREFIX of the
price rather than the price: `$1,200` was read as `$1` and `$30.99` as `$30`, both of which clear
the `$30` ceiling. Because `answer_affirms_number` collapses digit grouping, the stray `1` in
"table for 1" was enough to affirm the truncated figure, and a `$1,200 per person` recommendation
scored PASS under a verdict reading "Maintained all accumulated constraints". The pattern now
reads grouped digits and cents, and the ceiling is tested on the whole amount.

**This moves scores in both directions.** An over-budget amount written with grouping, cents or
leading zeros loses the constraint, which is the intent. And because the value looked up in the
answer is built from the amount, removing the truncation also changes that lookup wherever the
truncation changed the amount — that is, for whole parts longer than the old three-digit cap.
`$0025` was looked up as `2` and is now looked up as `25`, so it gains or loses the constraint
depending on which number the sentence affirms; `$007` and `$030` are unaffected. Both directions
are pinned by tests. Keeping the truncated capture for the lookup would avoid the movement
entirely, at the cost of leaving the same prefix bug in the affirmation half — the wider lookup is
a deliberate choice and can be reversed if you would rather the published numbers not move.
