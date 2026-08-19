**TC-52 stock fixture coherence** — `get_stock_price` enrichment now derives
`previous_close` from the declared `change` field when one is present
(`change = price - previous_close`), instead of always applying a hardcoded
`price - 1.23` offset. TC-52's AAPL fixture previously returned
`price 178.50`, `previous_close 177.27`, and `change -2.30`, which are
mathematically incompatible; it now returns `previous_close 180.80`
(`178.50 + 2.30`), consistent with `change -2.30` and `change_percent
-1.27%`. A fixture-integrity regression test verifies the change, percentage,
sign/direction, and evaluator-visible numbers agree with the mock response.

**This changes the TC-52 mock response.** Models that reported the old
`177.27` previous close will now see `180.80`; benchmark results produced
before this change are therefore **not comparable** with results produced
after it for identical model behaviour.
