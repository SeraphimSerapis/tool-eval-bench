**TC-69 checks the briefing date, and TC-40's customer profile lists one order.** TC-69 now
requires the briefing's `date` to be today's reference date (2026-03-20, or the `--reference-date`
value), since the system prompt gives the model that date. A briefing dated anything else is
PARTIAL. TC-40's `get_customer_profile` mock listed a second order, `ORD-2026-1512`, that
`get_order_status` always reported as not found, so checking both listed orders scored PARTIAL. The
profile now lists only `ORD-2026-1847`. TC-69 scores can drop for models that invent a date.
