`spec-live` now plots a 30-second rolling window of acceptance on the gauge and sparklines
instead of the session running average, which converged and then hid workload changes; the
session α moved to the grid and the exit summary, where it is the pooled counter ratio rather
than a mean of running means. A failed scrape turns the header red and dates the on-screen
numbers instead of leaving a green spinner over a dead server. Per-position rates divide by
vLLM's `spec_decode_num_draft_tokens_per_pos` when exported, so variable-length drafters are no
longer under-reported at later positions, and the draft window and inferred `k` are
session-relative like the rates they sit next to. The poll interval is honoured when stdin is
not a TTY; before, piped output spun the loop at full speed. The subtitle and history title
reflect `--spec-live-interval`.
