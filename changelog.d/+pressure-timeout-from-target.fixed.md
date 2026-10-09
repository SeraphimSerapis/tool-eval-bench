**Context-pressure runs get the same timeout every time.** `--context-pressure` raises the request
timeout for large fills, and it scaled that timeout from the calibrated fill. Unseeded filler
calibrates to a slightly different token count on every run, so the stored `timeout_seconds` moved
by fractions of a second between identical runs. Because `timeout_seconds` is part of the comparison
fingerprint, two unseeded pressure runs of one configuration never landed in the same cohort, and
`resume` refused them with a `timeout_seconds` mismatch. The timeout now scales from the fill
target, which only the context size and ratio decide, as `--context-pressure-sweep` already did.

Pressure runs saved before this fix whose timeout was raised land in a different comparison cohort
from new runs, seeded or not. For the same reason, an interrupted pressure run from before the fix
whose timeout was raised will usually be refused by `resume`; start a fresh run instead. Runs whose
`--timeout` was already above the raised value are unaffected.
