**TC-56 semantic reminder time (scoring)** — `set_reminder` now also accepts
an ISO `datetime` that resolves to the *next calendar day* in a documented
morning window (05:00 inclusive to 12:00 exclusive) relative to the scenario
reference date. Literal `"tomorrow morning"` text remains accepted for
backward compatibility. Timezone offsets/`Z` are ignored (calendar date +
hour only, same ignore-offset idea as `datetime_matches`), and month/year
rollover is handled. This is an additive scoring change: correct next-day
morning ISO timestamps that previously failed the literal substring gate can
now PASS; outside-window, wrong-day, malformed, and missing datetimes stay
PARTIAL as before.
