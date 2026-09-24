**Guessed addresses, stale recoveries, and contradicting replies no longer pass.** Scores for TC-18,
TC-38, TC-48, TC-53, TC-74, TC-78, TC-84, and TC-86 can drop against earlier runs:

- TC-18, TC-38, TC-48, and TC-74 require each recipient's address to come from a lookup that
  finished before the send or event. A correct address the model never looked up, or one whose
  lookup failed, now scores partial. TC-74 previously checked Sarah's lookup but not Mark's.
- TC-84 requires a fresh room search after `ROOM_TAKEN` before rebooking. Rebooking from the
  stale list scores partial.
- TC-78 scores partial when the reply states the right total and also asserts a different dollar
  figure, such as a "corrected" total.
- TC-86 requires the owner notice to name INC-442 and P1, in both the two-conflict scenario and
  its no-conflict variant. A generic notice scores partial.
- TC-53 requires a created event to keep the meeting's weekend date, not just a meeting-like
  title.
