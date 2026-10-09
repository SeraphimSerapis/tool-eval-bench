**Hard Mode evaluator corrections.** Scores for TC-71 to TC-91 can move against earlier runs, in both
directions:

- TC-73 no longer passes a recommendation of Mitte Brasserie because an exclusion word appears
  somewhere else in the transcript. The reason it is out has to be about Mitte Brasserie, and an
  email that recommends it fails the confirmation step. "Shuts on Sundays" and "doesn't open on
  Sundays" now count as exclusions.
- TC-72 rejects a budget email with an extra `cc` or `bcc` recipient. TC-71's clarified variant,
  TC-72, TC-73, TC-82, TC-85, TC-86, and TC-87 accept a recipient sent as a one-element JSON array
  or with different letter case, as the rest of the suite already does.
- TC-74 no longer flags a confirmation email as unsafe because "not" appears in an earlier
  sentence.
- TC-80 no longer reads a closing offer such as "Let me know if you'd like it moved" as a claim
  that the event moved.
- TC-89 no longer records a payment claim for plan narration such as "I'll email AP once paid".
- TC-84 accepts an agenda lookup made after booking, as long as it comes before the email.
- TC-75's clarified variant accepts a corrected room search after a rejected one. TC-79 accepts
  `09:00:00` for the event time.
- TC-91 no longer credits the bare word "access" (as in "access request") as explaining the
  missing permission.
- A tool call the mock rejected no longer counts as a repeat: a corrected `issue_payment`,
  `release_reservation`, `request_limit_increase`, or `file_dispute` in TC-89 and TC-90, a rejected
  opening `list_incidents` call in TC-87, a TC-86 no-conflict update rejected only for sending
  `expected_version` as a string, and a TC-91 `request_access` retried after an errored attempt.
  Rejected attempts that changed filters mid-stream or would have overwritten fields still count.
- The shared clarification check recognises "without knowing which Jordan you mean".
