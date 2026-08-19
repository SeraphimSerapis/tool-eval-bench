**TC-03 implicit-tool-need time phrasing (scoring)** — the email body must
still state that the meeting "moved" and name a time, but accepted spellings
now cover common 12-hour and 24-hour forms: `3pm`, `3 PM`, `3:00 PM`,
`3 p.m.`, and `15:00`/`1500`. Previously only the literal substrings `3pm`,
`3 pm`, and `15:00` passed, so a complete message like "the meeting has been
moved to 3:00 PM" scored PARTIAL instead of PASS. This is an additive
scoring change: the contact-lookup → email chain, the recipient, non-empty
subject/body, and the "moved" statement are all still required, and other
times such as 3:30 PM remain rejected.
