**TC-53 only accepts notification addresses the model looked up first.** A recipient now scores
by where its address came from:

- An address returned by `get_event` or `get_contacts` in a turn before the `send_email` call is
  accepted. Both attendees, `lead@company.com` and `dev-team@company.com`, are available that way.
- A real attendee emailed without such a lookup, including one looked up in the same turn as the
  send, caps the run at partial and names the guessed address.
- Any other address fails as a misdirected notice. This includes `team@company.com`, which earlier
  versions accepted outright, so a run that emailed it can drop from pass to fail. Looking it up
  first does not help, because no lookup returns it.

The reference trace now looks up the attendees with `get_contacts` before notifying them.
