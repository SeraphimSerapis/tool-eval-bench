Requests already queued when a second 429 arrives now re-enter the rate-limit queue under the newer pause and pacing interval, rather than all resuming together when that pause ends.
