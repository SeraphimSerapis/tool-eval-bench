Timed-out scenarios no longer render as `FAIL 0/2`. They show as `⏱ TIMEOUT` with
`–/2` points and a reason, because an infrastructure failure leaves the scenario
out of both the numerator and the denominator rather than scoring it zero. When a
run has timeouts, it now also prints what to change, using the slowest turn it
measured and the timeout that was in force.

Turns after the first are given a timeout scaled from turn 1's measured latency.
Only turn 1 is streamed, so on later turns the read timeout bounds the whole
generation instead of the gap between tokens, and a slow reasoning model could
blow it on turn 2 without having slowed down. A hung endpoint never completes
turn 1, so it still fails at the configured timeout.
