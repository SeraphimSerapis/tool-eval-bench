A single test spent 15.5 seconds of the suite's 23.6 asleep. It zeroed the post-429 retry delay but
not the rate-limit coordinator's adaptive spacing, which widens on every 429 and is enforced by a
real sleep. The full suite now runs in 8.4 seconds.
