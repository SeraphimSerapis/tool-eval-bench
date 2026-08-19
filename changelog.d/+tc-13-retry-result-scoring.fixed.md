**TC-13 retry-result scoring** — a successful retry that returns the Johnson
proposal is no longer erased by a later failed search. Recovery now requires
the target document in the retry's structured `results`; query echoes and
error messages that merely mention Johnson or `file_117` cannot earn PASS.
