**`--context-pressure-sweep` refuses held-out scenario packs.** The sweep report writes the full
trace of every scenario, so running a `--scenario-pack` through a sweep published the pack's
titles, prompts, and traces, and the sweep config recorded no pack attestation. Combining the two
flags is now a usage error, reported as `invalid_arguments` under `--json`, on the legacy flags
and on the `run` and `bench` commands alike. A single `--context-pressure` run with a pack is a
scored run and still withholds pack traces.
