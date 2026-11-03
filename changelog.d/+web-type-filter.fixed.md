**Web UI type filter.** "Spec decode" and "Context pressure" in the run list's type filter matched
no runs, because they asked for `spec_decode` and `context_pressure` while runs are stored as
`spec-bench` and `context-pressure`. Both now match, and Toolscale is listed.
