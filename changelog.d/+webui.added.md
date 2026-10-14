**Optional local web UI (`tool-eval-bench serve`).** Install `tool-eval-bench[web]` and run
`tool-eval-bench serve` to open a browser workbench over the run history. It shows stored runs
with their multi-turn traces, cohort leaderboards with CSV and JSON export, and two-run
comparisons with McNemar's test. It also launches runs, streams them live over Server-Sent
Events (CLI runs too, through their SQLite checkpoints), and stops or resumes them. The server
prints a sign-in link with a per-process token and refuses requests without it. State-changing
requests must come from the UI's own origin as JSON, so other websites cannot use the server or
the API key it holds. The UI stops only runs it launched and resumes only interrupted runs whose
stored conditions it can rebuild. The launcher rejects malformed numeric settings, such as a
negative temperature or a fractional turn limit, with an error that names the field.
