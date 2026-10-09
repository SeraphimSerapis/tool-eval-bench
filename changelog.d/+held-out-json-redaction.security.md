**`--json` output no longer publishes held-out pack scenarios.** The Markdown report withheld a
pack scenario's title, summary, and trace, but the `--json` envelope, `--json-file`, and the stderr
`scenario_start` and `safety_gate_failed` events still carried them, so uploading a CI artifact
burned the pack. JSON output now keeps each pack scenario's ID, status, points, failure kind,
timings, and token counts, marks it `"held_out": true`, and replaces its summary, trace, expected
behaviour, and called tools with `held out`. Safety warnings for pack scenarios keep their ID and
read `HOLD-01: held out`. Result fields are kept by allowlist, so a field added later is withheld
until it is listed. Pass `--include-held-out` to keep the full content. Public scenarios and the
SQLite record are unchanged.
