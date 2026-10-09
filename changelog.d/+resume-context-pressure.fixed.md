**Resume checks context pressure.** `resume` compared the model, endpoint, sampling, and scenario
settings but not context pressure. Resuming a `--context-pressure 0.5` run with `0.75`, or without
the flag, merged scenarios measured at two fill levels into one result whenever `--timeout` was
large enough that the pressure timeout scaling left it alone. With a smaller timeout the scaled
value happened to differ, and resume refused while naming only `timeout_seconds`. Resume now
refuses a different ratio, an added or dropped `--context-pressure`, and a context size that
changes the fill target, and names the difference, for example `context_pressure ratio (was 0.5,
now 0.75)`. A detected context size that changes without moving the fill target is still accepted,
since a restarted server can report a slightly different KV capacity. Runs without pressure resume
as before.
