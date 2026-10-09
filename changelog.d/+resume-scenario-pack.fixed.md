**Resume refuses to add a held-out pack.** A run without `--scenario-pack` stores no
`scenario_packs` key, and `resume` read that missing key as an older run that never recorded the
setting, so it skipped the check. Resuming such a run with `--scenario-pack` merged held-out
results into the public run. Runs stored with a scenario list were still refused, but only under
`scenario_ids`; older runs without one were not refused at all. Resume now reads a missing key as
"no packs", refuses an added pack, and names `scenario_packs`. Runs without packs resume as
before.
