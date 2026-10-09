**Pressure fingerprints follow the fill the model saw.** A context-pressure sweep's stored config
now records the effective context size, after `--context-size` and the KV-capacity cap, and the
seed. Both set every level's filler, yet two sweeps with fills of 16K and 244K tokens shared one
fingerprint. A scored `--context-pressure` run no longer fingerprints the calibrated
`fill_tokens`, which differs on every unseeded run and kept otherwise identical pressure runs out
of one comparison group; the stored config still records it. Leaderboard cohorts ignore it the
same way, so two models measured at the same pressure target rank together. Sweeps already regroup in this
release because of the mode fingerprint change (#264), so this adds no further break for them.
Scored pressure runs stored by earlier versions do not group with new ones.
