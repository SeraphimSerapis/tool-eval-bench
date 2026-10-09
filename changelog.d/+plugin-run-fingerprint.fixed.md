**Plugin run fingerprints include sampling parameters and the shuffle seed.** GSM8K, MMLU, IFEval,
and needle runs left the extra request parameters (`--top-p`, `--top-k`, `--min-p`,
`--repeat-penalty`, `--no-think`, `--backend-kwargs`) out of their stored config, so runs with
different sampling settings looked comparable. An unseeded `--gsm8k-shuffle` drew a fresh sample
each run without recording it. The config now carries `extra_params`, and an unseeded shuffle
records the seed it drew as `shuffle_seed`, which also makes that sample reproducible.
