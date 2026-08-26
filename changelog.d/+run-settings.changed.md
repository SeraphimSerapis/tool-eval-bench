The benchmark service built its persisted run config by passing the same seventeen arguments twice,
once before the run and once after merging resumed results. Those parameters are now a frozen
`RunSettings` value captured once, and the config builder is public as
`tool_eval_bench.application.run_config.build_run_config`. `BenchmarkService.run_benchmark` keeps
its existing keyword signature, and config fingerprints are unchanged.
