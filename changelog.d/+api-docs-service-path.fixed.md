The programmatic API docs pointed integrators at `tool_eval_bench.runner.service`, which is a
compatibility re-export. They now use `tool_eval_bench.application.service`, which owns
`BenchmarkService`. The version shown in the return-value tables was also stale in two places.
