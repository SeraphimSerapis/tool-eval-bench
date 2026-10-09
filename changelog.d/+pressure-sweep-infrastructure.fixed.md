**Context-pressure sweeps no longer score infrastructure failures as model failures.** A timeout,
connection error, or 5xx response at a sweep level counted as a failed scenario, and so did TC-45
on an endpoint that does not enforce `tool_choice='required'`. On such an endpoint every default
sweep reported no breaking point and flagged degradation at the first level, and two levels of
prefill timeouts stopped the sweep as if the model had collapsed. These results, and a level that
fails as a whole, are now left out of the level's pass rate, the breaking point, the first
degradation, and the all-fail early stop, as scored runs already leave them out of the quality
score. Each level stores an `excluded_count` and the `excluded_scenarios` IDs, and the report marks
each excluded scenario. A level where nothing was scored stores a `score_pct` of null, and two
such levels in a row stop the sweep. A sweep where no level was scored reports its breaking point
as n/a rather than none, and a sweep that stops early stores the reason as `stop_reason`. Breaking points can move up
compared with earlier sweeps. The needle benchmark still counts a request timeout as a miss.
