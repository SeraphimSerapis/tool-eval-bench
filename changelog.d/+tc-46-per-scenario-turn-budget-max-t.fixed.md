**TC-46 per-scenario turn budget (`max_turns_override`)** — the deep
multi-turn research workflow needs up to 11 assistant exchanges for its
canonical reference path (5 user turns plus tool-call rounds and final
answers), which exceeds the global `max_turns=8` default and cuts the run
off before the final email. `ScenarioDefinition` gains an optional
`max_turns_override` field; TC-46 sets it to 12, giving the reference path
finite headroom without raising the global default for every scenario.
The orchestrator now also flags turn-budget exhaustion distinctly
(`turn_budget_exceeded` plus `failure_kind="budget_exceeded"` when the run
stops before a final answer / before follow-ups are drained), so a budget
run-out is no longer indistinguishable from an evaluator verdict.
