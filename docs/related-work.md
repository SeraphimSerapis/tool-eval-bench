# Related work

How tool-eval-bench sits against the benchmarks it borrows from and competes
with. For the feature-by-feature matrix, see
[methodology.md](methodology.md#comparison-with-other-benchmarks).

| Benchmark | Focus | How tool-eval-bench differs |
|---|---|---|
| [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) | Berkeley Function Calling Leaderboard — large-scale function-calling eval (1,700+ tests) | We focus on *agentic* multi-turn orchestration, not single-turn completion. Our 69 scenarios emphasize chained reasoning, error recovery, and safety boundaries. |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | API discovery across 16K+ real-world APIs | We use deterministic mock tools with realistic payload noise for reproducible scoring. No external API dependencies. |
| [NexusRaven](https://nexusflow.ai/blogs/ravenv2) | Function-calling via fine-tuned models | We're model-agnostic — any OpenAI-compatible endpoint works. We also measure throughput (pp/tg) alongside correctness. |
| [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | Multi-turn API usage (73 APIs) | We add safety/boundary testing (Category K with 13 scenarios including prompt injection resistance), large-toolset scale testing (52 tools), and statistical rigor via `--trials`. |
| [ToolCall-15](https://github.com/stevibe/ToolCall-15) | 15-scenario quick assessment | Our direct ancestor. We extended it to 69 standard scenarios across categories A–O, plus 19 opt-in Hard Mode scenarios in Category P, and added multi-turn orchestration, autonomous planning, creative composition, structured output evaluation, throughput benchmarking, and production-grade persistence. |
| [PinchBench (OpenClaw)](https://github.com/open-claw/PinchBench) | Agentic task completion in real environments | PinchBench tests end-to-end task completion. We focus on the tool-calling substrate: does the model pick the right tool, pass the right params, and chain correctly? Complementary benchmarks. |

**Key differentiators:** local-first (no cloud APIs required), deterministic
scoring, multi-trial statistics with Pass@k/Pass^k, integrated throughput
measurement, token efficiency tracking, and safety-critical failure detection
with rating caps.

## Credits

Scenario methodology adapted from
[ToolCall-15](https://github.com/stevibe/ToolCall-15) by
[stevibe](https://x.com/stevibe) (MIT License).
