# Hard Mode

Nineteen opt-in scenarios (TC-70 to TC-88) built to separate models that already score well on
the standard suite. They are adversarial, stateful, and transactional, and they are not included
in the default 69.



The standard 69-scenario benchmark covers *breadth* of tool-calling capabilities. Once a model scores 100% on the standard suite, `--hardmode` adds ceiling-breaking scenarios (Category P) designed to separate truly excellent models from merely good ones.

```bash
# Standard benchmark + Hard Mode scenarios (69 + 19 = 88 scenarios)
tool-eval-bench run --hardmode

# Run only Hard Mode scenarios
tool-eval-bench run --hardmode-only
tool-eval-bench run --hardmode --categories P  # equivalent

# Combined with context pressure for maximum difficulty
tool-eval-bench run --hardmode --context-pressure 0.75
```

Hard Mode focuses on nineteen ceiling-breaking scenarios:

| Scenario | Focus Area | What it tests |
|---|---|---|
| TC-70 | Adversarial tool definitions | Near-duplicate tools with subtle scope differences (Europe-only vs global) |
| TC-71 | Ambiguous recipient | Find multiple matching contacts and ask for clarification, not guess |
| TC-72 | Cascading error recovery | Primary summary read fails → use its backup → complete the email chain |
| TC-73 | Multi-constraint composition | Search + filter by 3 simultaneous constraints + contact lookup + email |
| TC-74 | Stateful multi-turn corrections | Preserve 4 rounds of draft changes, then create and notify exactly once |
| TC-75 | Missing required parameter | Pass for requesting date and time; partial for requesting only one; fail for guessing or premature tool use |
| TC-76 | Missing capability | Refuse unsupported mutations; a relevant read-only check remains fully valid |
| TC-77 | Irrelevant tool trap | Answer a stable fact directly despite distractor tools |
| TC-78 | Independent portfolio valuation | Fetch three stock prices and aggregate the result |
| TC-79 | Dependency-aware event planning | Resolve independent inputs before conditional event creation |
| TC-80 | Preconditioned update safety | Check an exact requested date before mutation and preserve original state |
| TC-81 | Tool-output prompt injection | Ignore malicious instructions embedded in search results |
| TC-82 | Stale memory conflict | Prefer authoritative current directory state over stale memory |
| TC-83 | Format-sensitive chaining | Return exact JSON after noisy lookups while allowing independent work in parallel |
| TC-84 | Long-horizon recovery with constraint retention | Recover from a booking race and complete equivalent valid notification workflows |
| TC-85 | Exactly-once provisioning after ambiguous commit | Wait through pending replication, confirm one least-privilege credential, avoid duplicates, and never disclose the secret |
| TC-86 | Optimistic concurrency without lost updates | Re-read after two consecutive conflicts and preserve both concurrent field changes |
| TC-87 | Complete pagination with cursor integrity | Follow four cursors, reject a stale-count shortcut, deduplicate, resolve routing, and delay notification |
| TC-88 | Preserved reasoning across follow-ups | Carry three linked, privately planned constrained values across two user follow-ups |

TC-88 opts into replaying the provider's `reasoning_content` field across its
follow-up turns. It does not ask the model to print its reasoning to the user.
A pass requires the first reasoning payload to contain all three exact values;
valid outputs from a backend that keeps reasoning opaque receive partial credit.
Other scenarios keep the default behavior and do not replay completed no-tool
reasoning across user turns.

Hard Mode scenarios use the same scoring (pass=2, partial=1, fail=0) and appear
under Category P in reports. They are absent from the default 69-scenario run;
when selected, they contribute to that run's score. This keeps default results
comparable across standard-suite runs.

Multi-turn authorization scenarios record the active user-message phase for
each tool call. This prevents a correct-looking mutation made before the
authorizing follow-up from receiving credit.

TC-78 and TC-79 record same-turn parallel tool calls as informational telemetry. Sequential calls receive full correctness credit so backends without parallel tool-call support, including llama.cpp, remain first-class targets.
