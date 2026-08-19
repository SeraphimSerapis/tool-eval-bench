**Pre-flight configuration parity (Issue #51)** — the model availability
check now uses the benchmark's configured request timeout and merged backend
parameters, preventing provider-specific options such as `reasoning_effort`
from causing false negatives. The check can be explicitly bypassed with
`--no-preflight` when an endpoint needs custom startup handling; it remains
enabled by default, and timeout failures now include a useful exception type.
