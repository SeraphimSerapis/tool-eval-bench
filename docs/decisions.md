# Benchmark scoring decisions

## Visible correctness and safety outcomes

TC-88 scores observable constraints independently of exposed reasoning. Transport
observations live in diagnostics because reasoning visibility is endpoint-dependent.
Revisit this split only if a backend-independent observable planning contract replaces it.

Safety warnings come from explicit unsafe outcomes, including recovered intermediate
mutations. Category membership alone cannot distinguish an unsafe action from harmless
incompletion. The existing rating threshold remains, gated on an observed violation.
Historical results need fresh execution to establish the new safety outcome field.
