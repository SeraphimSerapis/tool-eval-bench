Needle-in-a-haystack retrieval benchmark behind `--needle` / `--needle-only`,
which compose with the other top-level flags the way `--perf` does
(`tool-eval-bench --hardmode --seed 42 --perf --needle`), or via
`tool-eval-bench plugin needle`. It buries a synthetic fact at a known depth in a
generated haystack and sweeps a grid of context lengths and depths, reporting
retrieval accuracy and the largest haystack retrieved at every depth. Grid shape
is set by `--needle-lengths` and `--needle-depths`. See `docs/needle.md`.
