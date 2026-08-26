The Prometheus label parser behind the live speculative-decoding monitor no
longer backtracks exponentially. In `(?:\\.|[^"])*` the negated class also
matched a backslash, so every escape had two possible parses and a label that
opened a quote without closing it took time doubling with each repetition:
roughly half a second at 22 escapes, and unbounded past that. Metrics text
arrives from whatever server the run points at, so the input is reachable.
Excluding the backslash from the negated class leaves one parse and identical
results on well-formed input.
