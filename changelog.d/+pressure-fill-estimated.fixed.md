**Estimated context-pressure fills are labelled as estimates.** Filler is sized at about 4
characters per token and calibrated through the server's `/tokenize`. On a server without a
compatible `/tokenize` the stored `fill_tokens` was that estimate, typically 10 to 17% above the
real count, presented as a measurement. A scored run now stores `fill_tokens_estimated: true` in
its `context_pressure` config in that case, a sweep stores it on each affected level, and both
reports label the fill as estimated.
