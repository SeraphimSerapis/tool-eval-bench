TC-63 (Accumulating Constraints) no longer scores an answer that kept all four
constraints below one that kept a single constraint. Both PASS branches require
a qualifying `web_search` call, and nothing handled 4/4 without one, so such an
answer fell past every count branch to the closing failure. It scored 0 points
under a summary reading "Final answer doesn't reflect any of the accumulated
constraints", while a 1/4 answer scored 1. It now scores PARTIAL, and the
summary says what the model actually did: it satisfied all four constraints but
never searched for a match.
