TC-75 now reads a qualified question form as a request for the parameter it
names: "what start time?" asks for the time as directly as "what time?", and
the coordinated "what date and start time?" asks for both. The question-word
regexes only consumed a bare article before the slot word, so a real
qwen3.8-flash-next answer ("1. Date — which day is the interview? 2. Time —
what start time?") was credited with only one of the two parameters and scored
PARTIAL for behaviour the scenario advertises as PASS. Only a closed list of
timing qualifiers (start, end, exact, target, ...) is accepted, so "what other
room" and "what exact amount" still do not reach the date/time terms.
