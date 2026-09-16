The orchestrator stops a scenario after three consecutive turns of identical
tool calls with identical results and records `repeated_call_loop` as the
failure kind, with the reason in the evaluation note. Gemini 3.8 Flash spent
its whole `TC-65` budget re-issuing the same `get_weather` call; that now ends
at turn three instead of eight, and the report distinguishes a stuck model
from one that ran out of turns. Polling that keeps calling until the result
changes is unaffected.
