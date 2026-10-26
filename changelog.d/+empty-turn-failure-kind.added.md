**Empty turns get their own failure kind.** A turn that ends normally with no visible text and no
tool call, which reasoning models produce when they think and then stop, now adds an `empty_turn=`
line to the trace and a note to the evaluation. A failing result gets the failure kind `empty_turn`
instead of a generic `missing_step` or `wrong_args`, so a batch of them points at the chat template
or reasoning parser rather than the model's judgement. Scores do not change. The idea comes from the
empty-turn diagnostics in QwenLM BabelArena.
