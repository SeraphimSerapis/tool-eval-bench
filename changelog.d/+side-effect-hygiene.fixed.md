**Unrequested side effects no longer pass.** Many evaluators found the one correct call and ignored
everything around it. On 32 of 88 scenarios, a run could send a second email, create an unrelated
calendar event, set a reminder, or run code and still pass. Each of those evaluators now declares
the writes its task allows through `forbid_unrequested_side_effects`, and any other write turns a
pass or partial into a fail. A retry of an allowed write after a tool error is not counted as a
duplicate. Scenarios where computing the answer is the task (TC-15, TC-20, TC-35, TC-52, TC-61)
still allow `run_code`. TC-80 no longer passes a run that calls `restore_event` on a booking that
never changed. Scores on the affected scenarios can drop for models that take extra actions.
