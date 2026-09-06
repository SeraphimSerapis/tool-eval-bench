Stopped scoring three serving-stack properties as model quality. A 4xx that
rejects the request before the model produces anything is now an infrastructure
failure that leaves the score's numerator and denominator, instead of having its
error string graded as the model's answer. TC-45 is excluded on an endpoint that
does not enforce `tool_choice="required"`, detected by one probe per run, because
a dropped parameter is otherwise indistinguishable from a model ignoring an
instruction it never received. TC-88 now says when an endpoint exposed no
reasoning channel, rather than reporting an unreachable PASS as a model failure.
