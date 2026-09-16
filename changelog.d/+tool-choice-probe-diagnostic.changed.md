The `tool_choice="required"` probe now tells the model not to call any tool,
so a tool call in the reply proves the endpoint enforced the constraint rather
than that the model was willing. It also checks that the forced call kept its
required argument, and every forced-call scenario carries the probe's verdict
under "Capability diagnostics", so an empty `calculator {}` can be attributed
to the tool parser or to the model. `TC-45` names the empty-argument case in
its summary instead of reporting an expression that "didn't evaluate to 56".
