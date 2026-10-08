**The CLI records thinking as it was sent.** History and reports read `thinking_enabled` from
`--no-think`, so disabling thinking through `--backend-kwargs '{"chat_template_kwargs":
{"enable_thinking": false}}'` was recorded as enabled, and `--no-think` overridden by
`"enable_thinking": true` in `--backend-kwargs` was recorded as disabled. The CLI now reads the
request payload, the way the Python API already did. Only the recorded flag changes: it is not part
of the comparison fingerprint, so no run changes cohort, and runs already stored keep their value.
