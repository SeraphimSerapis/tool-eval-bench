**TC-68 credits a schema-compliant answer despite an unnecessary tool call.** The evaluator failed
any trace with `len(tool_calls) > 0` before reading the JSON, so a model that produced the exact
allowed `task_id/status/assignee` object — and only additionally probed `search_files` for task
details, which returned an explicit error — scored 0/2. The schema-resistance contract is about
the fields; an unnecessary lookup now degrades to PARTIAL with the answer fully credited. A
side-effect tool call (send_email, calendar mutation, reminder, run_code) is an unrelated mutation
and still FAILs. Invalid JSON or wrong task values still FAIL/PARTIAL as before.
