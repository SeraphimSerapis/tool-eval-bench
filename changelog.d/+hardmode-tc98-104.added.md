**Seven new Hard Mode scenarios (TC-98 to TC-104).** `--hardmode` now runs 35 Category P
scenarios, 104 in total. The additions cover a prompt injection planted two hops away in a
runbook comment (TC-98), saga rollback where the compensation order has to be read from tool
results (TC-99), a partially failed non-idempotent batch that must not be resubmitted whole
(TC-100), conflicting sources where a signed policy outranks a newer chat message (TC-101), a
sunset v1 endpoint whose migration contract leads to v2 (TC-102), reconciling a pre-filled shared
cart under stock, shipping, coupon, and budget rules (TC-103), and a fix only the user can make on
their own device, which the agent has to hand over and then verify (TC-104). TC-103 and TC-104
adapt task patterns from QwenLM BabelArena's DeepPlanning and tau-cubed families. Hard Mode
totals and the capability breakdown change for any run that includes `--hardmode`.
