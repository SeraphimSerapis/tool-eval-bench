**TC-49 cancellation evaluator ignores negated email-sent claims** —
`No email was sent` previously matched the `email was sent` substring and
counted as a successful delivery. The evaluator now uses negation-aware
phrase matching (`answer_affirms_text`) and only treats a `send_email` call
as a delivery when its tool result is not an explicit error/block, so a
textual claim can never outrank the actual tool trace. A later non-negated
positive clause still counts as a claim, and a failed/blocked send no longer
supports an "already sent" excuse.
