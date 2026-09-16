A turn that ends on `finish_reason=length` with no visible answer and no tool
call stops the scenario with failure kind `reasoning_truncated`, a
`truncated=` trace line, and an evaluation note stating the ceiling and how
much reasoning was cut. The result still scores, since the model did not
answer, but the tag separates "ran out of room to think" from a wrong answer.
Both adapters now record the provider's finish reason; Gemini's
`MAX_TOKENS` maps to `length`.
