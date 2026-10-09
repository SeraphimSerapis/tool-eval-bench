**Answers cut off by the token budget are counted as truncated.** When a response had no content,
GSM8K, MMLU, and IFEval graded the reasoning text instead, even when generation stopped because it
ran out of tokens mid-thought. A stray number or letter in that unfinished reasoning could score
as correct. An empty response with `finish_reason == "length"` is now wrong and flagged
`truncated`, and the run reports a truncated count in its details, console summary, and report.
The reasoning fallback still applies when generation finished normally.
