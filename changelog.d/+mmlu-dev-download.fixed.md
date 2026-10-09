**A failed MMLU dev-split download is reported cleanly.** The few-shot dev split downloaded outside
the shared loader, so a rate-limited download ended the run with a traceback. It now uses the same
loader as the test split, with download progress, a clear error, and the hint that a re-run
resumes from `data/mmlu/dev.partial.jsonl`.
