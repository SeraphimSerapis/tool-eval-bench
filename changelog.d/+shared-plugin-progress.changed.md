GSM8K, MMLU, and IFEval each carried their own copy of the same Rich progress layout and
correct/wrong/error accounting. That now lives in `cli/plugin_progress.py`, which the three runners
share. Rendered output is unchanged, verified by diffing all three runners' console output before
and after.
