`MarkdownReporter` was a 949-line class holding five report writers that shared nothing but an
output directory. Each writer now lives in its own module under `storage/reports/`, with the shared
label, path, and table helpers in `_common.py`. `MarkdownReporter` remains the public entry point
with an unchanged interface, and all five reports render byte-identically.
