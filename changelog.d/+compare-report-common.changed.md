The two comparison report generators each defined the same eight formatting helpers, byte for byte.
They now live in `compare_reports/_common.py`, so the two reports cannot drift apart on how a
percentage or a delta is rendered. `short_label` stays per-generator, since the two genuinely
shorten model names differently. Generated HTML is unchanged.
