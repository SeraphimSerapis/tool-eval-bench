The `llama-benchy` coverage gate fails the build again. Passing
`--cov-config=/dev/null` alongside a dotted `--cov` target made pytest-cov 7.1.0
print `FAIL Required test coverage of 95% not reached` and still exit 0, so the
threshold had stopped gating anything. Coverage of `runner/llama_benchy.py` had
already slipped to 94.97% behind it. The step now uses a real config file
(`.coveragerc.perf`) and also runs `test_llama_benchy_redaction.py`, which covers
the URL-redaction helpers, restoring it to 96.65%.
