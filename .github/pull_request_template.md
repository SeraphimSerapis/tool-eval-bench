## Summary

<!-- What problem does this change solve? Link the relevant issue, if one exists. -->

## Changes

<!-- Summarize the implementation and any user-facing or scoring impact. -->

## Validation

<!-- List focused tests and the broader checks you ran. If a check is not applicable, say why. -->

- [ ] Focused tests pass.
- [ ] `.venv/bin/ruff check .` passes.
- [ ] `.venv/bin/ruff format --check .` passes.
- [ ] `.venv/bin/mypy` passes.
- [ ] Required pytest suite passes.
- [ ] Live-server testing is not required, or the test endpoint and scope are described.

## Contributor checklist

- [ ] This PR contains one focused, logically related change.
- [ ] Regression or behavior tests were added or updated where appropriate.
- [ ] Positive and negative/false-positive cases are covered for evaluator changes.
- [ ] `README.md` and/or `CHANGELOG.md` is updated when applicable.
- [ ] No credentials, live endpoints, generated run artifacts, or unrelated formatting changes are included.
