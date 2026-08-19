# Releasing tool-eval-bench

Checklist for publishing a new release.

## Pre-release

1. **Do not edit any version string.** The version comes from the git tag via
   setuptools-scm. `pyproject.toml` declares `dynamic = ["version"]` and
   `src/tool_eval_bench/__init__.py` resolves it from the generated
   `_version.py`. Tagging in the Tagging section below is what sets the version.

2. **Build the changelog from its fragments**:
   ```bash
   .venv/bin/towncrier build --draft --version X.Y.Z   # preview, writes nothing
   .venv/bin/towncrier build --version X.Y.Z           # writes and deletes fragments
   ```

   This replaces the old step of hand-editing `CHANGELOG.md`. `towncrier build`
   collects every file in `changelog.d/`, inserts a `## [X.Y.Z] — YYYY-MM-DD`
   section, and removes the fragments. Read the draft before writing: it is the
   last point where an unclear entry is cheap to fix.

   An empty `changelog.d/` means nothing user-visible changed since the last
   release, which is a reason to question whether the release is needed.

3. **Lint, format, and run the required randomized suite**:
   ```bash
   ruff check .
   ruff format --check .
   for seed in 104729 130363 155921; do
     .venv/bin/python -m pytest tests/ \
       --ignore=tests/test_llama_benchy.py -m "not live" \
       --randomly-seed="$seed"
   done
   ```

4. **Run coverage and the optional performance integration**:
   ```bash
   .venv/bin/python -m pytest tests/ \
     --ignore=tests/test_llama_benchy.py -m "not live" \
     --randomly-seed=104729 --cov=tool_eval_bench \
     --cov-report=term-missing --cov-fail-under=80

   .venv/bin/python -m pip install -e '.[dev,perf]'
   .venv/bin/python -m pytest tests/test_llama_benchy.py \
     --randomly-seed=181081
   ```

   The release gate is 80% branch coverage. Record any notable module-level
   gaps in the release notes even when the aggregate gate passes.

5. **Build and smoke-test the installed wheel in isolation**:
   ```bash
   rm -rf dist
   .venv/bin/python -m pip install build
   .venv/bin/python -m build --wheel
   .venv/bin/python -m venv /tmp/tool-eval-wheel-smoke
   /tmp/tool-eval-wheel-smoke/bin/python -m pip install dist/*.whl
   /tmp/tool-eval-wheel-smoke/bin/python -m pip check
   /tmp/tool-eval-wheel-smoke/bin/tool-eval-bench --version
   /tmp/tool-eval-wheel-smoke/bin/tool-eval-bench --help
   /tmp/tool-eval-wheel-smoke/bin/tool-eval-bench run --help
   /tmp/tool-eval-wheel-smoke/bin/tool-eval-bench plugin --help
   ```

   Also verify that `tool_eval_bench.evals.yaml_scenarios/weather.yaml` is
   available through `importlib.resources` in the clean environment.
   The isolated PEP 517 build must honor the `setuptools>=77` minimum required
   for the project's SPDX license expression.

## Tagging

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

The tag is what sets the version, so tag before building any artifact you intend
to publish. A wheel built before the tag reports the previous release plus a dev
suffix.

## Release notes

The notes are the changelog section `towncrier build` just wrote. Extract it
rather than retyping it, so the GitHub release and the changelog cannot drift:

```bash
scripts/release_notes.py X.Y.Z > /tmp/notes.md
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file /tmp/notes.md
```

Add anything that belongs in a release announcement but not in a changelog
(upgrade instructions, a known-issues note, coverage gaps recorded in step 4) by
editing `/tmp/notes.md` before creating the release. Leave `CHANGELOG.md` as the
generated record.

## Post-release

- `changelog.d/` is already empty; `towncrier build` deleted the fragments it
  consumed. There is no `## [Unreleased]` section to recreate.

## Live Certification (required before major releases)

Run the full benchmark against at least one backend to verify deployment
compatibility:

```bash
# vLLM
tool-eval-bench --backend vllm --base-url http://localhost:8000

# llama.cpp
tool-eval-bench --backend llamacpp --base-url http://localhost:8080

# LiteLLM
tool-eval-bench --backend litellm --base-url http://localhost:4000
```
