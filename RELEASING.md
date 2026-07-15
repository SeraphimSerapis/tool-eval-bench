# Releasing tool-eval-bench

Checklist for publishing a new release.

## Pre-release

1. **Update version strings** (all three MUST match):
   ```
   pyproject.toml        → version = "X.Y.Z"
   src/tool_eval_bench/__init__.py → __version__ = "X.Y.Z"
   CHANGELOG.md          → ## [X.Y.Z] — YYYY-MM-DD
   ```

2. **Lint, format, and run the required randomized suite**:
   ```bash
   ruff check .
   ruff format --check .
   for seed in 104729 130363 155921; do
     .venv/bin/python -m pytest tests/ \
       --ignore=tests/test_llama_benchy.py -m "not live" \
       --randomly-seed="$seed"
   done
   ```

3. **Run coverage and the optional performance integration**:
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

4. **Build and smoke-test the installed wheel in isolation**:
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

## Post-release

- Add a new `## [Unreleased]` section at the top of `CHANGELOG.md`

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
