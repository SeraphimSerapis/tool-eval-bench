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

Pushing the tag starts `.github/workflows/release.yml`, which does steps 5's
build and smoke checks again on a clean runner, verifies the built version
matches the tag, and opens a **draft** GitHub release with the changelog
section `towncrier build` just wrote.

Read the draft. Add anything that belongs in a release announcement but not in
a changelog (upgrade instructions, a known-issues note, coverage gaps recorded
in step 4), then publish it. Nothing is public until you do. Leave
`CHANGELOG.md` as the generated record.

Publishing the release also starts the PyPI upload described in the next
section. Until its one-time setup is done, that run fails at the upload step and
publishes nothing.

To produce the notes locally, or if the workflow is unavailable:

```bash
scripts/release_notes.py X.Y.Z > /tmp/notes.md
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file /tmp/notes.md
```

## Publishing to PyPI

`tool-eval-bench` is not on PyPI yet. `.github/workflows/publish-pypi.yml` is
ready to upload it, but **nothing reaches PyPI until the one-time setup below is
done and a person triggers the workflow.** It never runs on push.

The workflow uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): PyPI accepts an
OIDC token from this repository's workflow instead of an API token, so there is
no secret to store or rotate. It builds the sdist and wheel once from the tag,
checks the built version matches the tag, runs `twine check --strict`, smoke
tests the wheel, and hands those exact files to the publish job. The publish job
runs in a GitHub environment, so a required reviewer has to approve it.

### One-time setup

1. **Create the GitHub environments.** Under Settings, Environments, add
   `pypi` and `testpypi`. On `pypi`, enable **Required reviewers** and add
   yourself. Under **Deployment branches and tags**, choose selected branches
   and tags and allow the `main` branch and `v*` tags: a release event runs on
   the tag, a manual dispatch runs on the branch it was started from.
2. **Add a pending publisher on PyPI.** At
   <https://pypi.org/manage/account/publishing/>, add a GitHub publisher with:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `tool-eval-bench` |
   | Owner | `SeraphimSerapis` |
   | Repository name | `tool-eval-bench` |
   | Workflow name | `publish-pypi.yml` |
   | Environment name | `pypi` |

   A pending publisher does not reserve the name. The first successful upload
   creates the project, and the publisher becomes a normal one.
3. **Optionally, do the same on TestPyPI** at
   <https://test.pypi.org/manage/account/publishing/> with environment name
   `testpypi`. TestPyPI is a separate account and a separate index.

The workflow file name and environment names are part of the trust
configuration. Renaming either breaks publishing until PyPI is updated to match.

Do step 1 before step 2. If a job references an environment that does not
exist, GitHub creates it with no protection rules, so a publisher registered
first would let the next release upload without an approval.

### Releasing to PyPI

1. Tag and let `release.yml` open the draft, as above.
2. Optional dry run: run **Publish to PyPI** from the Actions tab with the tag
   (for example `v2.8.0`) and target `testpypi`, then check
   <https://test.pypi.org/project/tool-eval-bench/>. An index never accepts the
   same version twice, so a TestPyPI upload cannot be retried for that tag.
3. Publish the GitHub release. That starts **Publish to PyPI**, which builds,
   checks, and then waits for approval on the `pypi` environment. Approve it.

If the release was published before the setup existed, dispatch the workflow
with the tag and target `pypi` instead. PyPI files cannot be replaced: a broken
upload needs a new version, or a yank.

Before the first upload, note that the PyPI page renders `README.md` as is.
Relative links and the screenshot path (`docs/...`) resolve against PyPI, not
GitHub, so they break there unless the README uses absolute URLs.

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
