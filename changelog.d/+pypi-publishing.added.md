**PyPI publishing workflow.** `.github/workflows/publish-pypi.yml` uploads a tagged release to PyPI or
TestPyPI through Trusted Publishing, with no API token. It stays inactive until the maintainer
completes the one-time setup in `RELEASING.md`, then runs only when a release is published or the
workflow is dispatched for an existing tag, never on push. Package metadata now links to the changelog.
