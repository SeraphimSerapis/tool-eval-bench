**PyPI publishing workflow.** `.github/workflows/publish-pypi.yml` uploads a tagged release to PyPI or
TestPyPI through Trusted Publishing, with no API token. It stays inactive until the maintainer
completes the one-time setup in `RELEASING.md`. After that it runs when dispatched by hand for an
existing tag, or when a release is published once enabled with the `PYPI_PUBLISH_ON_RELEASE`
repository variable. It never runs on push. Package metadata now links to the changelog.
