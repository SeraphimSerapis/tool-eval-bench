The pre-push hook runs the test suite in parallel with `pytest-xdist`, cutting it from about 15 seconds to about 7. `pytest-xdist` is now a dev dependency.
