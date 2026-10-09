**A failing backend probe no longer stops a CLI run.** The CLI and the Python API now share one
backend detection implementation. If identifying the server raises an unexpected error, the CLI
logs a warning and records the backend as `unknown`, as the API already did, instead of exiting
with a traceback. The warning is a single plain-text line on stderr, including under `--json`.
Detection results are otherwise unchanged.
