Extracted the scenario-selection validation, the pre-flight and warm-up gate, and the run-context
collection out of the CLI's 760-line `main()` into named helpers. Behaviour is unchanged, verified
by diffing the output and exit code of 22 CLI invocations.
