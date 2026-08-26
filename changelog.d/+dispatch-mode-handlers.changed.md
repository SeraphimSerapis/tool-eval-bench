The throughput, speculative-decoding, and context-pressure-sweep branches of the CLI's `main()` are
now named handlers taking a single resolved-endpoint value instead of a dozen locals. `main()` is
down from 760 lines to 577. Behaviour is unchanged, verified against the output and exit code of 22
CLI invocations and the committed compatibility snapshots.
