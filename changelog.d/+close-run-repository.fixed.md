The `history`, `diff`, and `compare` CLI paths and the programmatic `run_benchmark` entry point
constructed a `RunRepository` and left its SQLite connection to `__del__`. Early returns, and the
`sys.exit(1)` on a missing run, skipped the close entirely. In WAL mode that can strand `-wal` and
`-shm` files. All four sites now close deterministically.
