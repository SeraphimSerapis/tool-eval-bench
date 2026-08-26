Scenario checkpoints were written to SQLite synchronously from inside an async callback, so every
commit stalled the event loop and every request in flight with it. Invisible at `--parallel 1`, and
costly above it. Checkpoint writes now run on a dedicated serialised writer thread.
