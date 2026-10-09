**Context pressure refuses a window too small to hold any filler.** 16,096 tokens of every window
are reserved for output and the scenario. On a smaller window, such as llama-server's default
4,096-token context or an 8K slot from `-c 32768 --parallel 4`, a `--context-pressure-sweep` ran
every level with no filler and saved a 100% breaking point, and `--context-pressure` stored the
requested ratio for an unpressured run. A sweep now fails before its first level when the top of
its range cannot hold one 2,048-token filler chunk, and a single `--context-pressure` run fails
when its ratio and window give no filler at all. Both name the window and point at
`--context-size`.
