The adaptive-pacing tests no longer fail on Windows. Both asserted a
wall-clock lower bound equal to the nominal sleep total, but `asyncio.sleep`
returns fractionally early against a clock that ticks about every 15.6ms
there, so three paced acquires measured 0.187 against an asserted 0.2. The CI
matrix pins the Windows runner's seed, so this failed on every run rather than
intermittently. Both assertions now allow one clock tick per sleep, which
still leaves them failing by four orders of magnitude when pacing is removed.
