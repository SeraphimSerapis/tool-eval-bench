Every turn is streamed, not only the first. The read timeout therefore bounds
the gap between tokens on every turn, so a model that thinks for minutes
while still emitting tokens stays alive and a hung endpoint still fails after
`--timeout` seconds of silence. The turn-1-derived budget for unstreamed
later turns is gone with the asymmetry it compensated for.
