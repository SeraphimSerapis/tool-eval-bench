The adaptive-pacing tests now assert on the spacing the rate-limit coordinator
reserves rather than on how long the wall clock said they took. Both used to
sleep for real and check a lower bound, which made them the slowest tests in
the suite and left them at the mercy of platform clock granularity. They now
run against a virtual clock the test advances, so they check exact values
instead of a floor: three paced requests wait one step each, and a 429 seen by
one request makes all four wait. Both finish in under five milliseconds.
