**An interrupted context-pressure sweep no longer reports a breaking point.** A sweep stopped with
Ctrl-C was saved like a finished one, with a breaking point taken from the levels it reached,
which is only a lower bound. Every sweep now stores `interrupted` and `planned_levels` in its
scores. An interrupted sweep keeps its first degradation, stores a null breaking point, and its
report says after how many of the planned levels it stopped. The run status stays `completed`.
