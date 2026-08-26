`get_scenario_results` always rehydrated every scenario trace, then discarded them when the caller
only wanted scores. The run diff, its one production caller, reads points and status only. It now
opts out, skipping a multi-megabyte read and a full result-dict rebuild.
