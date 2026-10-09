"""Stable CLI entrypoint and compatibility surface.

Implementation lives in :mod:`tool_eval_bench.cli.dispatch`.  Publishing the
dispatch module under this historical name preserves private imports and
monkeypatch seams used by existing integrations while keeping the entrypoint
itself intentionally thin.

``_metadata_for_storage``, ``_persist_plugin_run``, ``_with_config_fingerprint``
and ``_parse_sweep_range`` stay importable from here, but patching them no
longer changes how sweeps, spec-bench, throughput-only runs or plugins finalize.
Those modes go through ``application.mode_runs.finalize_mode_run``; patch
``application.run_queries.persist_run`` to intercept persistence,
``application.mode_runs.write_mode_report`` to intercept the report, and
``application.mode_runs.with_config_fingerprint`` (defined in
``utils.fingerprint``) to change the stored config.

``_decision_judge_kwargs``, ``RunSettings``, ``with_selected_checks`` and
``decision_judge_config`` are importable from here too, but patching them no
longer changes the scored runners or the resume compatibility check. Both build
from ``cli.scored_run.ScoredRun``; patch ``ScoredRun.service_kwargs`` or
``ScoredRun.run_settings`` instead.
"""

from __future__ import annotations

import sys

from tool_eval_bench.cli import dispatch as _dispatch

if __name__ == "__main__":
    _dispatch.main()
else:
    sys.modules[__name__] = _dispatch
