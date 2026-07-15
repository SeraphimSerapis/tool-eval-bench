"""Stable CLI entrypoint and compatibility surface.

Implementation lives in :mod:`tool_eval_bench.cli.dispatch`.  Publishing the
dispatch module under this historical name preserves private imports and
monkeypatch seams used by existing integrations while keeping the entrypoint
itself intentionally thin.
"""

from __future__ import annotations

import sys

from tool_eval_bench.cli import dispatch as _dispatch

if __name__ == "__main__":
    _dispatch.main()
else:
    sys.modules[__name__] = _dispatch
