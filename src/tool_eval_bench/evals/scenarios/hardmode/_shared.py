"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

_UNRELATED_UNIVERSAL_MUTATIONS = frozenset({"set_reminder", "run_code"})
