"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from typing import Any

from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)


def _is_engineering_department(value: Any) -> bool:
    """Return whether a department value requests Engineering itself.

    The scenario accepts natural variants such as "Engineering team", but a
    negated value like "not engineering" must not resolve to the canonical
    Engineering org chart merely because it contains the word "engineering".
    """
    department = _as_str(value)
    return bool(
        re.search(r"\bengineering\b", department, re.IGNORECASE)
        and not re.search(
            r"\b(?:not|no|without)\s+(?:the\s+)?engineering\b",
            department,
            re.IGNORECASE,
        )
    )
