"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

import re
from typing import Any

from tool_eval_bench.evals.helpers import (
    UNIVERSAL_SIDE_EFFECT_TOOLS,
)
from tool_eval_bench.evals.helpers import (
    as_str as _as_str,
)

LARGE_SIDE_EFFECT_TOOLS: frozenset[str] = UNIVERSAL_SIDE_EFFECT_TOOLS | frozenset(
    {
        "update_customer",
        "create_ticket",
        "resolve_ticket",
        "create_return",
        "create_invoice",
        "process_payment",
        "transfer_funds",
        "deploy_service",
        "rollback_deploy",
        "scale_service",
        "restart_service",
        "submit_timesheet",
        "request_leave",
        "export_csv",
        "create_dashboard",
        "send_slack_message",
        "create_channel",
        "schedule_meeting",
        "post_announcement",
        "publish_page",
        "draft_blog_post",
        "upload_asset",
    }
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
