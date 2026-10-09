"""Helpers shared by more than one scenario in this group.

Scoped to the group rather than global: several groups define helpers under the same name with deliberately different behaviour.
"""

from __future__ import annotations

from typing import Any

_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_DOCUMENT_ID = {"type": "string", "description": "Document identifier, such as DOC-HB-2024"}


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
