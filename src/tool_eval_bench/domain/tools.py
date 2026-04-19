"""Universal tool definitions for the agentic tool-call benchmark.

The 69 scenarios across 15 categories draw from this pool of 12 universal tools
in OpenAI function-calling format. Models receive the full set every time,
testing their ability to ignore irrelevant options.

Category L (Toolset Scale) uses an extended 52-tool set defined in
``domain/tools_large.py`` that includes these 12 plus 40 domain-specific tools.
"""

from __future__ import annotations

from typing import Any

UNIVERSAL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "cc": {"type": "string", "description": "CC recipient email address"},
                    "bcc": {"type": "string", "description": "BCC recipient email address"},
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name or content",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "file_type": {
                        "type": "string",
                        "enum": ["pdf", "docx", "xlsx", "any"],
                        "default": "any",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a specific file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                },
                "required": ["file_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string", "description": "Format: YYYY-MM-DD"},
                    "time": {"type": "string", "description": "Format: HH:MM"},
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone (e.g. Europe/Berlin, America/New_York). Defaults to UTC.",
                        "default": "UTC",
                    },
                    "duration_minutes": {"type": "integer", "default": 60},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["title", "date", "time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contacts",
            "description": "Look up contacts by name or group",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text from one language to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_language": {"type": "string"},
                    "target_language": {"type": "string"},
                },
                "required": ["text", "source_language", "target_language"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a future time",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "datetime": {"type": "string", "description": "Format: ISO 8601"},
                },
                "required": ["message", "datetime"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute a code snippet and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript"],
                    },
                    "code": {"type": "string"},
                },
                "required": ["language", "code"],
                "additionalProperties": False,
            },
        },
    },
]


# System prompt used for ALL scenarios
SYSTEM_PROMPT = """You are a helpful assistant with access to the tools provided.

Rules:
- Use a tool ONLY when it is necessary to fulfill the user's request.
- If you can answer directly from your own knowledge, do so without calling a tool.
- If a tool call fails, explain the failure and suggest an alternative approach.
- Never invent information that a tool should provide."""

# Fixed reference date for relative-time scenarios
BENCHMARK_REFERENCE_DATE = "2026-03-20"
BENCHMARK_REFERENCE_DAY = "Friday"
