"""Large tool set for toolset-scale benchmark scenarios.

Extends the 12 universal tools with ~40 domain-specific tools to create
a 50+ tool namespace. Used by scenarios TC-37 through TC-40 to test
model ability to navigate a crowded tool list.
"""

from __future__ import annotations

from typing import Any

from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS

# ---------------------------------------------------------------------------
# Domain-specific tools (40 additional tools)
# ---------------------------------------------------------------------------

_DOMAIN_TOOLS: list[dict[str, Any]] = [
    # --- CRM ---
    {
        "type": "function",
        "function": {
            "name": "get_customer_profile",
            "description": "Retrieve a customer's profile by name or customer ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID or full name"},
                    "include_history": {"type": "boolean", "default": False},
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_customer",
            "description": "Update a customer's profile fields",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "fields": {"type": "object", "description": "Key-value pairs to update"},
                },
                "required": ["customer_id", "fields"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket for a customer issue",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["customer_id", "subject", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_ticket",
            "description": "Resolve an existing support ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                    "resolution_notes": {"type": "string"},
                },
                "required": ["ticket_id", "resolution_notes"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status of a customer order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID or customer name"},
                    "include_tracking": {"type": "boolean", "default": True},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipping_status",
            "description": "Get shipping and tracking details for a shipment",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {"type": "string"},
                    "carrier": {"type": "string", "enum": ["fedex", "ups", "usps", "dhl"]},
                },
                "required": ["tracking_number"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Initiate a return/refund for an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "refund_method": {"type": "string", "enum": ["original", "store_credit"]},
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tickets",
            "description": "List open support tickets, optionally filtered by customer or priority",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Filter by customer ID"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "status": {"type": "string", "enum": ["open", "in_progress", "resolved"]},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    # --- Finance ---
    {
        "type": "function",
        "function": {
            "name": "get_invoice",
            "description": "Retrieve an invoice by invoice ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                },
                "required": ["invoice_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_invoice",
            "description": "Create a new invoice for a customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "object"}},
                    "due_date": {"type": "string", "description": "Format: YYYY-MM-DD"},
                },
                "required": ["customer_id", "items"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_payment",
            "description": "Process a payment against an invoice",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "method": {"type": "string", "enum": ["credit_card", "bank_transfer", "crypto"]},
                },
                "required": ["invoice_id", "amount"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Get the current account balance for a customer or account",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                },
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_funds",
            "description": "Transfer funds between accounts",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_account": {"type": "string"},
                    "to_account": {"type": "string"},
                    "amount": {"type": "number"},
                    "currency": {"type": "string", "default": "USD"},
                },
                "required": ["from_account", "to_account", "amount"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_financial_report",
            "description": "Generate a financial report for a given period",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_type": {"type": "string", "enum": ["income", "balance_sheet", "cash_flow"]},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["report_type", "start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    # --- DevOps ---
    {
        "type": "function",
        "function": {
            "name": "deploy_service",
            "description": "Deploy a service to production or staging environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "version": {"type": "string"},
                    "environment": {"type": "string", "enum": ["staging", "production"]},
                },
                "required": ["service_name", "version", "environment"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback_deploy",
            "description": "Rollback a deployment to the previous version",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "environment": {"type": "string", "enum": ["staging", "production"]},
                },
                "required": ["service_name", "environment"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_health",
            "description": "Check the health and status of a running service",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "environment": {"type": "string", "enum": ["staging", "production"]},
                },
                "required": ["service_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_containers",
            "description": "List running containers for a service",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "status_filter": {"type": "string", "enum": ["running", "stopped", "all"]},
                },
                "required": ["service_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_service",
            "description": "Scale a service up or down by adjusting replica count",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "replicas": {"type": "integer"},
                    "environment": {"type": "string", "enum": ["staging", "production"]},
                },
                "required": ["service_name", "replicas"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "Retrieve logs for a service or container",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "lines": {"type": "integer", "default": 100},
                    "level": {"type": "string", "enum": ["debug", "info", "warn", "error"]},
                },
                "required": ["service_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart a running service",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "environment": {"type": "string", "enum": ["staging", "production"]},
                    "graceful": {"type": "boolean", "default": True},
                },
                "required": ["service_name"],
                "additionalProperties": False,
            },
        },
    },
    # --- HR ---
    {
        "type": "function",
        "function": {
            "name": "get_employee",
            "description": "Look up employee details by name or employee ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_timesheet",
            "description": "Submit a weekly timesheet for an employee",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "week_start": {"type": "string", "description": "Format: YYYY-MM-DD"},
                    "hours": {"type": "number"},
                },
                "required": ["employee_id", "week_start", "hours"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_leave",
            "description": "Submit a leave/vacation request",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "type": {"type": "string", "enum": ["vacation", "sick", "personal"]},
                },
                "required": ["employee_id", "start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_payslip",
            "description": "Get a payslip for an employee for a specific month",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "month": {"type": "string", "description": "Format: YYYY-MM"},
                },
                "required": ["employee_id", "month"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_org_chart",
            "description": "Retrieve the organizational chart for a department",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {"type": "string"},
                    "include_contractors": {"type": "boolean", "default": False},
                },
                "required": ["department"],
                "additionalProperties": False,
            },
        },
    },
    # --- Data & Analytics ---
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a read-only SQL query against the analytics database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "database": {"type": "string", "enum": ["analytics", "reporting", "warehouse"]},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_csv",
            "description": "Export query results to a CSV file",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filename": {"type": "string"},
                },
                "required": ["query", "filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_dashboard",
            "description": "Create a new analytics dashboard",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "widgets": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analytics",
            "description": "Run a predefined analytics pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_name": {"type": "string"},
                    "parameters": {"type": "object"},
                },
                "required": ["pipeline_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "Get system or business metrics for a time range",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "granularity": {"type": "string", "enum": ["minute", "hour", "day", "week"]},
                },
                "required": ["metric_name"],
                "additionalProperties": False,
            },
        },
    },
    # --- Communication ---
    {
        "type": "function",
        "function": {
            "name": "send_slack_message",
            "description": "Send a message to a Slack channel or user",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                    "thread_ts": {"type": "string", "description": "Thread timestamp for replies"},
                },
                "required": ["channel", "message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_channel",
            "description": "Create a new Slack channel",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "is_private": {"type": "boolean", "default": False},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with participants via calendar integration",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                    "datetime": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 30},
                },
                "required": ["title", "participants", "datetime"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_meeting_notes",
            "description": "Retrieve notes from a past meeting",
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {"type": "string"},
                },
                "required": ["meeting_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_announcement",
            "description": "Post a company-wide announcement",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "audience": {"type": "string", "enum": ["all", "engineering", "management"]},
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    # --- Content ---
    {
        "type": "function",
        "function": {
            "name": "publish_page",
            "description": "Publish a wiki or documentation page",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "space": {"type": "string"},
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_blog_post",
            "description": "Create a draft blog post",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_asset",
            "description": "Upload a file or media asset to the asset library",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "category": {"type": "string", "enum": ["images", "documents", "videos"]},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_content_calendar",
            "description": "Retrieve the content publication calendar for a given week",
            "parameters": {
                "type": "object",
                "properties": {
                    "week_start": {"type": "string", "description": "Format: YYYY-MM-DD"},
                },
                "required": ["week_start"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Combined large toolset (12 universal + 40 domain = 52 total)
# ---------------------------------------------------------------------------

LARGE_TOOLSET: list[dict[str, Any]] = UNIVERSAL_TOOLS + _DOMAIN_TOOLS
"""52-tool set combining universal tools with domain-specific tools."""

LARGE_TOOLSET_SIZE: int = len(LARGE_TOOLSET)
"""Number of tools in the large toolset (for documentation/assertions)."""
