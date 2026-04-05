"""Deterministic payload enrichment for mock tool handlers.

Adds realistic "noise" fields to mock responses — extra metadata, timestamps,
nested objects, and pagination markers that real APIs return. This tests whether
models can extract the relevant fields from noisy payloads.

All enrichment is deterministic (no randomness) so benchmarks stay reproducible.
Enrichment wraps *around* existing data — core fields stay identical so
evaluators continue to work unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Any


def _seed_from_payload(payload: dict[str, Any], salt: str = "") -> int:
    """Derive a deterministic integer seed from payload content.

    Produces different noise for different scenarios while remaining
    fully reproducible (same input → same noise).
    """
    # Use sorted string representation of payload + salt for stability
    key = f"{salt}:{sorted(payload.items())}"
    return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:8], 16)


def _seeded_id(prefix: str, seed: int) -> str:
    """Generate a deterministic hex ID from a seed."""
    return f"{prefix}{seed:08x}"


# ---------------------------------------------------------------------------
# Per-tool enrichment functions
# ---------------------------------------------------------------------------

def enrich_weather(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to weather responses."""
    seed = _seed_from_payload(payload, "weather")
    return {
        **payload,
        "wind_speed_kmh": 14.2 + (seed % 20) / 10.0,
        "wind_direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][seed % 8],
        "uv_index": max(1, seed % 6),
        "visibility_km": 9.8,
        "pressure_hpa": 1008 + seed % 20,
        "feels_like": payload.get("temperature", 0) - 2,
        "dew_point": payload.get("temperature", 0) - 5,
        "forecast_summary": "Conditions expected to remain similar for the next 6 hours.",
        "last_updated": "2026-03-20T12:00:00Z",
        "data_source": "National Weather Service",
        "station_id": _seeded_id("WXSTN-", seed),
        "request_id": _seeded_id("req_wx_", seed),
    }


def enrich_search(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to web search responses."""
    seed = _seed_from_payload(payload, "search")
    results = payload.get("results", [])
    enriched_results = []
    for i, r in enumerate(results):
        enriched_results.append({
            **r,
            "url": f"https://example.com/result/{i + 1}",
            "rank": i + 1,
            "relevance_score": round(0.95 - i * 0.05, 2),
            "published_date": "2026-03-18",
            "source_domain": "example.com",
            "language": "en",
        })
    return {
        "results": enriched_results,
        "total_results": 1200 + seed % 200,
        "page": 1,
        "per_page": 5,
        "query_time_ms": 30 + seed % 40,
        "source_engine": "web-index-v3",
        "cached": False,
        "safe_search": True,
        "related_queries": ["similar topic", "related question"],
        "request_id": _seeded_id("req_ws_", seed),
    }


def enrich_file_search(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to file search responses."""
    results = payload.get("results", [])
    enriched_results = []
    for r in results:
        enriched_results.append({
            **r,
            "size_bytes": 28_416,
            "modified_at": "2026-03-15T09:22:11Z",
            "created_at": "2026-02-10T14:00:00Z",
            "owner": "system",
            "path": f"/documents/{r.get('name', 'unknown')}",
            "permissions": "read",
            "content_type": "application/octet-stream",
        })
    return {
        "results": enriched_results,
        "total_matches": len(results),
        "search_time_ms": 18,
        "index_version": "idx-2026.03",
        "request_id": "req_fs_3d4a8e2b",
    }


def enrich_file_read(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to file read responses."""
    content = payload.get("content", "")
    return {
        **payload,
        "encoding": "utf-8",
        "mime_type": "text/plain",
        "size_bytes": len(content.encode("utf-8")) if isinstance(content, str) else 0,
        "last_modified": "2026-03-15T09:22:11Z",
        "version": 3,
        "permissions": {"read": True, "write": False},
        "line_count": content.count("\n") + 1 if isinstance(content, str) else 0,
        "request_id": "req_rf_1b5c7d3e",
    }


def enrich_email(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to email send responses."""
    return {
        **payload,
        "timestamp": "2026-03-20T12:05:33Z",
        "thread_id": "thread_e9a1f4c2",
        "headers": {
            "X-Mailer": "tool-eval-bench/1.0",
            "Content-Type": "text/plain; charset=utf-8",
            "X-Priority": "3",
        },
        "delivery_status": "accepted",
        "queue_position": 0,
        "estimated_delivery": "2026-03-20T12:05:35Z",
        "request_id": "req_em_5f2a9c1d",
    }


def enrich_calendar(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to calendar event responses."""
    return {
        **payload,
        "calendar_id": "cal_primary",
        "created_at": "2026-03-20T12:00:00Z",
        "updated_at": "2026-03-20T12:00:00Z",
        "organizer": {"email": "user@company.com", "display_name": "Current User"},
        "reminders": {"use_default": True, "overrides": []},
        "conference_link": None,
        "visibility": "default",
        "color_id": "7",
        "recurrence": None,
        "request_id": "req_ce_4a1d8b3f",
    }


def enrich_contacts(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to contact lookup responses."""
    results = payload.get("results", [])
    enriched_results = []
    for i, r in enumerate(results):
        enriched_results.append({
            **r,
            "id": f"contact_{1000 + i}",
            "department": "Engineering",
            "phone": "+1-555-0100",
            "title": "Team Member",
            "last_contacted": "2026-03-18T15:30:00Z",
            "notes": "",
            "source": "directory",
        })
    return {
        "results": enriched_results,
        "total_contacts": len(results),
        "directory_version": "2026.03",
        "request_id": "req_ct_6e3b2a1c",
    }


def enrich_stock(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to stock price responses."""
    price = payload.get("price", 0)
    return {
        **payload,
        "timestamp": "2026-03-20T16:00:00Z",
        "exchange": "NASDAQ",
        "volume": 52_314_800,
        "market_cap": "2.89T",
        "pe_ratio": 28.4,
        "day_high": round(price * 1.012, 2),
        "day_low": round(price * 0.988, 2),
        "week_52_high": round(price * 1.25, 2),
        "week_52_low": round(price * 0.72, 2),
        "previous_close": round(price - 1.23, 2),
        "after_hours": None,
        "request_id": "req_sp_8c1d4e2a",
    }


def enrich_translation(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to translation responses."""
    text = payload.get("translated", "")
    return {
        **payload,
        "source_detected": "en",
        "confidence": 0.98,
        "alternatives": [],
        "word_count": len(text.split()) if isinstance(text, str) else 0,
        "character_count": len(text) if isinstance(text, str) else 0,
        "api_version": "v3.1",
        "model": "nmt-2026",
        "request_id": "req_tr_2b7a5d1e",
    }


def enrich_code_execution(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to code execution responses."""
    return {
        **payload,
        "execution_time_ms": 12,
        "memory_used_kb": 2048,
        "sandbox_id": "sandbox_f3a1c9d2",
        "runtime_version": "3.11.8",
        "cpu_time_ms": 8,
        "wall_time_ms": 14,
        "request_id": "req_rc_9d3f1a2c",
    }


def enrich_reminder(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to reminder responses."""
    return {
        **payload,
        "created_at": "2026-03-20T12:00:00Z",
        "notification_channels": ["push", "email"],
        "repeat": None,
        "priority": "normal",
        "request_id": "req_rm_4c2a1d3b",
    }


def enrich_generic_error(payload: dict[str, Any]) -> dict[str, Any]:
    """Add realistic metadata to error responses."""
    seed = _seed_from_payload(payload, "error")
    return {
        **payload,
        "error_code": "ERR_TOOL_UNAVAILABLE",
        "timestamp": "2026-03-20T12:00:00Z",
        "trace_id": _seeded_id("trace_", seed),
        "documentation_url": "https://docs.example.com/errors/ERR_TOOL_UNAVAILABLE",
        "request_id": _seeded_id("req_err_", seed),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ENRICHERS: dict[str, Any] = {
    "get_weather": enrich_weather,
    "web_search": enrich_search,
    "search_files": enrich_file_search,
    "read_file": enrich_file_read,
    "send_email": enrich_email,
    "create_calendar_event": enrich_calendar,
    "get_contacts": enrich_contacts,
    "get_stock_price": enrich_stock,
    "translate_text": enrich_translation,
    "run_code": enrich_code_execution,
    "set_reminder": enrich_reminder,
    "calculator": lambda p: p,  # calculator results are already minimal by design
}


def enrich_payload(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic payload enrichment for a given tool.

    Wraps the original payload with realistic extra fields.
    If the payload contains an "error" key, enriches it with error metadata instead.
    Unknown tools get the payload returned as-is.
    """
    if not isinstance(payload, dict):
        return payload

    if "error" in payload:
        return enrich_generic_error(payload)

    enricher = _ENRICHERS.get(tool_name)
    if enricher is not None:
        return enricher(payload)

    return payload
