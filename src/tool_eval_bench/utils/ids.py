from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json


def build_run_id(payload: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:6]
    return f"{ts}_{digest}"
