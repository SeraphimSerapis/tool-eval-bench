from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any

import httpx

from tool_eval_bench.utils.urls import models_url as _models_url

from tool_eval_bench.domain.models import BenchmarkConfig


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


async def _probe_backend(base_url: str, api_key: str | None) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    probe: dict[str, Any] = {"ok": False}
    url = _models_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            probe["status_code"] = resp.status_code
            if resp.status_code == 200:
                body = resp.json()
                data = body.get("data") if isinstance(body, dict) else None
                if isinstance(data, list) and data:
                    first = data[0] if isinstance(data[0], dict) else {}
                    probe["first_model_id"] = first.get("id")
                probe["ok"] = True
    except Exception as exc:
        probe["error"] = str(exc)
    return probe


async def collect_run_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
        "config": {
            "model": config.model,
            "backend": config.backend,
            "base_url": config.base_url,
        },
        "backend_probe": await _probe_backend(config.base_url, config.api_key),
    }

