"""Starlette ASGI application and JSON API routes for the local web UI."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tool_eval_bench import __version__
from tool_eval_bench.application.leaderboard import (
    compute_mcnemar_stats,
    extract_leaderboard_rows,
    shorten_model_name,
)
from tool_eval_bench.application.run_queries import (
    get_run,
    recent_runs,
    resolve_run,
    run_progress,
)
from tool_eval_bench.web.launcher import (
    RunConflict,
    build_launcher_schema,
    launch_web_run,
    probe_endpoint_models,
    resume_web_run,
    stop_web_run,
)
from tool_eval_bench.web.live import LiveRunHub, stream_run_events
from tool_eval_bench.web.sanitize import (
    sanitize_result,
    sanitize_run_for_web,
    stored_held_out_ids,
)
from tool_eval_bench.web.security import SecurityHeaders, SessionGate

# Matches the CLI leaderboard's window, so both rank the same runs.
_LEADERBOARD_WINDOW = 500

_STATIC_MIME_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
}


def _read_static_asset(name: str) -> tuple[bytes, str] | None:
    content_type = _STATIC_MIME_TYPES.get(name)
    if content_type is None:
        return None
    resource = files("tool_eval_bench.web.static").joinpath(name)
    if not resource.is_file():
        return None
    return resource.read_bytes(), content_type


def _parse_limit(raw: str) -> int | None:
    try:
        return max(1, min(500, int(raw)))
    except ValueError:
        return None


def _format_export_payload(rows: list[dict[str, Any]], fmt: str) -> tuple[str, str, str]:
    """Build export payload (body, content_type, filename) in CSV or JSON format."""
    all_cats: set[str] = set()
    for row in rows:
        all_cats.update(row.get("cat_scores", {}).keys())
    cat_order = sorted(all_cats)

    if fmt == "json":
        export_data: list[dict[str, Any]] = []
        for row in rows:
            entry: dict[str, Any] = {
                "model": row["model"],
                "run_id": row["run_id"],
                "date": row["date"],
                "final_score": row["final_score"],
                "rating": row["rating"],
                "total_points": row["total_points"],
                "max_points": row["max_points"],
                "passes": row["passes"],
                "partials": row["partials"],
                "fails": row["fails"],
                "scenario_count": row["scenario_count"],
                "backend": row["backend"],
                "total_tokens": row["total_tokens"],
                "num_runs": row["num_runs"],
                "cohort_label": row["cohort_label"],
                "cohort_fingerprint": row["cohort_fingerprint"],
                "categories": {cat: row["cat_scores"].get(cat) for cat in cat_order},
            }
            for opt_key in (
                "token_efficiency",
                "deployability",
                "median_turn_ms",
                "tool_version",
                "engine_name",
                "engine_version",
                "quantization",
                "max_model_len",
                "temperature",
                "server_model_root",
            ):
                if row.get(opt_key) is not None:
                    entry[opt_key] = row[opt_key]
            export_data.append(entry)
        return (
            json.dumps(export_data, indent=2, default=str),
            "application/json; charset=utf-8",
            "tool-eval-leaderboard.json",
        )

    headers = [
        "rank",
        "model",
        "run_id",
        "date",
        "final_score",
        "rating",
        "total_points",
        "max_points",
        "passes",
        "partials",
        "fails",
        "scenario_count",
        "backend",
        "total_tokens",
        "num_runs",
        "tool_version",
        "engine_name",
        "engine_version",
        "quantization",
        "max_model_len",
        "temperature",
        "server_model_root",
        "cohort",
        "cohort_fingerprint",
        *(f"cat_{cat}" for cat in cat_order),
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        csv_row: dict[str, Any] = {
            "rank": row["cohort_rank"],
            "model": row["model"],
            "run_id": row["run_id"],
            "date": row["date"],
            "final_score": row["final_score"],
            "rating": row["rating"],
            "total_points": row["total_points"],
            "max_points": row["max_points"],
            "passes": row["passes"],
            "partials": row["partials"],
            "fails": row["fails"],
            "scenario_count": row["scenario_count"],
            "backend": row["backend"],
            "total_tokens": row["total_tokens"],
            "num_runs": row["num_runs"],
            "tool_version": row.get("tool_version") or "",
            "engine_name": row.get("engine_name") or "",
            "engine_version": row.get("engine_version") or "",
            "quantization": row.get("quantization") or "",
            "max_model_len": row.get("max_model_len") or "",
            "temperature": row.get("temperature") if row.get("temperature") is not None else "",
            "server_model_root": row.get("server_model_root") or "",
            "cohort": row["cohort_label"],
            "cohort_fingerprint": row["cohort_fingerprint"],
        }
        for cat in cat_order:
            csv_row[f"cat_{cat}"] = row["cat_scores"].get(cat, "")
        writer.writerow(csv_row)
    return buf.getvalue(), "text/csv; charset=utf-8", "tool-eval-leaderboard.csv"


def _build_comparison_payload(raw_a: Mapping[str, Any], raw_b: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two stored runs after applying web redaction and enrichment."""
    run_a = sanitize_run_for_web(raw_a, include_traces=True)
    run_b = sanitize_run_for_web(raw_b, include_traces=True)

    results_a = run_a.get("scores", {}).get("scenario_results", [])
    results_b = run_b.get("scores", {}).get("scenario_results", [])
    map_a = {
        r["scenario_id"]: r for r in results_a if isinstance(r, Mapping) and "scenario_id" in r
    }
    map_b = {
        r["scenario_id"]: r for r in results_b if isinstance(r, Mapping) and "scenario_id" in r
    }
    all_ids = list(dict.fromkeys([*map_a.keys(), *map_b.keys()]))

    cfg_a = run_a.get("config") or {}
    cfg_b = run_b.get("config") or {}
    fp_a = cfg_a.get("config_fingerprint")
    fp_b = cfg_b.get("config_fingerprint")
    config_mismatch = bool(fp_a and fp_b and fp_a != fp_b)
    config_diffs: list[str] = []
    if config_mismatch:
        for key in ("scenario_count", "backend", "temperature", "seed", "error_rate"):
            va, vb = cfg_a.get(key), cfg_b.get(key)
            if va != vb:
                config_diffs.append(f"{key} ({va} vs {vb})")
        if not config_diffs:
            config_diffs.append("config fingerprints differ")

    improved = regressed = unchanged = 0
    deltas: list[dict[str, Any]] = []
    for sc_id in all_ids:
        ra = map_a.get(sc_id)
        rb = map_b.get(sc_id)
        ref = rb or ra or {}
        title = ref.get("title", sc_id)
        category = ref.get("category", "?")

        if ra and not rb:
            deltas.append(
                {
                    "scenario_id": sc_id,
                    "title": title,
                    "category": category,
                    "outcome": "removed",
                    "points_delta": None,
                    "duration_delta": None,
                    "a": ra,
                    "b": None,
                }
            )
            continue
        if rb and not ra:
            deltas.append(
                {
                    "scenario_id": sc_id,
                    "title": title,
                    "category": category,
                    "outcome": "added",
                    "points_delta": None,
                    "duration_delta": None,
                    "a": None,
                    "b": rb,
                }
            )
            continue
        if ra is None or rb is None:
            continue

        pts_a = int(ra.get("points", 0))
        pts_b = int(rb.get("points", 0))
        dur_a = float(ra.get("duration_seconds", 0.0) or 0.0)
        dur_b = float(rb.get("duration_seconds", 0.0) or 0.0)
        p_delta = pts_b - pts_a
        if p_delta > 0:
            improved += 1
            outcome = "improved"
        elif p_delta < 0:
            regressed += 1
            outcome = "regressed"
        else:
            unchanged += 1
            outcome = "unchanged"

        deltas.append(
            {
                "scenario_id": sc_id,
                "title": title,
                "category": category,
                "outcome": outcome,
                "points_delta": p_delta,
                "duration_delta": round(dur_b - dur_a, 2),
                "a": ra,
                "b": rb,
            }
        )

    total_a = sum(int(r.get("points", 0)) for r in results_a if isinstance(r, Mapping))
    total_b = sum(int(r.get("points", 0)) for r in results_b if isinstance(r, Mapping))
    score_a = run_a.get("final_score")
    score_b = run_b.get("final_score")
    score_delta = (
        score_b - score_a
        if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float))
        else None
    )
    mcnemar = compute_mcnemar_stats(map_a, map_b, all_ids)
    mcnemar["chi2"] = round(mcnemar["chi2"], 4)
    mcnemar["p_value"] = round(mcnemar["p_value"], 6)

    return {
        "run_a": run_a,
        "run_b": run_b,
        "config_mismatch": config_mismatch,
        "config_diffs": config_diffs,
        "summary": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "total_points_a": total_a,
            "total_points_b": total_b,
            "total_delta": total_b - total_a,
            "score_a": score_a,
            "score_b": score_b,
            "score_delta": score_delta,
        },
        "mcnemar": mcnemar,
        "deltas": deltas,
    }


async def _json_object(request: Request, *, required: bool) -> dict[str, Any]:
    """Parse the request body as a JSON object, raising ValueError for the client."""
    raw = await request.body()
    if not raw:
        if required:
            raise ValueError("Request body must be a JSON object")
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise ValueError("Request body must be valid JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError("Request body must be a JSON object")
    return parsed


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def build_starlette_app(
    *, token: str, output_dir: str | None = None, redact_url: bool = False
) -> Starlette:
    """Construct the Starlette application serving static assets and JSON endpoints.

    Read-only endpoints are plain functions, which Starlette runs in a worker
    thread: their SQLite reads would otherwise stall the event loop that also
    drives web-launched benchmarks and skew their latency measurements.
    """
    hub = LiveRunHub()

    def index_page(request: Request) -> Response:
        asset = _read_static_asset("index.html")
        if asset is None:
            return _error("UI assets missing", 500)
        body, content_type = asset
        return Response(body, media_type=content_type)

    def static_asset(request: Request) -> Response:
        asset = _read_static_asset(request.path_params.get("filename", ""))
        if asset is None:
            return _error("Asset not found", 404)
        body, content_type = asset
        return Response(body, media_type=content_type)

    def list_runs_endpoint(request: Request) -> JSONResponse:
        params = request.query_params
        limit = _parse_limit(params.get("limit", "50"))
        if limit is None:
            return _error(f"Invalid limit: {params.get('limit')!r}", 400)
        stored = recent_runs(
            limit=limit,
            model=params.get("model") or None,
            run_type=params.get("run_type") or None,
            status=params.get("status") or None,
            search=(params.get("q") or "").strip() or None,
        )
        return JSONResponse(
            {
                "version": __version__,
                "runs": [sanitize_run_for_web(run, include_traces=False) for run in stored],
            }
        )

    def get_run_endpoint(request: Request) -> JSONResponse:
        run_id = request.path_params.get("run_id", "")
        if run_id.lower() == "latest":
            resolved = resolve_run("latest")
            run = resolved[1] if resolved is not None else None
        else:
            run = get_run(run_id, include_traces=True)
        if run is None:
            return _error(f"Run '{run_id}' not found", 404)
        return JSONResponse(sanitize_run_for_web(run, include_traces=True))

    def leaderboard_endpoint(request: Request) -> JSONResponse:
        params = request.query_params
        limit = _parse_limit(params.get("limit", "50"))
        if limit is None:
            return _error(f"Invalid limit: {params.get('limit')!r}", 400)

        cohort_filter = (params.get("cohort") or "").strip()
        backend_filter = (params.get("backend") or "").strip().lower()
        search = (params.get("q") or "").strip().lower()

        all_rows = extract_leaderboard_rows(recent_runs(limit=_LEADERBOARD_WINDOW))
        cohorts_map: dict[str, dict[str, Any]] = {}
        for row in all_rows:
            row["display_model"] = shorten_model_name(str(row["model"]))
            fp = row["cohort_fingerprint"]
            if fp not in cohorts_map:
                cohorts_map[fp] = {
                    "cohort_fingerprint": fp,
                    "cohort_label": row["cohort_label"],
                    "count": 0,
                }
            cohorts_map[fp]["count"] += 1

        filtered = all_rows
        if cohort_filter:
            filtered = [r for r in filtered if r["cohort_fingerprint"] == cohort_filter]
        if backend_filter:
            filtered = [r for r in filtered if str(r.get("backend", "")).lower() == backend_filter]
        if search:
            filtered = [
                r
                for r in filtered
                if search in str(r.get("model", "")).lower()
                or search in str(r.get("display_model", "")).lower()
                or search in str(r.get("cohort_label", "")).lower()
            ]

        filtered = filtered[:limit]
        all_cats: set[str] = set()
        for row in filtered:
            all_cats.update(row.get("cat_scores", {}).keys())
        return JSONResponse(
            {
                "rows": filtered,
                "cohorts": list(cohorts_map.values()),
                "categories": sorted(all_cats),
            }
        )

    def export_endpoint(request: Request) -> Response:
        fmt = (request.query_params.get("format") or "json").lower()
        if fmt not in {"csv", "json"}:
            return _error(f"Unsupported format {fmt!r}. Use 'csv' or 'json'.", 400)
        rows = extract_leaderboard_rows(recent_runs(limit=_LEADERBOARD_WINDOW))
        body, content_type, filename = _format_export_payload(rows, fmt)
        return Response(
            body,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def compare_endpoint(request: Request) -> JSONResponse:
        params = request.query_params
        left = (params.get("left") or "").strip()
        right = (params.get("right") or "").strip()
        if not left or not right:
            return _error("Both 'left' and 'right' run IDs are required", 400)
        res_a = resolve_run(left)
        if res_a is None:
            return _error(f"Run '{left}' not found", 404)
        res_b = resolve_run(right)
        if res_b is None:
            return _error(f"Run '{right}' not found", 404)

        _, raw_a = res_a
        _, raw_b = res_b
        results_a = (raw_a.get("scores") or {}).get("scenario_results")
        results_b = (raw_b.get("scores") or {}).get("scenario_results")
        if not results_a or not results_b:
            return _error("Both runs must contain scenario results to compare", 400)
        return JSONResponse(_build_comparison_payload(raw_a, raw_b))

    def run_progress_endpoint(request: Request) -> JSONResponse:
        run_id = request.path_params.get("run_id", "")
        progress = run_progress(run_id)
        if progress is None:
            return _error(f"Run '{run_id}' not found", 404)
        run, checkpoints = progress
        held_out = stored_held_out_ids(run)
        return JSONResponse(
            {
                "run": sanitize_run_for_web(run, include_traces=False),
                "checkpoints": [
                    sanitize_result(cp, held_out) for cp in checkpoints if isinstance(cp, Mapping)
                ],
            }
        )

    def schema_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(build_launcher_schema())

    async def run_events_endpoint(request: Request) -> Response:
        run_id = request.path_params.get("run_id", "")
        if hub.get(run_id) is None:
            stored = await run_in_threadpool(get_run, run_id, include_traces=False)
            if stored is None:
                return _error(f"Run '{run_id}' not found", 404)
        return StreamingResponse(
            stream_run_events(hub, run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def probe_endpoint(request: Request) -> JSONResponse:
        try:
            payload = await _json_object(request, required=False)
        except ValueError as exc:
            return _error(str(exc), 400)
        result = await probe_endpoint_models(
            base_url=str(payload.get("base_url") or "") or None,
            api_key=str(payload.get("api_key") or "") or None,
            wire_format=str(payload.get("wire_format") or "auto"),
            redact=redact_url,
        )
        return JSONResponse(result)

    async def launch_run_endpoint(request: Request) -> JSONResponse:
        try:
            payload = await _json_object(request, required=True)
            started = await launch_web_run(hub, payload, output_dir=output_dir)
        except ValueError as exc:
            return _error(str(exc), 400)
        return JSONResponse(started, status_code=202)

    async def stop_run_endpoint(request: Request) -> JSONResponse:
        run_id = request.path_params.get("run_id", "")
        try:
            stopped = await stop_web_run(hub, run_id)
        except KeyError:
            return _error(f"Run '{run_id}' not found", 404)
        except RunConflict as exc:
            return _error(str(exc), 409)
        return JSONResponse(stopped)

    async def resume_run_endpoint(request: Request) -> JSONResponse:
        run_id = request.path_params.get("run_id", "")
        try:
            payload = await _json_object(request, required=False)
            resumed = await resume_web_run(hub, run_id, payload, output_dir=output_dir)
        except KeyError:
            return _error(f"Run '{run_id}' not found", 404)
        except RunConflict as exc:
            return _error(str(exc), 409)
        except ValueError as exc:
            return _error(str(exc), 400)
        return JSONResponse(resumed, status_code=202)

    routes = [
        Route("/", endpoint=index_page, methods=["GET"]),
        Route("/static/{filename:path}", endpoint=static_asset, methods=["GET"]),
        Route("/api/schema", endpoint=schema_endpoint, methods=["GET"]),
        Route("/api/probe", endpoint=probe_endpoint, methods=["POST"]),
        Route("/api/runs", endpoint=list_runs_endpoint, methods=["GET"]),
        Route("/api/runs", endpoint=launch_run_endpoint, methods=["POST"]),
        Route("/api/runs/{run_id}", endpoint=get_run_endpoint, methods=["GET"]),
        Route("/api/runs/{run_id}/progress", endpoint=run_progress_endpoint, methods=["GET"]),
        Route("/api/runs/{run_id}/events", endpoint=run_events_endpoint, methods=["GET"]),
        Route("/api/runs/{run_id}/stop", endpoint=stop_run_endpoint, methods=["POST"]),
        Route("/api/runs/{run_id}/resume", endpoint=resume_run_endpoint, methods=["POST"]),
        Route("/api/leaderboard", endpoint=leaderboard_endpoint, methods=["GET"]),
        Route("/api/export", endpoint=export_endpoint, methods=["GET"]),
        Route("/api/compare", endpoint=compare_endpoint, methods=["GET"]),
    ]
    app = Starlette(
        debug=False,
        routes=routes,
        # Outermost first: the gate's own 401 and 403 responses get the headers too.
        middleware=[Middleware(SecurityHeaders), Middleware(SessionGate, token=token)],
    )
    app.state.output_dir = output_dir
    app.state.redact_url = redact_url
    app.state.live_hub = hub
    return app
