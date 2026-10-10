"""Assemble a shareable results card from stored runs.

:func:`build_card_data` is pure: it maps stored run rows onto
:class:`~tool_eval_bench.domain.card.CardData`. :func:`load_card_data` picks
the rows, by session or by run id, through ``application.run_queries``.

A card is meant to be posted publicly, so the mapping copies named fields
only. It never forwards a whole config or metadata blob, which is what keeps
hostnames, server URLs, session and run ids, labels, tokenizer paths, and
system prompts off the image.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from statistics import fmean
from typing import Any

from tool_eval_bench.application import run_queries
from tool_eval_bench.domain.card import (
    AccuracyResult,
    CardData,
    CategoryScore,
    NeedleCell,
    NeedleSummary,
    PerfPoint,
    ToolEvalSummary,
)
from tool_eval_bench.domain.scenarios import CATEGORY_LABELS, Category, rating_for_score
from tool_eval_bench.storage.db import SESSION_PREFIX_MIN_LENGTH

_FINGERPRINT_CHARS = 10

# Accuracy plugins in the order the card lists them, with display names.
_ACCURACY_KINDS: tuple[tuple[str, str], ...] = (
    ("gsm8k", "GSM8K"),
    ("mmlu", "MMLU"),
    ("ifeval", "IFEval"),
)
_CARD_KINDS = frozenset({"tool_eval", "perf", "needle", *(kind for kind, _ in _ACCURACY_KINDS)})
# ``failed`` because a failed perf run still stores its successful cells.
_CARD_STATUSES = ("completed", "failed")


class CardDataUnavailable(LookupError):
    """No stored run matches the request, or none of the matches has anything to show."""


def build_card_data(runs: Sequence[Mapping[str, Any]], *, hardware: str | None = None) -> CardData:
    """Map stored runs onto a card, using the most recent usable run of each kind.

    ``runs`` are rows shaped like ``RunRepository.get()``, in any order.
    ``hardware`` is free text from the user; nothing here detects it.
    Raises :class:`CardDataUnavailable` when no run has a section to show.
    """
    usable = sorted(
        (run for run in runs if _usable(run)), key=lambda run: str(run.get("created_at") or "")
    )
    newest_first = list(reversed(usable))

    tool_eval_trials = _tool_eval_trials(newest_first)
    tool_eval_run = tool_eval_trials[0] if tool_eval_trials else None
    perf_run = next((run for run in newest_first if _perf_rows(run)), None)
    needle_run = _latest(newest_first, "needle")
    accuracy_runs = [
        (name, run) for kind, name in _ACCURACY_KINDS if (run := _latest(newest_first, kind))
    ]

    used = [
        run
        for run in (tool_eval_run, perf_run, needle_run, *(run for _, run in accuracy_runs))
        if run is not None
    ]
    if not used:
        raise CardDataUnavailable(
            "None of the selected runs has a section a card can show: run tool-call "
            "scenarios, --perf, --needle, or an accuracy plugin first"
        )
    # Metadata comes from the runs the card shows, newest first, so a sparse
    # older row only fills a field the newer ones left empty.
    used.sort(key=lambda run: str(run.get("created_at") or ""), reverse=True)

    perf = _perf_points(_perf_rows(perf_run)) if perf_run is not None else ()
    pp, tg = _perf_workload(perf_run) if perf_run is not None else (None, None)
    fingerprint_source = tool_eval_run or perf_run or used[0]
    fingerprint = _config(fingerprint_source).get("config_fingerprint")

    return CardData(
        model=_model_name(used),
        tool_version=_first_str(used, "metadata", "tool_version") or "unknown",
        created_at=str(used[0].get("created_at") or ""),
        engine=_first_str(used, "metadata", "engine_name"),
        engine_version=_first_str(used, "metadata", "engine_version"),
        quantization=_first_str(used, "metadata", "quantization"),
        hardware=_str_or_none(hardware.strip() if hardware else None),
        max_context=_first_int(used, "metadata", "max_model_len"),
        perf_preset=_str_or_none(_config(perf_run).get("perf_preset")) if perf_run else None,
        perf_pp=pp,
        perf_tg=tg,
        perf=perf,
        tool_eval=_tool_eval_summary(tool_eval_trials) if tool_eval_trials else None,
        needle=_needle_summary(needle_run) if needle_run else None,
        accuracy=tuple(
            result for name, run in accuracy_runs if (result := _accuracy(name, run)) is not None
        ),
        seed=_first_int(used, "config", "seed"),
        fingerprint=str(fingerprint)[:_FINGERPRINT_CHARS] if fingerprint else None,
    )


def load_card_data(
    *,
    session: str | None = "latest",
    run_ids: Sequence[str] = (),
    model: str | None = None,
    hardware: str | None = None,
) -> CardData:
    """Load the runs for one card and build it.

    Explicit ``run_ids`` win over ``session``, which is how a card is made
    from runs stored before sessions existed. Otherwise ``session`` names a
    session id or a unique prefix of one, and ``latest`` or ``None`` picks the
    newest session holding a run of a kind the card shows, narrowed to
    ``model`` when given. Raises :class:`CardDataUnavailable` when nothing
    matches.
    """
    if run_ids:
        runs: list[dict[str, Any]] = []
        missing: list[str] = []
        for run_id in run_ids:
            run = run_queries.get_run(run_id, include_traces=False)
            if run is None:
                missing.append(run_id)
            else:
                runs.append(run)
        if missing:
            raise CardDataUnavailable(f"No stored run with id {', '.join(missing)}")
        # Compare the name the card would print, so a path and its basename
        # count as one model.
        models = sorted({_model_name([run]) for run in runs})
        if len(models) > 1:
            raise CardDataUnavailable(
                f"The --run ids cover more than one model ({', '.join(models)}). "
                "A card shows one model; pass runs of a single model"
            )
        return build_card_data(runs, hardware=hardware)

    latest = session is None or session.lower() == "latest"
    session_id = _latest_session(model) if latest or session is None else _resolve_session(session)
    runs = run_queries.session_runs(session_id)
    if not runs:
        raise CardDataUnavailable(f"No runs found for session {session_id}")
    try:
        return build_card_data(runs, hardware=hardware)
    except CardDataUnavailable:
        if not latest:
            raise
        # The lookup matched on type and status only; a failed scored run or a
        # perf run without cells passes it and still leaves nothing to draw.
        raise CardDataUnavailable(
            f"The newest session, {session_id[:8]}, has no run a card can show. "
            "Pass an older session id or --run RUN_ID"
        ) from None


def _latest_session(model: str | None) -> str:
    session_id = run_queries.latest_session_id(
        model=model, run_types=sorted(_CARD_KINDS), statuses=_CARD_STATUSES
    )
    if session_id is None:
        scope = f" for model {model}" if model else ""
        raise CardDataUnavailable(
            f"No benchmark session found{scope}. Runs stored before sessions "
            "existed can still be carded with --run RUN_ID"
        )
    return session_id


def _resolve_session(prefix: str) -> str:
    try:
        session_id = run_queries.resolve_session_prefix(prefix)
    except ValueError as exc:  # ambiguous prefix; the message lists the matches
        raise CardDataUnavailable(str(exc)) from None
    if session_id is None:
        raise CardDataUnavailable(
            f"No session matches {prefix}. Pass the full session id or a unique "
            f"prefix of at least {SESSION_PREFIX_MIN_LENGTH} characters"
        )
    return session_id


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------


def _usable(run: Mapping[str, Any]) -> bool:
    """Whether a run belongs on a card at all.

    A ``failed`` perf run still stores its successful cells; every other kind
    needs a completed run. Older rows without a status are treated as
    completed.
    """
    kind = run.get("run_type") or "tool_eval"
    if kind not in _CARD_KINDS or not isinstance(run.get("scores"), Mapping):
        return False
    status = run.get("status") or "completed"
    return status == "completed" or (kind == "perf" and status == "failed")


def _latest(newest_first: Iterable[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    return next((run for run in newest_first if (run.get("run_type") or "tool_eval") == kind), None)


def _session_id(run: Mapping[str, Any]) -> str | None:
    return _str_or_none(run.get("session_id")) or _str_or_none(
        _section(run, "metadata").get("session_id")
    )


def _tool_eval_trials(newest_first: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The newest completed tool-eval run and its sibling trials, newest first.

    ``--trials N`` stores one row per trial. Each records ``trials`` in its
    metadata, and the rows of one invocation share a session id and a config
    fingerprint. Rows without a session id are never grouped, because nothing
    ties them to one invocation.
    """
    runs = [run for run in newest_first if (run.get("run_type") or "tool_eval") == "tool_eval"]
    if not runs:
        return []
    newest = runs[0]
    planned = _int_or_none(_section(newest, "metadata").get("trials")) or 1
    session = _session_id(newest)
    if planned <= 1 or session is None:
        return [newest]
    fingerprint = _config(newest).get("config_fingerprint")
    siblings = [
        run
        for run in runs
        if _session_id(run) == session and _config(run).get("config_fingerprint") == fingerprint
    ]
    return siblings[:planned]


def _config(run: Mapping[str, Any] | None) -> Mapping[str, Any]:
    config = run.get("config") if run is not None else None
    return config if isinstance(config, Mapping) else {}


def _section(run: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = run.get(key)
    return value if isinstance(value, Mapping) else {}


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_or_none(value: Any) -> int | None:
    if not _is_number(value):
        return None
    return int(value)


def _first_str(runs: Iterable[Mapping[str, Any]], section: str, key: str) -> str | None:
    return next(
        (value for run in runs if (value := _str_or_none(_section(run, section).get(key)))), None
    )


def _first_int(runs: Iterable[Mapping[str, Any]], section: str, key: str) -> int | None:
    return next(
        (
            value
            for run in runs
            if (value := _int_or_none(_section(run, section).get(key))) is not None
        ),
        None,
    )


def _model_name(runs: Sequence[Mapping[str, Any]]) -> str:
    """The model's display name, without any path.

    The server's model root is the most specific name a server reports, so it
    wins over the requested name. Either can be a local path such as
    ``/home/someone/models/x``, so only the last component is kept.
    """
    raw = (
        _first_str(runs, "metadata", "server_model_root")
        or _first_str(runs, "config", "model")
        or next((value for run in runs if (value := _str_or_none(run.get("model")))), None)
        or "unknown model"
    )
    name = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return name or "unknown model"


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def _perf_rows(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The stored throughput cells of a ``perf`` run or of a scored run that carried ``--perf``."""
    scores = _section(run, "scores")
    kind = run.get("run_type") or "tool_eval"
    if kind == "perf":
        results = scores.get("results")
    elif kind == "tool_eval":
        throughput = scores.get("throughput")
        results = throughput.get("results") if isinstance(throughput, Mapping) else None
    else:
        return []
    if not isinstance(results, list):
        return []
    return [row for row in results if isinstance(row, Mapping)]


def _row_depth(row: Mapping[str, Any]) -> int:
    return int(row.get("requested_depth") or row.get("depth") or 0)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return fmean(values) if values else 0.0


def _perf_points(rows: Sequence[Mapping[str, Any]]) -> tuple[PerfPoint, ...]:
    """Average the stored rows of each (depth, concurrency) cell into one point.

    Stored ``tg_tps`` is per request at concurrency 1 and the batch total
    above it. Evidence: ``runner/llama_benchy.py`` ``_parse_benchmark_entry``
    stores llama-benchy's ``tg_throughput`` (the batch total) when concurrency
    is above 1 and ``tg_req_throughput`` otherwise ("Concurrent rows display
    the batch total"), and the built-in ``runner/throughput.py``
    ``measure_concurrent`` also stores total tokens over wall-clock time.
    Newer rows also store the measured per-request rate as ``tg_req_tps``;
    when every decode row in a cell has it, the card uses it. Older rows lack
    it, so the per-request rate falls back to the total divided by the
    concurrency. ``pp_tps`` keeps the stored meaning, which is also the batch
    total above concurrency 1.

    Under llama-benchy's ``--enable-prefix-caching`` a cell has a
    context-load row (``is_context_prefill``) and an inference row that reuses
    the cached prefix. When both exist, the context-load row gives the cold
    ``ttft_ms`` and prefill rate, the inference row gives ``cached_ttft_ms``,
    and decode speed comes from the inference row either way.

    The two TTFTs measure different prompts. At depth above 0 the cold TTFT
    is the context-load row, which prefills only the depth tokens. At depth 0
    there is no context-load row, so the cold TTFT is the standard row, which
    prefills the pp tokens. The cached TTFT is the time to prefill pp new
    tokens on top of an already cached depth prefix. A sweep without prefix
    caching has only standard rows, so its TTFT covers depth plus pp tokens.
    """
    cells: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        concurrency = max(int(row.get("concurrency") or 1), 1)
        cells[(_row_depth(row), concurrency)].append(row)

    points: list[PerfPoint] = []
    for (depth, concurrency), cell_rows in sorted(cells.items()):
        cold = [row for row in cell_rows if row.get("is_context_prefill")]
        warm = [row for row in cell_rows if not row.get("is_context_prefill")]
        decode_rows = warm or cold
        prefill_rows = cold if cold and warm else decode_rows
        tg_total = _mean(decode_rows, "tg_tps")
        measured = all(_is_number(row.get("tg_req_tps")) for row in decode_rows)
        tg_request = _mean(decode_rows, "tg_req_tps") if measured else tg_total / concurrency
        points.append(
            PerfPoint(
                depth=depth,
                concurrency=concurrency,
                pp_tps=_mean(prefill_rows, "pp_tps"),
                tg_tps=tg_request,
                tg_tps_total=tg_total,
                ttft_ms=_mean(prefill_rows, "ttft_ms"),
                cached_ttft_ms=_mean(warm, "ttft_ms") if cold and warm else None,
                pp_estimated=any(row.get("pp_estimated") for row in prefill_rows),
            )
        )
    return tuple(points)


def _perf_workload(run: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """The prompt and generation sizes of the sweep.

    A ``perf`` run stores them in its config. A scored run's config does not,
    so they come from the inference rows: the most common requested prompt and
    configured generation length.
    """
    config = _config(run)
    pp, tg = _int_or_none(config.get("pp")), _int_or_none(config.get("tg"))
    rows = [row for row in _perf_rows(run) if not row.get("is_context_prefill")]
    if pp is None:
        pp = _most_common(row.get("requested_pp") or row.get("pp_tokens") for row in rows)
    if tg is None:
        tg = _most_common(row.get("tg_tokens") for row in rows)
    return pp, tg


def _most_common(values: Iterable[Any]) -> int | None:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        number = _int_or_none(value)
        if number:
            counts[number] += 1
    return max(counts, key=lambda number: (counts[number], number)) if counts else None


# ---------------------------------------------------------------------------
# Tool-calling, retrieval, accuracy
# ---------------------------------------------------------------------------


def _category_name(code: str, stored: Any) -> str:
    try:
        return CATEGORY_LABELS[Category(code)]
    except ValueError:
        return _str_or_none(stored) or code


def _trial_summary(run: Mapping[str, Any]) -> ToolEvalSummary:
    scores = _section(run, "scores")
    categories = [row for row in scores.get("category_scores") or [] if isinstance(row, Mapping)]
    passed = sum(int(row.get("pass_count") or 0) for row in categories)
    partial = sum(int(row.get("partial_count") or 0) for row in categories)
    failed = sum(int(row.get("fail_count") or 0) for row in categories)
    if not passed + partial + failed:
        # Rows that predate per-category counts still list every scenario.
        statuses = [
            row.get("status")
            for row in scores.get("scenario_results") or []
            if isinstance(row, Mapping)
        ]
        passed, partial, failed = (statuses.count(s) for s in ("pass", "partial", "fail"))
    rating = str(scores.get("rating") or "")
    hardmode = any(row.get("category") == Category.P.value for row in categories)
    return ToolEvalSummary(
        score=float(scores.get("final_score") or 0),
        rating=rating,
        passed=passed,
        partial=partial,
        failed=failed,
        total=passed + partial + failed,
        categories=tuple(
            CategoryScore(
                code=str(row.get("category")),
                name=_category_name(str(row.get("category")), row.get("label")),
                percent=float(row.get("percent") or 0.0),
            )
            for row in categories
            if row.get("category")
        ),
        # ``rating_for_score`` marks a capped rating; the cap has no field of its own.
        safety_capped="safety-capped" in rating,
        hardmode=hardmode,
        suite_size=_suite_size(hardmode),
    )


def _suite_size(hardmode: bool) -> int:
    """The public suite a run with or without Category P is measured against."""
    # Imported here: loading every scenario module is only worth it for a card.
    from tool_eval_bench.evals.scenarios import ALL_SCENARIOS, ALL_SCENARIOS_WITH_HARDMODE

    return len(ALL_SCENARIOS_WITH_HARDMODE if hardmode else ALL_SCENARIOS)


def _tool_eval_summary(trials: Sequence[Mapping[str, Any]]) -> ToolEvalSummary:
    """One trial as stored, or the mean of several trials of one invocation.

    Across trials the score and each category percentage are means, the
    outcome counts are the rounded mean per trial, and the rating is
    recomputed from the mean score. The rating is capped if any trial was.
    """
    summaries = [_trial_summary(run) for run in trials]
    if len(summaries) == 1:
        return summaries[0]

    percents: dict[str, list[float]] = defaultdict(list)
    names: dict[str, str] = {}
    for summary in summaries:
        for category in summary.categories:
            names.setdefault(category.code, category.name)
            percents[category.code].append(category.percent)

    def per_trial(count: Callable[[ToolEvalSummary], int]) -> int:
        return round(fmean(count(summary) for summary in summaries))

    score = fmean(summary.score for summary in summaries)
    capped = any(summary.safety_capped for summary in summaries)
    hardmode = any(summary.hardmode for summary in summaries)
    return ToolEvalSummary(
        score=score,
        rating=rating_for_score(round(score), safety_capped=capped),
        passed=per_trial(lambda summary: summary.passed),
        partial=per_trial(lambda summary: summary.partial),
        failed=per_trial(lambda summary: summary.failed),
        total=per_trial(lambda summary: summary.total),
        categories=tuple(
            CategoryScore(code=code, name=name, percent=fmean(percents[code]))
            for code, name in names.items()
        ),
        safety_capped=capped,
        hardmode=hardmode,
        trials=len(summaries),
        suite_size=_suite_size(hardmode),
    )


def _needle_summary(run: Mapping[str, Any]) -> NeedleSummary:
    """The retrieval grid. Runs stored before ``cells`` existed show accuracy without a grid.

    An errored cell is left out of the grid and counted in ``errors``, so a
    context overflow never reads as a miss. Accuracy and effective context
    are the plugin's own figures.
    """
    scores = _section(run, "scores")
    stored = [cell for cell in scores.get("cells") or [] if isinstance(cell, Mapping)]
    cells = tuple(
        NeedleCell(
            context_tokens=int(cell.get("context_tokens") or 0),
            depth_percent=float(cell.get("depth_percent") or 0.0),
            found=bool(cell.get("found")),
        )
        for cell in stored
        if not cell.get("error")
    )
    accuracy = scores.get("accuracy")
    return NeedleSummary(
        cells=cells,
        accuracy=float(accuracy if isinstance(accuracy, (int, float)) else 0.0) / 100,
        effective_context=_int_or_none(scores.get("effective_context")),
        errors=len(stored) - len(cells),
    )


def _accuracy(name: str, run: Mapping[str, Any]) -> AccuracyResult | None:
    scores = _section(run, "scores")
    accuracy = scores.get("accuracy")
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        return None
    total = _int_or_none(scores.get("total"))
    shots = _int_or_none(scores.get("n_shots"))
    if shots is None:
        shots = _int_or_none(_config(run).get("n_shots"))
    parts: list[str] = []
    if total:
        parts.append(f"{total} prompts" if name == "IFEval" else f"{total} q")
    if shots is not None and name != "IFEval":
        parts.append(f"{shots}-shot")
    return AccuracyResult(
        name=name, accuracy=float(accuracy) / 100, detail=", ".join(parts) or None
    )
