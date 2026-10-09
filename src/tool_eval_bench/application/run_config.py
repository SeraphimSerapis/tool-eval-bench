"""The inputs that define a run's identity, and the persisted config built from them.

`BenchmarkService.run_benchmark` used to pass the same seventeen arguments to
`_build_run_config` twice, once before the run and once after merging resumed
results, differing only in the scenario list.  `RunSettings` captures the
parameters that do not change between those two calls so the second is a
one-liner.

Every persisted key is declared once, in :data:`RUN_CONFIG_FIELDS`, with the
rules that used to live in three places that had to agree: how the stored
config is built and fingerprinted, how ``--resume`` compares it with the
current flags, and which keys the leaderboard ignores when it groups different
models into one cohort.  Adding a key means adding a row, and the row has no
defaults, so it cannot be added without deciding each rule.

This is an internal composition helper.  The service's own keyword signature is
the published API and is unchanged; nothing here appears in it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tool_eval_bench.adapters.wire_format import detect_wire_format
from tool_eval_bench.domain.scenarios import ScenarioDefinition
from tool_eval_bench.utils.fingerprint import comparison_fingerprint
from tool_eval_bench.utils.urls import (
    canonical_endpoint_path,
    endpoint_identity,
    legacy_endpoint_identities,
)
from tool_eval_bench.utils.urls import redact_url as _redact_url

#: The key :func:`build_run_config` stores the fingerprint under.
CONFIG_FINGERPRINT_KEY = "config_fingerprint"


@dataclass(frozen=True)
class RunSettings:
    """Run parameters that determine the persisted config and its fingerprint.

    Frozen because two calls to :func:`build_run_config` within one run must see
    exactly the same values; a fingerprint that shifted mid-run would put the
    resumed half of a run in a different cohort from the first half.
    """

    model: str
    backend: str
    base_url: str
    temperature: float
    timeout_seconds: float
    max_turns: int
    seed: int | None
    reference_date: str | None
    concurrency: int
    error_rate: float
    alpha: float
    extra_params: dict[str, Any] | None
    context_pressure_config: dict[str, Any] | None
    weight_by_difficulty: bool
    system_prompt: str | None = None
    decision_judge: dict[str, Any] | None = None


@dataclass(frozen=True)
class _Inputs:
    settings: RunSettings
    scenarios: list[ScenarioDefinition]
    scenario_packs: list[dict[str, Any]] | None


class Presence(Enum):
    """When a key is written to the stored config."""

    ALWAYS = "always"
    #: Optional keys added after runs were already stored.  Writing them only
    #: when set keeps a default run's config, and so its cohort, unchanged.
    WHEN_NOT_NONE = "when_not_none"
    WHEN_NONEMPTY = "when_nonempty"

    def keeps(self, value: Any) -> bool:
        if self is Presence.WHEN_NOT_NONE:
            return value is not None
        if self is Presence.WHEN_NONEMPTY:
            return bool(value)
        return True


class Fingerprint(Enum):
    """How a key contributes to ``config_fingerprint``."""

    INCLUDE = "include"
    #: Order-insensitive: the same scenarios in another order are one cohort.
    SORTED = "sorted"
    #: Stored for resume, but never splits a cohort.
    EXCLUDE = "exclude"


class Absent(Enum):
    """What a key missing from a stored run means to the resume check."""

    #: The run predates the key.  Validate only what it recorded.
    SKIP = "skip"
    #: The key is written only when set, so absence means the unset value.
    UNSET = "unset"


@dataclass(frozen=True)
class ResumeCheck:
    """How ``--resume`` compares a stored key with the current run's.

    ``describe(stored, current)`` returns a mismatch message, or None when the
    two are compatible.  Without it, any inequality reports the bare key.
    """

    absent: Absent
    unset: Any = None
    describe: Callable[[Any, Any], str | None] | None = None


@dataclass(frozen=True)
class ConfigField:
    """One persisted config key and every rule that applies to it.

    The classification attributes deliberately have no defaults.
    """

    key: str
    value: Callable[[_Inputs], Any]
    presence: Presence
    fingerprint: Fingerprint
    #: Names the model or endpoint under test.  The leaderboard cohort compares
    #: different models under the same conditions, so it drops these keys.
    identifies_model: bool
    #: None when resume does not compare the key.
    resume: ResumeCheck | None
    #: Projects the stored value to what an ``INCLUDE`` fingerprint hashes.
    #: None hashes the value exactly as stored.
    fingerprint_view: Callable[[Any], Any] | None = None


def _pressure_mismatch(previous: Any, current: dict[str, Any] | None) -> str | None:
    """Name a context-pressure difference that would merge two fill levels into one run.

    A run without pressure persists no ``context_pressure`` key; the key has been
    written since the option existed, so absence always means "no pressure".

    The ratio and the fill target decide what the model sees. The detected
    ``context_size`` is not compared: a restarted server can report a slightly
    different KV capacity, and the fill target is quantised to whole filler
    chunks, so drift that leaves the fill unchanged must not block a resume.
    The calibrated ``fill_tokens`` is not compared either, because an unseeded
    run draws fresh filler and lands within the calibration tolerance, never
    on the same count twice.
    """
    if not previous and not current:
        return None
    if not previous or not current:
        was = previous["ratio"] if previous else "off"
        now = current["ratio"] if current else "off"
        return f"context_pressure (was {was}, now {now})"
    if previous.get("ratio") != current["ratio"]:
        return f"context_pressure ratio (was {previous.get('ratio')}, now {current['ratio']})"
    # Runs from before tokenizer calibration recorded no target.
    old_target = previous.get("fill_tokens_target")
    if old_target is not None and old_target != current["fill_tokens_target"]:
        return (
            f"context_pressure fill (was {old_target:,} tokens, "
            f"now {current['fill_tokens_target']:,}; the context size changed)"
        )
    return None


def _pressure_fingerprint(value: dict[str, Any]) -> dict[str, Any]:
    """The context-pressure block without its calibrated ``fill_tokens``.

    The ratio, fill target, and window decide what the model sees. The
    calibrated count differs on every unseeded run (see
    :func:`_pressure_mismatch`), so hashing it would put two otherwise identical
    pressure runs in separate cohorts.
    """
    return {key: item for key, item in value.items() if key != "fill_tokens"}


def _canonical_judge(config: Any) -> Any:
    """A judge config with its URL redacted and reduced to its request root."""
    if isinstance(config, dict) and isinstance(config.get("base_url"), str):
        return {**config, "base_url": canonical_endpoint_path(_redact_url(config["base_url"]))}
    return config


def _judge_mismatch(previous: Any, current: Any) -> str | None:
    """Name the judge difference a user must undo to resume, when it is a set or check."""
    # Runs audited before judge URLs were redacted stored the raw URL, and any
    # spelling of the same judge (with or without /v1 or a trailing slash)
    # sends requests to the same place. Compare the redacted canonical form on
    # both sides; endpoint_id still tells two hosts apart.
    previous, current = _canonical_judge(previous), _canonical_judge(current)
    if previous == current:
        return None
    if isinstance(previous, dict) and "set" not in previous:
        # Audited before judge sets existed, which meant TC-89 only. No set
        # reproduces that selection, so say why rather than name a bare key.
        return "decision_judge (a TC-89-only audit from an earlier version)"
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return "decision_judge"
    if previous["set"] != current["set"]:
        return f"decision_judge set (was {previous['set']}, now {current['set']})"
    old_checks = set(previous.get("checks") or [])
    new_checks = set(current.get("checks") or [])
    if old_checks == new_checks:
        # A URL or model change; the generic key already says enough.
        return "decision_judge"
    changes = []
    if removed := sorted(old_checks - new_checks):
        changes.append(f"was {', '.join(removed)}")
    if added := sorted(new_checks - old_checks):
        changes.append(f"now {', '.join(added)}")
    return f"decision_judge checks ({'; '.join(changes)})"


def _canonical_base_url(url: Any) -> Any:
    """Canonicalise a stored, redacted base URL the way endpoint identities are."""
    if not isinstance(url, str):
        return url
    return canonical_endpoint_path(url, wire_format=detect_wire_format(url))


def _base_url_mismatch(previous: Any, current: Any) -> str | None:
    """Ignore spellings of one base URL that build the same requests.

    A redacted URL hides the host, so a Gemini URL is canonicalised like an
    OpenAI one here. ``endpoint_id`` is compared too and keeps those apart.
    """
    if _canonical_base_url(previous) == _canonical_base_url(current):
        return None
    return "base_url"


# Older stored runs predate some always-written keys, so resume validates only
# the ones a run recorded.
_LEGACY = ResumeCheck(Absent.SKIP)

#: The persisted config schema, in stored key order.
RUN_CONFIG_FIELDS: tuple[ConfigField, ...] = (
    ConfigField(
        "model",
        lambda i: i.settings.model,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=True,
        resume=_LEGACY,
    ),
    ConfigField(
        "backend",
        lambda i: i.settings.backend,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    # endpoint_id already tells endpoints apart, and this redacted string keeps
    # spelling differences (a trailing slash, /v1) that the requests do not.
    ConfigField(
        "base_url",
        lambda i: _redact_url(i.settings.base_url),
        Presence.ALWAYS,
        Fingerprint.EXCLUDE,
        identifies_model=True,
        resume=ResumeCheck(Absent.SKIP, describe=_base_url_mismatch),
    ),
    # The wire format comes from the URL, as auto-detection resolves it. The
    # settings do not carry --format, so a native Gemini proxy on another host
    # is canonicalised like an OpenAI endpoint.
    ConfigField(
        "endpoint_id",
        lambda i: endpoint_identity(
            i.settings.base_url, wire_format=detect_wire_format(i.settings.base_url)
        ),
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=True,
        resume=_LEGACY,
    ),
    ConfigField(
        "temperature",
        lambda i: i.settings.temperature,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "timeout_seconds",
        lambda i: i.settings.timeout_seconds,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "max_turns",
        lambda i: i.settings.max_turns,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "seed",
        lambda i: i.settings.seed,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "reference_date",
        lambda i: i.settings.reference_date,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    # Derived from scenario_ids, which resume already compares.
    ConfigField(
        "scenario_count",
        lambda i: len(i.scenarios),
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=None,
    ),
    # Resume compares the run order; the fingerprint does not.
    ConfigField(
        "scenario_ids",
        lambda i: [s.id for s in i.scenarios],
        Presence.ALWAYS,
        Fingerprint.SORTED,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "concurrency",
        lambda i: i.settings.concurrency,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "error_rate",
        lambda i: i.settings.error_rate,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "alpha",
        lambda i: i.settings.alpha,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "extra_params",
        lambda i: i.settings.extra_params,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    ConfigField(
        "weight_by_difficulty",
        lambda i: i.settings.weight_by_difficulty,
        Presence.ALWAYS,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=_LEGACY,
    ),
    # Absent means the built-in prompt: resuming it with an override would
    # otherwise merge two personas into one result.
    ConfigField(
        "system_prompt",
        lambda i: i.settings.system_prompt,
        Presence.WHEN_NOT_NONE,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=ResumeCheck(Absent.UNSET),
    ),
    # Audits never change official scores, so the judge stays out of the
    # fingerprint: judged and unjudged runs of one configuration are one cohort.
    # The stored config keeps it for the resume compatibility check.
    ConfigField(
        "decision_judge",
        lambda i: i.settings.decision_judge,
        Presence.WHEN_NOT_NONE,
        Fingerprint.EXCLUDE,
        identifies_model=False,
        resume=ResumeCheck(Absent.UNSET, describe=_judge_mismatch),
    ),
    ConfigField(
        "scenario_variants",
        lambda i: {s.id: s.variant_metadata for s in i.scenarios if s.variant_metadata},
        Presence.WHEN_NONEMPTY,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=ResumeCheck(Absent.UNSET, unset={}),
    ),
    ConfigField(
        "context_pressure",
        lambda i: i.settings.context_pressure_config,
        Presence.WHEN_NONEMPTY,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=ResumeCheck(Absent.UNSET, describe=_pressure_mismatch),
        fingerprint_view=_pressure_fingerprint,
    ),
    # Written only when set, so a missing key means the run had no packs.
    # Resuming such a run with a pack would score held-out scenarios under the
    # public run's identity.
    ConfigField(
        "scenario_packs",
        lambda i: i.scenario_packs,
        Presence.WHEN_NONEMPTY,
        Fingerprint.INCLUDE,
        identifies_model=False,
        resume=ResumeCheck(Absent.UNSET),
    ),
)

#: Keys the leaderboard drops to rank different models under the same
#: conditions: the ones naming the model or endpoint, the ones that never split
#: a cohort, and the per-model fingerprint itself.
COHORT_EXCLUDED_KEYS = frozenset(
    {
        CONFIG_FINGERPRINT_KEY,
        *(
            field.key
            for field in RUN_CONFIG_FIELDS
            if field.identifies_model or field.fingerprint is Fingerprint.EXCLUDE
        ),
    }
)

_FINGERPRINT_VIEWS = {
    field.key: field.fingerprint_view
    for field in RUN_CONFIG_FIELDS
    if field.fingerprint_view is not None
}


def cohort_config(config: dict[str, Any]) -> dict[str, Any]:
    """The part of a stored config that groups different models into one cohort.

    Drops :data:`COHORT_EXCLUDED_KEYS` and projects each field that declares a
    ``fingerprint_view``, so the cohort ignores what the fingerprint ignores.
    Keys only an older version stored are kept, so they still split cohorts.
    """
    comparable: dict[str, Any] = {}
    for key, value in config.items():
        if key in COHORT_EXCLUDED_KEYS:
            continue
        view = _FINGERPRINT_VIEWS.get(key)
        comparable[key] = view(value) if view is not None and isinstance(value, dict) else value
    return comparable


def build_run_config(
    settings: RunSettings,
    *,
    scenarios: list[ScenarioDefinition],
    metadata: dict[str, Any],
    scenario_packs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the persisted config and its deterministic comparison fingerprint.

    The fingerprint decides which runs the leaderboard groups together, so the
    key set and value shapes here are a stored contract: changing them
    re-cohorts every historical run.
    """
    inputs = _Inputs(settings, scenarios, scenario_packs)
    config: dict[str, Any] = {}
    fingerprint_config: dict[str, Any] = {}
    for field in RUN_CONFIG_FIELDS:
        value = field.value(inputs)
        if not field.presence.keeps(value):
            continue
        config[field.key] = value
        if field.fingerprint is Fingerprint.INCLUDE:
            view = field.fingerprint_view
            fingerprint_config[field.key] = view(value) if view is not None else value
        elif field.fingerprint is Fingerprint.SORTED:
            fingerprint_config[field.key] = sorted(value)
    config[CONFIG_FINGERPRINT_KEY] = comparison_fingerprint(fingerprint_config, metadata)
    return config


def resume_mismatches(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    base_url: str | None = None,
    judge_base_url: str | None = None,
) -> list[str]:
    """Name every scoring condition in which a stored run differs from the current one.

    Both arguments are configs from :func:`build_run_config` (``previous`` may
    come from an older version).  Messages follow stored key order.
    ``base_url`` is the current run's unredacted URL. With it, a run stored
    before endpoint identities were canonical still resumes under any spelling
    of the same endpoint. ``judge_base_url`` is the current decision judge's
    unredacted URL and does the same for the judge.
    """
    previous = _with_current_endpoint_ids(previous, current, base_url, judge_base_url)
    mismatches: list[str] = []
    for field in RUN_CONFIG_FIELDS:
        check = field.resume
        if check is None:
            continue
        if check.absent is Absent.SKIP and field.key not in previous:
            continue
        old = previous.get(field.key, check.unset)
        new = current.get(field.key, check.unset)
        if check.describe is not None:
            message = check.describe(old, new)
        else:
            message = field.key if old != new else None
        if message is not None:
            mismatches.append(message)
    return mismatches


def _with_current_endpoint_ids(
    previous: dict[str, Any],
    current: dict[str, Any],
    base_url: str | None,
    judge_base_url: str | None,
) -> dict[str, Any]:
    """Treat an identity the old algorithm computed for this endpoint as the current one.

    Identities used to keep ``/v1`` in the path. A run started before that
    changed must still resume under any spelling of the same server, and the
    stored hash cannot be recomputed, so every legacy identity a spelling of
    the current URL could have produced is accepted instead. The judge speaks
    the OpenAI format, and its stored config holds only a redacted URL, so the
    caller passes the judge's raw URL as ``judge_base_url``.
    """
    updated = dict(previous)
    stored = previous.get("endpoint_id")
    if base_url is not None and stored in legacy_endpoint_identities(
        base_url, wire_format=detect_wire_format(base_url)
    ):
        updated["endpoint_id"] = current.get("endpoint_id")
    old_judge, new_judge = previous.get("decision_judge"), current.get("decision_judge")
    if (
        isinstance(old_judge, dict)
        and isinstance(new_judge, dict)
        and judge_base_url is not None
        and old_judge.get("endpoint_id") in legacy_endpoint_identities(judge_base_url)
    ):
        updated["decision_judge"] = {**old_judge, "endpoint_id": new_judge.get("endpoint_id")}
    return updated
