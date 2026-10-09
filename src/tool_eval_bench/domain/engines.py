"""What the benchmark knows about each inference engine, in one place.

Identification, spec-decode detection, the spec-live dashboard and context
pressure each need a few facts about the engine behind an endpoint. Those facts
live here as one :class:`EngineProfile` per engine. How an engine is probed
(which endpoints, in what order, how a body is parsed) stays with the probing
code in ``utils.metadata``; this module only says what is true once it is known.

A profile records only facts the code acts on. Adding an engine means adding a
profile and, if it is identified some new way, the probe that finds it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

IdentitySource = Literal["owned_by", "server", "service", "software", "build_info"]
"""Where a server can name itself.

- ``owned_by``: ``/v1/models`` ``data[0].owned_by``
- ``server``: the HTTP ``Server`` header, product token before any ``/version``
- ``service``: ``/health`` ``service``
- ``software``: ``/.well-known/serviceinfo`` ``software.name``
- ``build_info``: ``/props`` ``build_info``; llama.cpp's ``b6123-abc1234`` names no server
"""

UNKNOWN_BACKEND = "unknown"


@dataclass(frozen=True, slots=True)
class EngineProfile:
    """Facts about one engine that more than its own probe depends on."""

    key: str
    """Backend label and first element of an identity tuple, e.g. ``llamacpp``."""
    display_name: str
    """``RunContext.engine_name`` and second element of an identity tuple."""
    aliases: tuple[str, ...] = ()
    """Other backend labels accepted for this engine, lower case."""
    hosted: bool = False
    """A vendor API rather than a self-hosted engine: no engine probes run."""
    metrics_prefixes: tuple[str, ...] = ()
    """Native Prometheus namespaces. They identify the engine only at line start,
    never inside a label value such as ``model_name="acme/strata:7b"``."""
    declared_names: tuple[tuple[IdentitySource, str], ...] = ()
    """Lower-case names the server gives itself, keyed by where it gives them.
    Matched exactly against the value's leading name token."""
    spec_counters_always_rendered: bool = False
    """The ``spec_decode`` counters appear on every scrape, drafting or not, so
    speculation is active only once drafted tokens are above zero."""
    absent_draft_n_is_zero: bool = False
    """A response ``timings`` object without ``draft_n`` means this request
    drafted nothing. Elsewhere a missing field means "not reported". Sets
    ``SpecDecodeInfo.has_per_request_timings``, which ``speculative`` and
    ``throughput`` also read as "llama.cpp detected" for their log hints."""
    fixed_spec_method: str | None = None
    """The only proposer the engine has. Spec detection and spec-live report it
    when no explicit method label names one; a label wins."""
    spec_counter_detail: str | None = None
    """``SpecDecodeInfo.detail`` when this engine's spec counters are found."""
    reports_request_context_window: bool = False
    """The ``max_model_len`` the run-context probe records is verified upstream
    to be the longest single request, so context pressure can size from it."""

    @property
    def identity(self) -> tuple[str, str]:
        """``(key, display_name)``, the shape backend detection returns."""
        return self.key, self.display_name

    @property
    def labels(self) -> tuple[str, ...]:
        """Every backend label that selects this engine."""
        return self.key, *self.aliases

    def declared_name(self, source: IdentitySource) -> str | None:
        """The name this engine gives itself at *source*, if it gives one there."""
        return next((token for s, token in self.declared_names if s == source), None)


VLLM = EngineProfile(
    key="vllm",
    display_name="vLLM",
    metrics_prefixes=("vllm:",),
    declared_names=(("server", "vllm"),),
)
LITELLM = EngineProfile(
    key="litellm",
    display_name="LiteLLM",
    declared_names=(("server", "litellm"),),
)
LLAMACPP = EngineProfile(
    key="llamacpp",
    display_name="llama.cpp",
    aliases=("llama.cpp", "llama_cpp"),
    metrics_prefixes=("llamacpp:",),
    declared_names=(("owned_by", "llamacpp"), ("server", "llama.cpp")),
    spec_counters_always_rendered=True,
    # server_slot_stats::to_json (tools/server/server-common.cpp) writes
    # draft_n only when n_draft_tokens > 0.
    absent_draft_n_is_zero=True,
    spec_counter_detail=(
        "llama.cpp draft counters available; per-request counts "
        "come from response timings (draft_n/draft_n_accepted)"
    ),
    # /props default_generation_settings.n_ctx is the per-slot context, the
    # longest single request. See context_pressure.reported_context_window.
    reports_request_context_window=True,
)
SGLANG = EngineProfile(
    key="sglang",
    display_name="SGLang",
    # SGLang >=0.5.4 renamed the metric prefix.
    metrics_prefixes=("sglang:", "sglang_"),
    declared_names=(("owned_by", "sglang"), ("server", "sglang")),
)
GEMINI = EngineProfile(key="gemini", display_name="Google Gemini API", hosted=True)
OPENAI = EngineProfile(key="openai", display_name="OpenAI API", hosted=True)
ANTHROPIC = EngineProfile(key="anthropic", display_name="Anthropic Messages API", hosted=True)
NINFER = EngineProfile(
    key="ninfer",
    display_name="NInfer",
    declared_names=(("owned_by", "ninfer"),),
)
TENSORFOLD = EngineProfile(
    key="tensorfold",
    display_name="TensorFold",
    metrics_prefixes=("tensorfold:",),
    declared_names=(("owned_by", "tensorfold"),),
    # The mtp_* aliases count every proposer, not just MTP, so the method
    # stays unknown.
    spec_counters_always_rendered=True,
    spec_counter_detail="TensorFold draft counters available; proposer configuration unknown",
)
HALOGEN = EngineProfile(
    key="halogen",
    display_name="Halogen Flash",
    metrics_prefixes=("halogen:",),
)
STRATA = EngineProfile(
    key="strata",
    display_name="Strata",
    # Strata's Prometheus format uses vLLM's metric names plus this namespace,
    # which every scrape carries.
    metrics_prefixes=("strata:",),
    declared_names=(("service", "strata"), ("build_info", "strata")),
    spec_counters_always_rendered=True,
    # Its drafts come from the model's MTP head. Strata reports draft_n in
    # timings but may omit it for "not reported", so absence is not zero.
    fixed_spec_method="mtp",
    spec_counter_detail="Strata MTP draft counters available",
    # /props n_ctx or /health max_context; serve/server.py never lets one
    # request's prompt plus completion plus an 8-token margin exceed it.
    reports_request_context_window=True,
)
TABBYAPI = EngineProfile(
    key="tabbyapi",
    display_name="TabbyAPI",
    declared_names=(("owned_by", "tabbyapi"), ("software", "tabbyapi")),
)

# Public label order, as the argument schema lists it.
ENGINE_PROFILES: tuple[EngineProfile, ...] = (
    VLLM,
    LITELLM,
    LLAMACPP,
    SGLANG,
    GEMINI,
    OPENAI,
    ANTHROPIC,
    NINFER,
    TENSORFOLD,
    HALOGEN,
    STRATA,
    TABBYAPI,
)

# Which engine a /metrics scrape belongs to when several namespaces appear.
# An engine's own namespace beats the compatibility names it also exports:
# Halogen deliberately exports llama.cpp metrics and Strata uses vLLM's names,
# so scrape order must not decide identity.
METRICS_IDENTITY_ORDER: tuple[EngineProfile, ...] = (
    HALOGEN,
    TENSORFOLD,
    STRATA,
    VLLM,
    SGLANG,
    LLAMACPP,
)

# Which engine's spec-counter rules apply when spec detection finds
# ``spec_decode`` counters. This is not METRICS_IDENTITY_ORDER: llama.cpp is
# checked before Strata here, so a scrape carrying both namespaces is
# identified as Strata but read with llama.cpp's counter rules, and Halogen's
# exported llama.cpp counters are read as llama.cpp's.
SPEC_COUNTER_ORDER: tuple[EngineProfile, ...] = (TENSORFOLD, LLAMACPP, STRATA)

# The backend labels the argument schema offers, in order.
BACKEND_LABELS: tuple[str, ...] = (*(p.key for p in ENGINE_PROFILES), UNKNOWN_BACKEND)

# Every label a caller may pass, including aliases.
ACCEPTED_BACKEND_LABELS: frozenset[str] = frozenset(
    {label for p in ENGINE_PROFILES for label in p.labels} | {UNKNOWN_BACKEND}
)

_BY_LABEL = {label: p for p in ENGINE_PROFILES for label in p.labels}
_BY_DISPLAY_NAME = {p.display_name: p for p in ENGINE_PROFILES}


def engine_profile(label: str | None) -> EngineProfile | None:
    """Return the profile a backend label selects, ignoring case."""
    return _BY_LABEL.get(label.lower()) if label else None


def engine_profile_by_name(display_name: str | None) -> EngineProfile | None:
    """Return the profile whose ``display_name`` is exactly *display_name*."""
    return _BY_DISPLAY_NAME.get(display_name) if display_name else None


def metrics_namespace_present(text: str, profile: EngineProfile) -> bool:
    """Whether a Prometheus line starts with one of *profile*'s namespaces."""
    return any(
        re.search(f"^{re.escape(prefix)}", text, re.MULTILINE) is not None
        for prefix in profile.metrics_prefixes
    )
