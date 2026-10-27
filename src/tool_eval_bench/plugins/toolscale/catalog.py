"""Tool catalog for the toolscale sweep: loading, validation, and lint.

A catalog is a three-level tree: domains hold families, families hold variants.
The structure defines the similarity tiers, so it is the benchmark's ground
truth rather than a convenience:

- the variants of one family do the same job and differ by one qualifier, so
  they are near-duplicates of each other;
- families in one domain share vocabulary but do different jobs;
- tools in different domains are unrelated.

Every variant has a literal ``query`` that names its qualifier, and may have a
``query_implied`` that only implies it ("prepared but not sent yet" for a
draft). The literal query is the clean control; the implied one tests whether
a model understands the tools or only matches keywords.

Each family may also carry a ``description_terse``: one generic sentence that
is true of all its variants. The ``terse`` description level shows it in place
of every variant's own description, so near-duplicates differ only by name.

The built-in catalog ships as ``catalog/v1.json``. A lab can run a private
catalog with the same schema through ``--toolscale-catalog PATH``, which keeps
the items out of public training data.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

BUILTIN_CATALOG_VERSION = "1"

# The grid draws three neighbours from one tier, so a family needs three
# siblings and a domain needs three other families.
MIN_VARIANTS_PER_FAMILY = 4
MIN_FAMILIES_PER_DOMAIN = 4

# OpenAI's function-name rule, the strictest of the supported wire formats.
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PARAM_TYPES = frozenset({"string", "integer", "number", "boolean"})
# Parameter-schema keys kept at the terse and bare description levels.
_DEGRADED_SCHEMA_KEYS = frozenset({"type", "enum"})


class CatalogError(ValueError):
    """The catalog file is missing, unreadable, or does not match the schema."""


@dataclass(frozen=True, slots=True)
class CatalogTool:
    """One tool variant and the query whose only correct answer it is."""

    name: str
    domain: str
    family: str
    qualifier: str
    description: str
    parameters: Mapping[str, Any]
    required: tuple[str, ...]
    query: str
    query_implied: str | None = None
    family_description: str | None = None

    def query_for(self, cue: str) -> str:
        """The user message for *cue*: ``"literal"`` or ``"implied"``."""
        if cue == "implied":
            if self.query_implied is None:
                raise KeyError(f"{self.name} has no implied query")
            return self.query_implied
        return self.query

    def to_openai_tool(self, descriptions: str = "clean") -> dict[str, Any]:
        """The tool as an OpenAI function definition, the adapters' common input.

        *descriptions* degrades what the model reads: ``"clean"`` is the
        catalog as written; ``"terse"`` swaps in the family's shared
        description and drops parameter descriptions; ``"bare"`` drops every
        description, leaving names, types, and enums.
        """
        properties = json.loads(json.dumps(self.parameters))
        if descriptions != "clean":
            # Keep only what a schema needs to be callable. A private catalog
            # may carry titles or examples, which would leak the very text the
            # degraded levels exist to remove.
            properties = {
                name: {k: v for k, v in schema.items() if k in _DEGRADED_SCHEMA_KEYS}
                for name, schema in properties.items()
            }
        function: dict[str, Any] = {"name": self.name}
        if descriptions == "clean":
            function["description"] = self.description
        elif descriptions == "terse":
            if self.family_description is None:
                raise KeyError("this tool's family has no description_terse")
            function["description"] = self.family_description
        elif descriptions != "bare":
            raise ValueError(f"unknown description level {descriptions!r}")
        function["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": list(self.required),
            "additionalProperties": False,
        }
        return {"type": "function", "function": function}


@dataclass(frozen=True, slots=True)
class Family:
    """Near-duplicate variants of one job."""

    id: str
    domain: str
    variants: tuple[CatalogTool, ...]
    description_terse: str | None = None


@dataclass(frozen=True, slots=True)
class Domain:
    """Families that share a subject area."""

    id: str
    families: tuple[Family, ...]


@dataclass(frozen=True, slots=True)
class Catalog:
    """A validated catalog and the identity that pins it in a run's config."""

    version: str
    source: str  # "builtin" or "custom"
    sha256: str
    domains: tuple[Domain, ...]

    @property
    def is_custom(self) -> bool:
        """Whether the catalog came from ``--toolscale-catalog`` rather than the package."""
        return self.source == "custom"

    @property
    def families(self) -> tuple[Family, ...]:
        """Every family, domain by domain."""
        return tuple(family for domain in self.domains for family in domain.families)

    @property
    def tools(self) -> tuple[CatalogTool, ...]:
        """Every tool, family by family."""
        return tuple(tool for family in self.families for tool in family.variants)

    def family_of(self, tool: CatalogTool) -> Family:
        """The family *tool* belongs to."""
        for family in self.families:
            if family.domain == tool.domain and family.id == tool.family:
                return family
        raise KeyError(tool.name)

    def domain_of(self, tool: CatalogTool) -> Domain:
        """The domain *tool* belongs to."""
        for domain in self.domains:
            if domain.id == tool.domain:
                return domain
        raise KeyError(tool.name)


def load_builtin_catalog() -> Catalog:
    """Load the catalog shipped with the package."""
    ref = resources.files("tool_eval_bench.plugins.toolscale").joinpath(
        f"catalog/v{BUILTIN_CATALOG_VERSION}.json"
    )
    try:
        raw = ref.read_bytes()
    except OSError as exc:
        raise CatalogError(f"built-in toolscale catalog is missing: {exc}") from exc
    return parse_catalog(raw, source="builtin")


def load_catalog_file(path: str | Path) -> Catalog:
    """Load a private catalog with the built-in schema."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise CatalogError(f"cannot read toolscale catalog {path}: {exc}") from exc
    return parse_catalog(raw, source="custom")


def parse_catalog(raw: bytes, *, source: str) -> Catalog:
    """Validate *raw* JSON and build a :class:`Catalog`.

    Validation covers what the grid needs to run and what every backend
    accepts. Authoring quality, such as whether a query names its qualifier,
    is :func:`lint_catalog`'s job, because a private catalog may break those
    conventions on purpose.
    """
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"toolscale catalog is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError("toolscale catalog must be a JSON object")

    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise CatalogError("toolscale catalog needs a non-empty string 'version'")

    raw_domains = _list(data, "domains", "catalog")
    if len(raw_domains) < 2:
        raise CatalogError("toolscale catalog needs at least two domains")

    seen_names: set[str] = set()
    seen_domains: set[str] = set()
    domains: list[Domain] = []
    for raw_domain in raw_domains:
        domain_id = _ident(raw_domain, "domain")
        if domain_id in seen_domains:
            raise CatalogError(f"duplicate domain id {domain_id!r}")
        seen_domains.add(domain_id)

        raw_families = _list(raw_domain, "families", f"domain {domain_id!r}")
        if len(raw_families) < MIN_FAMILIES_PER_DOMAIN:
            raise CatalogError(
                f"domain {domain_id!r} has {len(raw_families)} families; "
                f"at least {MIN_FAMILIES_PER_DOMAIN} are needed"
            )
        families: list[Family] = []
        seen_families: set[str] = set()
        for raw_family in raw_families:
            family_id = _ident(raw_family, f"family in domain {domain_id!r}")
            if family_id in seen_families:
                raise CatalogError(f"duplicate family id {family_id!r} in domain {domain_id!r}")
            seen_families.add(family_id)
            where = f"family {domain_id}/{family_id}"
            raw_variants = _list(raw_family, "variants", where)
            if len(raw_variants) < MIN_VARIANTS_PER_FAMILY:
                raise CatalogError(
                    f"{where} has {len(raw_variants)} variants; "
                    f"at least {MIN_VARIANTS_PER_FAMILY} are needed"
                )
            terse = (
                _string(raw_family, "description_terse", where)
                if raw_family.get("description_terse") is not None
                else None
            )
            variants = tuple(
                replace(
                    _parse_tool(raw_variant, domain_id, family_id, seen_names),
                    family_description=terse,
                )
                for raw_variant in raw_variants
            )
            families.append(
                Family(
                    id=family_id,
                    domain=domain_id,
                    variants=variants,
                    description_terse=terse,
                )
            )
        domains.append(Domain(id=domain_id, families=tuple(families)))

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return Catalog(
        version=version,
        source=source,
        sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        domains=tuple(domains),
    )


def _parse_tool(raw: Any, domain: str, family: str, seen_names: set[str]) -> CatalogTool:
    if not isinstance(raw, dict):
        raise CatalogError(f"a variant in {domain}/{family} is not an object")
    name = _string(raw, "name", f"variant in {domain}/{family}")
    where = f"tool {name!r}"
    if not _NAME_RE.match(name):
        raise CatalogError(f"{where}: name must match {_NAME_RE.pattern}")
    if name in seen_names:
        raise CatalogError(f"duplicate tool name {name!r}")
    seen_names.add(name)

    parameters = raw.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise CatalogError(f"{where}: 'parameters' must be a non-empty object")
    for param, schema in parameters.items():
        if not isinstance(schema, dict) or schema.get("type") not in _PARAM_TYPES:
            raise CatalogError(
                f"{where}: parameter {param!r} needs a 'type' of {sorted(_PARAM_TYPES)}"
            )
    required = raw.get("required")
    if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
        raise CatalogError(f"{where}: 'required' must be a list of parameter names")
    unknown = [r for r in required if r not in parameters]
    if unknown:
        raise CatalogError(f"{where}: required parameters {unknown} are not declared")

    return CatalogTool(
        name=name,
        domain=domain,
        family=family,
        qualifier=_string(raw, "qualifier", where),
        description=_string(raw, "description", where),
        parameters=parameters,
        required=tuple(required),
        query=_string(raw, "query", where),
        query_implied=(
            _string(raw, "query_implied", where) if raw.get("query_implied") is not None else None
        ),
    )


def _list(container: Any, key: str, where: str) -> list[Any]:
    value = container.get(key) if isinstance(container, dict) else None
    if not isinstance(value, list):
        raise CatalogError(f"{where}: {key!r} must be a list")
    return value


def _string(container: dict[str, Any], key: str, where: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where}: {key!r} must be a non-empty string")
    return value.strip()


def _ident(container: Any, where: str) -> str:
    if not isinstance(container, dict):
        raise CatalogError(f"{where} is not an object")
    return _string(container, "id", where)


# ---------------------------------------------------------------------------
# Authoring lint
# ---------------------------------------------------------------------------


def _phrase_in(phrase: str, text: str) -> bool:
    """Whether *phrase* occurs in *text* as whole words, ignoring case."""
    words = [re.escape(word) for word in phrase.lower().split()]
    if not words:
        return False
    return re.search(r"\b" + r"\s+".join(words) + r"\b", text.lower()) is not None


def _strip_phrase(phrase: str, text: str) -> str:
    """*text* with every whole-word occurrence of *phrase* removed, ignoring case."""
    words = [re.escape(word) for word in phrase.lower().split()]
    if not words:
        return text
    return re.sub(r"\b" + r"\s+".join(words) + r"\b", " ", text, flags=re.IGNORECASE)


# Words too common to count as giving a qualifier away on their own. The whole
# qualifier phrase is always checked; these only exempt single words of a
# multi-word qualifier ("same day" may not appear, "day" may).
_GENERIC_QUALIFIER_WORDS = frozenset(
    "a an and the of to in on by for per with only day days through out up off".split()
)


def _lint_implied(tool: CatalogTool, family: Family, all_names: list[str]) -> list[str]:
    """Problems with *tool*'s implied query: it must not give the answer away."""
    query = tool.query_implied
    if query is None:
        return []
    problems: list[str] = []
    if _phrase_in(tool.qualifier, query):
        problems.append(f"{tool.name}: implied query names its qualifier {tool.qualifier!r}")
    else:
        for word in tool.qualifier.lower().split():
            if word not in _GENERIC_QUALIFIER_WORDS and _phrase_in(word, query):
                problems.append(f"{tool.name}: implied query contains qualifier word {word!r}")
    for sibling in family.variants:
        if sibling is not tool and _phrase_in(sibling.qualifier, query):
            problems.append(
                f"{tool.name}: implied query names sibling qualifier {sibling.qualifier!r}"
            )
    lowered = query.lower()
    leaked = [name for name in all_names if name.lower() in lowered]
    if leaked:
        problems.append(f"{tool.name}: implied query contains tool name(s) {leaked}")
    return problems


def lint_catalog(catalog: Catalog) -> list[str]:
    """Return authoring problems that would make an item ambiguous or give it away.

    Each literal query must name its own variant's qualifier and none of its
    siblings', every variant name must carry its qualifier (no generic default
    that also matches a sibling's query), and no query may contain a tool name.
    An implied query must name neither its own qualifier nor a sibling's, so
    the answer has to be inferred. These checks keep the near tier measuring
    confusion instead of ambiguity. An empty list means the catalog is clean.
    """
    problems: list[str] = []
    all_names = [tool.name for tool in catalog.tools]
    for family in catalog.families:
        qualifiers = [tool.qualifier for tool in family.variants]
        if len(set(qualifiers)) != len(qualifiers):
            problems.append(f"{family.domain}/{family.id}: qualifiers repeat")
        for tool in family.variants:
            token = "_".join(tool.qualifier.lower().split())
            if token not in tool.name:
                problems.append(f"{tool.name}: name does not carry qualifier {tool.qualifier!r}")
            if not _phrase_in(tool.qualifier, tool.query):
                problems.append(f"{tool.name}: query does not name qualifier {tool.qualifier!r}")
            # A qualifier may contain a sibling's ("premium economy" next to
            # "economy"). Strip the tool's own qualifier first so only a
            # sibling named on its own counts.
            own_query = _strip_phrase(tool.qualifier, tool.query)
            own_name = tool.name.replace(token, "#")
            for sibling in family.variants:
                if sibling is tool:
                    continue
                if _phrase_in(sibling.qualifier, own_query):
                    problems.append(
                        f"{tool.name}: query names sibling qualifier {sibling.qualifier!r}"
                    )
                sibling_token = "_".join(sibling.qualifier.lower().split())
                if re.search(rf"(^|_){re.escape(sibling_token)}(_|$)", own_name):
                    problems.append(
                        f"{tool.name}: name contains sibling qualifier {sibling.qualifier!r}"
                    )
            lowered = tool.query.lower()
            leaked = [name for name in all_names if name.lower() in lowered]
            if leaked:
                problems.append(f"{tool.name}: query contains tool name(s) {leaked}")
            problems.extend(_lint_implied(tool, family, all_names))
        problems.extend(_lint_terse(family))
    return problems


def _lint_terse(family: Family) -> list[str]:
    """Problems with a family's shared description: it must favour no variant."""
    terse = family.description_terse
    if terse is None:
        return []
    problems: list[str] = []
    for tool in family.variants:
        # A tool name is snake_case, so a word-boundary match on its qualifier
        # would miss it; check the whole name as a substring first.
        if tool.name.lower() in terse.lower():
            problems.append(
                f"{family.domain}/{family.id}: terse description contains tool name {tool.name!r}"
            )
            continue
        if _phrase_in(tool.qualifier, terse):
            problems.append(
                f"{family.domain}/{family.id}: terse description names qualifier {tool.qualifier!r}"
            )
            continue
        for word in tool.qualifier.lower().split():
            if word not in _GENERIC_QUALIFIER_WORDS and _phrase_in(word, terse):
                problems.append(
                    f"{family.domain}/{family.id}: terse description contains qualifier "
                    f"word {word!r}"
                )
    return problems
