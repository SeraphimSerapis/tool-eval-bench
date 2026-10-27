"""Toolscale: catalog validation, grid construction, grading, scoring, and CLI wiring."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from tool_eval_bench.domain.adapters import ChatCompletionResult, ProviderToolCall
from tool_eval_bench.domain.tools_large import LARGE_TOOLSET
from tool_eval_bench.plugins.registry import available_plugins, get_plugin
from tool_eval_bench.plugins.toolscale.catalog import (
    Catalog,
    CatalogError,
    lint_catalog,
    load_builtin_catalog,
    parse_catalog,
)
from tool_eval_bench.plugins.toolscale.grid import (
    CONTROL,
    CORRECT,
    HALLUCINATED,
    NO_CALL,
    TRUNCATED,
    WRONG_FILLER,
    WRONG_NEIGHBOUR,
    GridError,
    ToolscaleItem,
    build_items,
    grade,
    max_count,
    openai_tools,
    select_targets,
    validate_counts,
)
from tool_eval_bench.plugins.toolscale.plugin import ToolscalePlugin
from tool_eval_bench.plugins.toolscale.similarity import TfidfSpace, ToolSimilarity
from tool_eval_bench.plugins.toolscale.stats import (
    bootstrap_mean_interval,
    slope_per_doubling,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# Synthetic catalog
# ---------------------------------------------------------------------------

_QUALIFIERS = ("alpha", "beta", "gamma", "delta")
# Implied queries refer to a variant by its position, never by its qualifier.
_ORDINALS = {"alpha": "first", "beta": "second", "gamma": "third", "delta": "fourth"}


def _catalog_data(domains: int = 5, families: int = 4, variants: int = 4) -> dict[str, Any]:
    """A structurally valid catalog whose names and queries follow the lint rules."""
    out: list[dict[str, Any]] = []
    for d in range(domains):
        fams = []
        for f in range(families):
            vs = []
            for v in range(variants):
                qualifier = _QUALIFIERS[v % len(_QUALIFIERS)] + ("" if v < 4 else str(v))
                vs.append(
                    {
                        "name": f"op{f}_dom{d}_{qualifier}",
                        "qualifier": qualifier,
                        "description": f"Operation {f} in domain {d}, {qualifier} flavour only.",
                        "parameters": {f"key{d}": {"type": "string", "description": "Key"}},
                        "required": [f"key{d}"],
                        "query": f"Please run the {qualifier} operation {f} for key K-{d}{f}{v}.",
                        "query_implied": (
                            f"Please run operation {f} the {_ORDINALS.get(qualifier, 'other')} "
                            f"way for key K-{d}{f}{v}."
                        ),
                    }
                )
            fams.append(
                {
                    "id": f"fam{f}",
                    "description_terse": f"Run operation {f} in domain {d}.",
                    "variants": vs,
                }
            )
        out.append({"id": f"dom{d}", "families": fams})
    return {"version": "test", "domains": out}


def _catalog(**kwargs: Any) -> Catalog:
    return parse_catalog(json.dumps(_catalog_data(**kwargs)).encode(), source="builtin")


def _items(catalog: Catalog | None = None, **overrides: Any) -> list[ToolscaleItem]:
    params: dict[str, Any] = {
        "counts": [1, 4, 8, 16],
        "similarities": ["near", "related", "distinct"],
        "limit": None,
        "trials": 1,
        "seed": 7,
    }
    params.update(overrides)
    return build_items(catalog or _catalog(), **params)


def _by_name(catalog: Catalog) -> dict[str, Any]:
    return {tool.name: tool for tool in catalog.tools}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestCatalogParsing:
    def test_valid_catalog_builds_the_tree(self) -> None:
        catalog = _catalog()
        assert len(catalog.domains) == 5
        assert len(catalog.families) == 20
        assert len(catalog.tools) == 80

    def test_hash_pins_content(self) -> None:
        data = _catalog_data()
        first = parse_catalog(json.dumps(data).encode(), source="builtin").sha256
        assert first == parse_catalog(json.dumps(data, indent=2).encode(), source="builtin").sha256
        data["domains"][0]["families"][0]["variants"][0]["description"] += " Changed."
        assert parse_catalog(json.dumps(data).encode(), source="builtin").sha256 != first

    def test_openai_definition_is_strict(self) -> None:
        tool = _catalog().tools[0].to_openai_tool()
        params = tool["function"]["parameters"]
        assert tool["type"] == "function"
        assert params["additionalProperties"] is False
        assert params["required"] == ["key0"]

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            pytest.param(
                lambda d: d["domains"][0]["families"][0]["variants"].pop(),
                "at least 4 are needed",
                id="three-variants",
            ),
            pytest.param(
                lambda d: d["domains"][0]["families"].pop(), "at least 4 are needed", id="3-fams"
            ),
            pytest.param(
                lambda d: d["domains"][1]["families"][0]["variants"][0].update(
                    name=d["domains"][0]["families"][0]["variants"][0]["name"]
                ),
                "duplicate tool name",
                id="duplicate-name",
            ),
            pytest.param(
                lambda d: d["domains"][0]["families"][0]["variants"][0].update(name="bad name!"),
                "name must match",
                id="bad-name",
            ),
            pytest.param(
                lambda d: d["domains"][0]["families"][0]["variants"][0].update(required=["nope"]),
                "not declared",
                id="undeclared-required",
            ),
            pytest.param(
                lambda d: d["domains"][0]["families"][0]["variants"][0].update(
                    parameters={"x": {"type": "object"}}
                ),
                "needs a 'type'",
                id="bad-param-type",
            ),
            pytest.param(
                lambda d: d["domains"][0]["families"][0]["variants"][0].update(query=" "),
                "'query' must be a non-empty string",
                id="empty-query",
            ),
            pytest.param(lambda d: d.pop("version"), "version", id="no-version"),
            pytest.param(lambda d: d.update(domains=d["domains"][:1]), "two", id="one-domain"),
        ],
    )
    def test_rejects_invalid_catalogs(self, mutate: Any, message: str) -> None:
        data = _catalog_data()
        mutate(data)
        with pytest.raises(CatalogError, match=message):
            parse_catalog(json.dumps(data).encode(), source="custom")

    def test_description_levels_degrade_what_the_model_reads(self) -> None:
        tool = _catalog().tools[0]
        clean = tool.to_openai_tool()["function"]
        terse = tool.to_openai_tool("terse")["function"]
        bare = tool.to_openai_tool("bare")["function"]
        assert clean["description"] == tool.description
        assert terse["description"] == "Run operation 0 in domain 0."
        assert "description" not in bare
        for degraded in (terse, bare):
            assert degraded["name"] == tool.name
            assert all(
                "description" not in p for p in degraded["parameters"]["properties"].values()
            )
            assert degraded["parameters"]["required"] == clean["parameters"]["required"]
        # The catalog's own parameter schema is never mutated by degrading a copy.
        assert "description" in tool.parameters["key0"]
        with pytest.raises(ValueError, match="unknown description level"):
            tool.to_openai_tool("vague")

    def test_degraded_levels_keep_only_callable_schema_keys(self) -> None:
        data = _catalog_data()
        data["domains"][0]["families"][0]["variants"][0]["parameters"]["key0"].update(
            title="Alpha key", examples=["alpha flavour"], enum=["K-000", "K-001"]
        )
        tool = parse_catalog(json.dumps(data).encode(), source="custom").tools[0]
        for level in ("terse", "bare"):
            schema = tool.to_openai_tool(level)["function"]["parameters"]["properties"]["key0"]
            assert schema == {"type": "string", "enum": ["K-000", "K-001"]}
        clean = tool.to_openai_tool()["function"]["parameters"]["properties"]["key0"]
        assert clean["title"] == "Alpha key"

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(CatalogError, match="not valid JSON"):
            parse_catalog(b"{nope", source="custom")


class TestCatalogLint:
    def test_clean_catalog_has_no_findings(self) -> None:
        assert lint_catalog(_catalog()) == []

    def _lint_after(self, **variant_changes: Any) -> list[str]:
        data = _catalog_data()
        data["domains"][0]["families"][0]["variants"][0].update(variant_changes)
        return lint_catalog(parse_catalog(json.dumps(data).encode(), source="custom"))

    def test_query_must_name_its_qualifier(self) -> None:
        findings = self._lint_after(query="Run operation 0 for key K-1.")
        assert any("does not name qualifier 'alpha'" in f for f in findings)

    def test_query_must_not_name_a_sibling_qualifier(self) -> None:
        findings = self._lint_after(query="Run the alpha operation 0, not beta, for K-1.")
        assert any("sibling qualifier 'beta'" in f for f in findings)

    def test_query_must_not_contain_a_tool_name(self) -> None:
        findings = self._lint_after(query="Call op0_dom0_alpha with alpha for key K-1.")
        assert any("contains tool name" in f for f in findings)

    def test_a_qualifier_may_contain_a_sibling_qualifier(self) -> None:
        data = _catalog_data()
        variant = data["domains"][0]["families"][0]["variants"][1]
        variant.update(
            name="op0_dom0_super_alpha",
            qualifier="super alpha",
            query="Please run the super alpha operation 0 for key K-001.",
        )
        catalog = parse_catalog(json.dumps(data).encode(), source="custom")
        assert lint_catalog(catalog) == []

    def test_implied_query_must_not_name_its_qualifier(self) -> None:
        findings = self._lint_after(query_implied="Run the alpha operation 0 for K-1.")
        assert any("implied query names its qualifier 'alpha'" in f for f in findings)

    def test_implied_query_must_not_name_a_sibling(self) -> None:
        findings = self._lint_after(query_implied="Run operation 0, not the beta one, for K-1.")
        assert any("implied query names sibling qualifier 'beta'" in f for f in findings)

    def test_implied_query_must_not_use_a_distinctive_qualifier_word(self) -> None:
        data = _catalog_data()
        variant = data["domains"][0]["families"][0]["variants"][1]
        variant.update(
            name="op0_dom0_same_day",
            qualifier="same day",
            query="Please run the same day operation 0 for key K-001.",
            query_implied="Please run operation 0 so it is done the same evening, key K-001.",
        )
        findings = lint_catalog(parse_catalog(json.dumps(data).encode(), source="custom"))
        assert any("qualifier word 'same'" in f for f in findings)
        # "day" alone is too generic to give the answer away.
        variant["query_implied"] = "Please run operation 0 before the day ends, key K-001."
        assert lint_catalog(parse_catalog(json.dumps(data).encode(), source="custom")) == []

    def test_terse_description_must_favour_no_variant(self) -> None:
        data = _catalog_data()
        data["domains"][0]["families"][0]["description_terse"] = "Run the alpha operation."
        findings = lint_catalog(parse_catalog(json.dumps(data).encode(), source="custom"))
        assert any("terse description names qualifier 'alpha'" in f for f in findings)

    def test_terse_description_must_not_contain_a_tool_name(self) -> None:
        data = _catalog_data()
        data["domains"][0]["families"][0]["description_terse"] = "Run op0_dom0_alpha for a key."
        findings = lint_catalog(parse_catalog(json.dumps(data).encode(), source="custom"))
        assert any("terse description contains tool name" in f for f in findings)

    def test_every_variant_name_carries_its_qualifier(self) -> None:
        findings = self._lint_after(name="op0_dom0_generic")
        assert any("does not carry qualifier" in f for f in findings)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class TestGridValidation:
    def test_max_count_is_outside_families_plus_one(self) -> None:
        # 20 families, 4 in the target's domain: 16 outside, so N <= 17.
        assert max_count(_catalog()) == 17

    def test_crowded_max_count_reserves_the_largest_families(self) -> None:
        # 16 outside families of 4 tools; 3 reserved for distinct neighbours leaves
        # 52 filler tools, plus the target and 3 neighbours.
        assert max_count(_catalog(), "crowded") == 56
        data = _catalog_data()
        extra = dict(data["domains"][1]["families"][0]["variants"][0])
        extra.update(name="op0_dom1_epsilon", qualifier="epsilon", query="epsilon op K-1.")
        data["domains"][1]["families"][0]["variants"].append(extra)
        catalog = parse_catalog(json.dumps(data).encode(), source="custom")
        # A 5-variant family may be one of the reserved three, so the worst case
        # does not grow; for the target's own domain it is simply not outside.
        assert max_count(catalog, "crowded") == 56

    def test_counts_beyond_sparse_need_crowded(self) -> None:
        catalog = _catalog()
        with pytest.raises(GridError, match="crowded fillers reach 56"):
            validate_counts([32], catalog)
        assert validate_counts([32], catalog, ["sparse", "crowded"]) == [32]
        with pytest.raises(GridError, match="maximum of 56"):
            validate_counts([57], catalog, ["sparse", "crowded"])

    def test_beyond_sparse_only_crowded_cells_exist(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[16, 32, 56], densities=["sparse", "crowded"])
        sparse_counts = {i.count for i in items if i.density == "sparse"}
        crowded_counts = {i.count for i in items if i.density == "crowded"}
        assert sparse_counts == {16}
        assert crowded_counts == {16, 32, 56}
        for item in items:
            assert len(item.tools) == item.count == len({t.name for t in item.tools})
            assert [r for _, r in item.roles].count("neighbour") == 3

    @pytest.mark.parametrize("bad", [0, 2, 3, 18])
    def test_rejects_unbuildable_counts(self, bad: int) -> None:
        with pytest.raises(GridError):
            validate_counts([bad], _catalog())

    def test_counts_are_sorted_and_deduplicated(self) -> None:
        assert validate_counts([8, 1, 4, 8], _catalog()) == [1, 4, 8]

    def test_unknown_similarity_is_rejected(self) -> None:
        with pytest.raises(GridError, match="unknown similarity"):
            _items(similarities=["near", "fuzzy"])

    def test_largest_supported_count_builds(self) -> None:
        items = _items(counts=[17], similarities=["distinct"], limit=4)
        assert {len(item.tools) for item in items} == {17}


class TestTargets:
    def test_default_takes_one_target_per_family(self) -> None:
        catalog = _catalog()
        targets = select_targets(catalog, None, seed=1)
        assert len(targets) == 20
        assert len({(t.domain, t.family) for t in targets}) == 20

    def test_small_limit_spreads_across_domains(self) -> None:
        targets = select_targets(_catalog(), 5, seed=1)
        assert sorted(t.domain for t in targets) == [f"dom{d}" for d in range(5)]

    def test_limit_beyond_families_moves_to_second_variants(self) -> None:
        targets = select_targets(_catalog(), 40, seed=1)
        assert len({t.name for t in targets}) == 40

    def test_zero_limit_is_rejected(self) -> None:
        with pytest.raises(GridError):
            select_targets(_catalog(), 0, seed=1)


class TestGridConstruction:
    def test_each_item_has_target_three_neighbours_and_fillers(self) -> None:
        for item in _items():
            roles = [role for _, role in item.roles]
            assert len(item.tools) == item.count
            assert len({t.name for t in item.tools}) == item.count
            assert roles.count("target") == 1
            if item.similarity == CONTROL:
                assert item.tools == (item.target,)
                continue
            assert roles.count("neighbour") == 3
            assert roles.count("filler") == item.count - 4

    def test_neighbours_come_from_their_tier(self) -> None:
        catalog = _catalog()
        tools = _by_name(catalog)
        for item in _items(catalog):
            neighbours = [tools[n] for n, role in item.roles if role == "neighbour"]
            fillers = [tools[n] for n, role in item.roles if role == "filler"]
            t = item.target
            if item.similarity == "near":
                assert all(n.domain == t.domain and n.family == t.family for n in neighbours)
            elif item.similarity == "related":
                assert all(n.domain == t.domain and n.family != t.family for n in neighbours)
                assert len({n.family for n in neighbours}) == 3
            elif item.similarity == "distinct":
                assert all(n.domain != t.domain for n in neighbours)
            # Fillers are unrelated to the target and to each other: one per family.
            assert all(f.domain != t.domain for f in fillers)
            outside = [(x.domain, x.family) for x in fillers + neighbours if x.domain != t.domain]
            assert len(outside) == len(set(outside))

    def _group(self, items: Iterable[ToolscaleItem]) -> dict[tuple, dict[str, ToolscaleItem]]:
        groups: dict[tuple, dict[str, ToolscaleItem]] = {}
        for item in items:
            if item.similarity != CONTROL:
                key = (item.target.name, item.trial, item.count)
                groups.setdefault(key, {})[item.similarity] = item
        return groups

    def test_tiers_share_fillers_and_slots(self) -> None:
        for by_level in self._group(_items()).values():
            items = list(by_level.values())
            fillers = [{n for n, r in i.roles if r == "filler"} for i in items]
            assert all(f == fillers[0] for f in fillers)
            # Fillers sit in the same slots, the neighbours in the remaining
            # ones, and the target at the same index, whichever tier is shown.
            layouts = [[dict(i.roles)[t.name] for t in i.tools] for i in items]
            assert all(layout == layouts[0] for layout in layouts)
            filler_order = [
                [t.name for t in i.tools if dict(i.roles)[t.name] == "filler"] for i in items
            ]
            assert all(order == filler_order[0] for order in filler_order)

    def test_fillers_are_nested_as_count_grows(self) -> None:
        items = _items()
        for target in {i.target.name for i in items}:
            fillers_by_count = {
                i.count: {n for n, r in i.roles if r == "filler"}
                for i in items
                if i.target.name == target and i.similarity == "near"
            }
            assert fillers_by_count[4] <= fillers_by_count[8] <= fillers_by_count[16]
            assert len(fillers_by_count[16]) == 12

    def test_relative_position_is_held_across_counts(self) -> None:
        items = [i for i in _items() if i.similarity == "near"]
        for target in {i.target.name for i in items}:
            positions = {i.count: i.position for i in items if i.target.name == target}
            # index = round(u * (N - 1)), so each position is within 0.5/(N-1) of u.
            assert abs(positions[8] - positions[16]) <= 0.5 / 7 + 0.5 / 15 + 1e-9

    def test_same_seed_reproduces_the_grid(self) -> None:
        a = _items(seed=3)
        b = _items(seed=3)
        assert [(i.item_id, [t.name for t in i.tools]) for i in a] == [
            (i.item_id, [t.name for t in i.tools]) for i in b
        ]

    def test_seed_and_trial_change_the_draw(self) -> None:
        def layout(items: list[ToolscaleItem]) -> list[list[str]]:
            return [[t.name for t in i.tools] for i in items if i.count == 16]

        assert layout(_items(seed=3)) != layout(_items(seed=4))
        two = _items(seed=3, trials=2)
        first = [i for i in two if i.trial == 0]
        second = [i for i in two if i.trial == 1]
        assert len(first) == len(second)
        assert layout(first) != layout(second)

    def test_cues_share_the_tool_list_and_differ_in_the_query(self) -> None:
        items = _items(cues=["literal", "implied"], counts=[1, 8])
        pairs: dict[tuple, dict[str, ToolscaleItem]] = {}
        for item in items:
            key = (item.target.name, item.trial, item.count, item.similarity)
            pairs.setdefault(key, {})[item.cue] = item
        assert pairs and all(set(p) == {"literal", "implied"} for p in pairs.values())
        for pair in pairs.values():
            literal, implied = pair["literal"], pair["implied"]
            assert literal.tools == implied.tools
            assert literal.query == literal.target.query
            assert implied.query == implied.target.query_implied
            assert literal.item_id.replace("/literal/", "/implied/") == implied.item_id

    def test_unknown_cue_is_rejected(self) -> None:
        with pytest.raises(GridError, match="unknown cue"):
            _items(cues=["vague"])

    def test_implied_cue_needs_every_implied_query(self) -> None:
        data = _catalog_data()
        del data["domains"][2]["families"][1]["variants"][3]["query_implied"]
        catalog = parse_catalog(json.dumps(data).encode(), source="custom")
        assert _items(catalog, cues=["literal"])  # literal alone still builds
        with pytest.raises(GridError, match="1 lack one"):
            _items(catalog, cues=["implied"])

    def test_crowded_fillers_are_whole_families(self) -> None:
        catalog = _catalog()
        tools = _by_name(catalog)
        items = _items(catalog, counts=[1, 4, 8, 16], densities=["sparse", "crowded"])
        crowded = [i for i in items if i.density == "crowded"]
        # The control and N=4 have no fillers, so crowded skips them.
        assert {i.count for i in crowded} == {8, 16}
        for item in crowded:
            fillers = [tools[n] for n, r in item.roles if r == "filler"]
            families: dict[tuple[str, str], int] = {}
            for f in fillers:
                assert f.domain != item.target.domain
                families[(f.domain, f.family)] = families.get((f.domain, f.family), 0) + 1
            assert set(families.values()) == {4}  # N-4 is a multiple of 4 here
            neighbour_families = {
                (tools[n].domain, tools[n].family) for n, r in item.roles if r == "neighbour"
            }
            assert not neighbour_families & set(families)

    def test_crowded_and_sparse_share_neighbours_and_slots(self) -> None:
        items = _items(counts=[8, 16], densities=["sparse", "crowded"])
        groups: dict[tuple, dict[str, ToolscaleItem]] = {}
        for item in items:
            key = (item.target.name, item.trial, item.count, item.similarity)
            groups.setdefault(key, {})[item.density] = item
        for pair in groups.values():
            sparse, crowded = pair["sparse"], pair["crowded"]
            layout = [[dict(i.roles)[t.name] for t in i.tools] for i in (sparse, crowded)]
            assert layout[0] == layout[1]
            neighbours = [
                [t.name for t in i.tools if dict(i.roles)[t.name] != "filler"]
                for i in (sparse, crowded)
            ]
            assert neighbours[0] == neighbours[1]
            assert crowded.item_id.endswith("/crowded/t0")

    def test_crowded_fillers_are_nested_and_leave_sparse_unchanged(self) -> None:
        both = _items(counts=[8, 16], densities=["sparse", "crowded"])
        sparse_only = _items(counts=[8, 16])
        assert [(i.item_id, i.tools) for i in both if i.density == "sparse"] == [
            (i.item_id, i.tools) for i in sparse_only
        ]
        for target in {i.target.name for i in both}:
            by_count = {
                i.count: {n for n, r in i.roles if r == "filler"}
                for i in both
                if i.target.name == target and i.density == "crowded" and i.similarity == "near"
            }
            assert by_count[8] <= by_count[16]

    def test_counts_without_cells_are_rejected(self) -> None:
        # Crowded skips N=4 (no fillers), so a crowded-only grid at N=4 is empty.
        with pytest.raises(GridError, match=r"\[4\] produce no cells"):
            _items(counts=[4], densities=["crowded"])
        with pytest.raises(GridError, match=r"\[4\] produce no cells"):
            _items(counts=[1, 4, 8], densities=["crowded"])
        assert _items(counts=[1, 8], densities=["crowded"])

    def test_private_catalog_names_stay_out_of_cue_errors(self) -> None:
        data = _catalog_data()
        del data["domains"][2]["families"][1]["variants"][3]["query_implied"]
        catalog = parse_catalog(json.dumps(data).encode(), source="custom")
        with pytest.raises(GridError) as caught:
            _items(catalog, cues=["implied"])
        assert "op1_dom2" not in str(caught.value)

    def test_description_levels_share_everything_but_the_text(self) -> None:
        items = _items(counts=[1, 8], descriptions=["clean", "terse", "bare"])
        groups: dict[tuple, dict[str, ToolscaleItem]] = {}
        for item in items:
            key = (item.target.name, item.trial, item.count, item.similarity, item.cue)
            groups.setdefault(key, {})[item.descriptions] = item
        assert groups and all(set(g) == {"clean", "terse", "bare"} for g in groups.values())
        for group in groups.values():
            assert len({tuple(t.name for t in g.tools) for g in group.values()}) == 1
            assert len({g.query for g in group.values()}) == 1
            assert group["terse"].item_id.endswith("/terse/t0")
            assert group["clean"].item_id.endswith(f"/{group['clean'].cue}/t0")
        sent = openai_tools(next(i for i in items if i.descriptions == "bare"))
        assert all("description" not in t["function"] for t in sent)

    def test_terse_needs_every_family_description(self) -> None:
        data = _catalog_data()
        del data["domains"][1]["families"][2]["description_terse"]
        catalog = parse_catalog(json.dumps(data).encode(), source="custom")
        assert _items(catalog, descriptions=["clean", "bare"])
        with pytest.raises(GridError, match="1 lack one") as caught:
            _items(catalog, descriptions=["terse"])
        assert "dom1" not in str(caught.value)  # private catalog: no names
        with pytest.raises(GridError, match="unknown description level"):
            _items(descriptions=["poetic"])

    def test_unknown_density_is_rejected(self) -> None:
        with pytest.raises(GridError, match="unknown filler density"):
            _items(densities=["packed"])

    def test_trials_must_be_positive(self) -> None:
        with pytest.raises(GridError):
            _items(trials=0)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _response(name: str | None, *, finish: str = "tool_calls") -> ChatCompletionResult:
    calls = [ProviderToolCall(id="c1", name=name, arguments_str="{}")] if name else []
    return ChatCompletionResult(content="", tool_calls=calls, finish_reason=finish)


class TestGrading:
    def _item(self) -> ToolscaleItem:
        return next(i for i in _items(counts=[8]) if i.similarity == "near")

    def test_target_is_correct(self) -> None:
        item = self._item()
        assert grade(item, _response(item.target.name)) == (CORRECT, item.target.name)

    def test_neighbour_and_filler_picks_are_told_apart(self) -> None:
        item = self._item()
        neighbour = next(n for n, r in item.roles if r == "neighbour")
        filler = next(n for n, r in item.roles if r == "filler")
        assert grade(item, _response(neighbour))[0] == WRONG_NEIGHBOUR
        assert grade(item, _response(filler))[0] == WRONG_FILLER

    def test_unoffered_name_is_hallucinated(self) -> None:
        assert grade(self._item(), _response("not_offered"))[0] == HALLUCINATED

    def test_no_call_and_truncation(self) -> None:
        assert grade(self._item(), _response(None, finish="stop"))[0] == NO_CALL
        assert grade(self._item(), _response(None, finish="length"))[0] == TRUNCATED


# ---------------------------------------------------------------------------
# Stats and similarity
# ---------------------------------------------------------------------------


class TestStats:
    def test_wilson_handles_the_extremes(self) -> None:
        assert wilson_interval(0, 10) == (0.0, 27.8)
        assert wilson_interval(10, 10) == (72.2, 100.0)
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_slope_per_doubling(self) -> None:
        assert slope_per_doubling([(4, 100.0), (8, 90.0), (16, 80.0)]) == pytest.approx(-10.0)
        assert slope_per_doubling([(4, 100.0)]) is None

    def test_bootstrap_is_deterministic_and_brackets_the_mean(self) -> None:
        values = [0.0, 100.0, 50.0, 100.0, 0.0, 100.0]
        low, high = bootstrap_mean_interval(values)
        assert (low, high) == bootstrap_mean_interval(values)
        assert low <= sum(values) / len(values) <= high


class TestSimilarity:
    def test_cosine_bounds(self) -> None:
        space = TfidfSpace([["invoice", "open"], ["weather", "marine"]])
        a = space.vector(["invoice", "open"])
        assert space.cosine(a, a) == pytest.approx(1.0)
        assert space.cosine(a, space.vector(["weather"])) == 0.0

    def test_unseen_words_get_the_highest_weight(self) -> None:
        space = TfidfSpace([["invoice"], ["invoice", "open"]])
        assert space.idf("never_seen") > space.idf("open") > space.idf("invoice")

    def test_category_l_tools_are_measurable_in_catalog_space(self) -> None:
        sim = ToolSimilarity(t.to_openai_tool() for t in _catalog().tools)
        assert sim.between(LARGE_TOOLSET[0], LARGE_TOOLSET[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Picks the target, unless *policy* says otherwise for an item."""

    def __init__(self, policy: Any = None, tools_policy: Any = None) -> None:
        self.policy = policy
        self.tools_policy = tools_policy
        self.requests: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        self.requests.append(kwargs)
        query = kwargs["messages"][-1]["content"]
        names = [t["function"]["name"] for t in kwargs["tools"]]
        if self.tools_policy is not None:
            outcome = self.tools_policy(query, names, kwargs["tools"])
        elif self.policy is not None:
            outcome = self.policy(query, names)
        else:
            outcome = None
        if outcome is not None:
            if isinstance(outcome, ChatCompletionResult):
                return outcome
            if isinstance(outcome, Exception):
                raise outcome
            name = outcome
        else:
            # The synthetic queries name the qualifier and operation: pick by both.
            name = next(n for n in names if _matches(query, n))
        result = _response(name)
        result.prompt_tokens = 50 * len(names)
        result.completion_tokens = 5
        return result


def _matches(query: str, name: str) -> bool:
    """Whether a synthetic query, literal or implied, asks for tool *name*."""
    op, dom, qualifier = name.split("_")
    literal = f"{qualifier} operation {op[2:]}" in query
    implied = f"operation {op[2:]} the {_ORDINALS.get(qualifier, 'other')} way" in query
    return (literal or implied) and f"K-{dom[3:]}" in query


def _near_confuser(query: str, names: list[str]) -> str:
    """Wrong whenever a near sibling is present: picks a same-family other variant."""
    target = next(n for n in names if _matches(query, n))
    op, dom, _ = target.split("_")
    siblings = [n for n in names if n.startswith(f"{op}_{dom}_") and n != target]
    return siblings[0] if siblings else target


async def _run(items: list[ToolscaleItem], adapter: Any, catalog: Catalog) -> Any:
    return await ToolscalePlugin().run(
        adapter, model="m", base_url="http://x", items=items, catalog=catalog, concurrency=4
    )


@pytest.mark.asyncio
class TestToolscalePlugin:
    async def test_perfect_selection_scores_100(self) -> None:
        catalog = _catalog()
        result = await _run(_items(catalog), _StubAdapter(), catalog)
        assert result.score == 100.0
        assert result.details["control"]["literal"]["accuracy"] == 100.0
        assert (result.details["call_rate"], result.details["accuracy_when_called"]) == (100, 100)
        assert result.details["errors"] == 0
        assert all(e["delta"] == 0 for e in result.details["similarity_effects"])

    async def test_cue_effect_isolates_keyword_reliance(self) -> None:
        """A model that only matches keywords: right on literal, confused on implied."""
        catalog = _catalog()
        items = _items(catalog, cues=["literal", "implied"], counts=[1, 4, 8])

        def keyword_matcher(query: str, names: list[str]) -> str:
            target = next(n for n in names if _matches(query, n))
            if target.split("_")[2] in query:
                return target
            return next(n for n in names if n != target)

        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(keyword_matcher), catalog)
        d = result.details
        assert d["by_cue"]["literal"]["accuracy_when_called"] == 100.0
        assert d["by_cue"]["implied"]["accuracy_when_called"] == 0.0
        assert d["by_cue"]["implied"]["call_rate"] == 100.0
        assert d["cue_effects"] and all(e["delta"] == 100 for e in d["cue_effects"])
        assert set(d["control"]) == {"literal", "implied"}
        # The literal query leans toward its target by word overlap; the implied
        # one carries no qualifier, so its lean is smaller.
        q = d["query_check"]
        assert q["literal"]["margin_over_sibling"] > q["implied"]["margin_over_sibling"]
        report = "\n".join(plugin.render_report_section(result))
        for heading in ("Headline by Cue", "Cue Effect", "Query Check", "(implied cue)"):
            assert heading in report

    async def test_density_effect_isolates_crowding(self) -> None:
        """A model that loses the target only when the list holds whole clusters."""
        catalog = _catalog()
        items = _items(catalog, counts=[4, 8, 16], densities=["sparse", "crowded"])

        def crowd_victim(query: str, names: list[str]) -> str:
            target = next(n for n in names if _matches(query, n))
            families = [n.rsplit("_", 1)[0] for n in names if n != target]
            clustered = [n for n in names if families.count(n.rsplit("_", 1)[0]) >= 4]
            return clustered[0] if clustered else target

        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(crowd_victim), catalog)
        d = result.details
        assert d["by_density"]["sparse"]["accuracy_when_called"] == 100.0
        assert d["by_density"]["crowded"]["accuracy_when_called"] == 0.0
        assert d["density_effects"] and all(e["delta"] == 100 for e in d["density_effects"])
        crowded_cells = [c for c in d["cells"] if c["density"] == "crowded"]
        assert all(c["outcomes"]["wrong_filler"] == c["items"] for c in crowded_cells)
        crowding = d["similarity_check"]["filler_crowding"]
        assert crowding["crowded"] > crowding["sparse"]
        report = "\n".join(plugin.render_report_section(result))
        for heading in ("Density Effect", "Headline by Fillers", "(literal cue, crowded fillers)"):
            assert heading in report

    async def test_counts_beyond_sparse_aggregate_cleanly(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[8, 16, 32], densities=["sparse", "crowded"], limit=6)
        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(), catalog)
        d = result.details
        assert d["counts"] == [8, 16, 32]
        # Count effects run over each slice's own measured range.
        ranges = {(e["density"], e["from_count"], e["to_count"]) for e in d["count_effects"]}
        assert ranges == {("sparse", 8, 16), ("crowded", 8, 32)}
        # Density effects only where both exist.
        assert {e["count"] for e in d["density_effects"]} == {8, 16}
        # Crowding is measured on each density's own largest list.
        assert set(d["similarity_check"]["filler_crowding"]) == {"sparse", "crowded"}
        report = "\n".join(plugin.render_report_section(result))
        assert "N=32" in report

    async def test_control_only_run_scores_the_control(self) -> None:
        catalog = _catalog()
        result = await _run(_items(catalog, counts=[1]), _StubAdapter(), catalog)
        assert result.score == 100.0
        assert result.details["grid_items"] == result.details["total"]
        assert result.rating.startswith("★★★★★")

    async def test_failed_items_table_survives_hostile_model_text(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[4], similarities=["near"], limit=2)

        def hostile(query: str, names: list[str]) -> Any:
            if query == items[0].target.query:
                return _response("made|up\rname")
            return ChatCompletionResult(content="Which | one?\r\nTell me.", finish_reason="stop")

        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(hostile), catalog)
        report = plugin.render_report_section(result)
        start = report.index(next(line for line in report if line.startswith("### Failed Items")))
        rows = [line for line in report[start:] if line.startswith("| `")]
        assert len(rows) == 2
        for row in rows:
            # 4 cells: unescaped pipes are exactly the 5 separators.
            assert len([c for i, c in enumerate(row) if c == "|" and row[i - 1] != "\\"]) == 5
            assert "\r" not in row

    async def test_private_catalog_errors_log_coordinates_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        catalog = parse_catalog(json.dumps(_catalog_data()).encode(), source="custom")
        items = _items(catalog, counts=[4], similarities=["near"], limit=1)

        def boom(query: str, names: list[str]) -> Any:
            return RuntimeError("server exploded")

        with caplog.at_level(logging.DEBUG, logger="tool_eval_bench.plugins.toolscale.plugin"):
            await _run(items, _StubAdapter(boom), catalog)
        assert caplog.records
        assert all(items[0].target.name not in r.getMessage() for r in caplog.records)

    async def test_sparse_lists_define_the_flat_nearest_check(self) -> None:
        catalog = _catalog()
        both = _items(catalog, counts=[8, 16], densities=["sparse", "crowded"], limit=6)
        sparse = _items(catalog, counts=[8, 16], limit=6)
        from tool_eval_bench.plugins.toolscale.plugin import similarity_check

        assert (
            similarity_check(both, catalog)["nearest_by_count"]
            == similarity_check(sparse, catalog)["nearest_by_count"]
        )

    async def test_description_effect_isolates_reliance_on_descriptions(self) -> None:
        """A model that needs descriptions: right when clean, confused when degraded."""
        catalog = _catalog()
        items = _items(
            catalog,
            counts=[1, 4, 8],
            cues=["literal", "implied"],
            descriptions=["clean", "terse", "bare"],
        )

        def needs_descriptions(query: str, names: list[str], tools: list[dict]) -> str:
            target = next(n for n in names if _matches(query, n))
            described = {t["function"]["name"]: t["function"].get("description") for t in tools}
            if described[target] and "flavour" in described[target]:
                return target
            siblings = [n for n in names if n != target]
            return siblings[0] if siblings else target

        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(tools_policy=needs_descriptions), catalog)
        d = result.details
        assert d["by_descriptions"]["clean"]["accuracy_when_called"] == 100.0
        assert d["by_descriptions"]["terse"]["accuracy_when_called"] == 0.0
        assert d["by_descriptions"]["bare"]["accuracy_when_called"] == 0.0
        effects = d["description_effects"]
        assert {e["versus"] for e in effects} == {"terse", "bare"}
        assert all(e["delta"] == 100 for e in effects)
        assert {e["cue"] for e in effects} == {"literal", "implied"}
        # Controls exist per cue and description level, labelled by both.
        assert set(d["control"]) == {
            f"{cue}, {level}"
            for cue in ("literal", "implied")
            for level in ("clean", "terse", "bare")
        }
        # Degraded descriptions remove the words that explain each variant.
        q = d["query_check"]
        assert q["literal, bare"]["query_to_target"] < q["literal"]["query_to_target"]
        report = "\n".join(plugin.render_report_section(result))
        for heading in (
            "Description Effect",
            "Headline by Descriptions",
            "(implied cue, terse descriptions)",
        ):
            assert heading in report

    async def test_crowded_only_run_has_no_phantom_sparse_slice(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[1, 8, 16], densities=["crowded"], limit=4)
        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(), catalog)
        d = result.details
        assert d["densities"] == ["crowded"]
        assert set(d["by_density"]) == {"crowded"}
        assert d["control"]  # the control still ran, as sparse
        report = "\n".join(plugin.render_report_section(result))
        assert "sparse fillers" not in report and "Headline by Fillers" not in report

    async def test_effect_tables_say_all_when_no_other_axis_varies(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[8], descriptions=["clean", "bare"], limit=3)
        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(), catalog)
        report = plugin.render_report_section(result)
        start = report.index("### Description Effect")
        row = next(line for line in report[start:] if line.startswith("| all | bare |"))
        assert row

    async def test_requests_carry_the_item_tool_list(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[8], similarities=["near"], limit=2)
        adapter = _StubAdapter()
        await _run(items, adapter, catalog)
        sent = sorted([t["function"]["name"] for t in r["tools"]] for r in adapter.requests)
        assert sent == sorted([t.name for t in i.tools] for i in items)
        assert all(r["stream"] is True and r["tool_choice"] == "auto" for r in adapter.requests)

    async def test_near_confusion_is_attributed_to_similarity(self) -> None:
        catalog = _catalog()
        result = await _run(_items(catalog), _StubAdapter(_near_confuser), catalog)
        d = result.details
        near_cells = [c for c in d["cells"] if c["similarity"] == "near"]
        other_cells = [c for c in d["cells"] if c["similarity"] != "near"]
        assert all(c["accuracy"] == 0 for c in near_cells)
        assert all(c["outcomes"]["wrong_neighbour"] == c["items"] for c in near_cells)
        assert all(c["accuracy"] == 100 for c in other_cells)
        near_effects = [e for e in d["similarity_effects"] if e["versus"] == "near"]
        assert near_effects and all(e["delta"] == 100 for e in near_effects)
        # Similarity, not count, drove the drop: no tier changes with N.
        assert all(e["delta"] == 0 for e in d["count_effects"])

    async def test_abstention_is_separated_from_confusion(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[4], similarities=["near"], limit=4)
        declined = items[0].target.query

        def policy(query: str, names: list[str]) -> Any:
            if query == declined:
                return ChatCompletionResult(
                    content="Which environment do you mean?", finish_reason="stop"
                )
            return next(n for n in names if _matches(query, n))

        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(policy), catalog)
        cell = result.details["cells"][0]
        assert (cell["accuracy"], cell["called"], cell["accuracy_when_called"]) == (75.0, 3, 100.0)
        assert (result.details["call_rate"], result.details["accuracy_when_called"]) == (75, 100)
        report = "\n".join(plugin.render_report_section(result))
        assert "Accuracy When a Tool Was Called" in report
        assert "*(no call)* Which environment do you mean?" in report

    async def test_request_errors_count_as_misses(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[4], similarities=["distinct"], limit=4)
        failing = items[0].target.query

        def policy(query: str, names: list[str]) -> Any:
            if query == failing:
                return RuntimeError("server exploded")
            return next(n for n in names if _matches(query, n))

        plugin = ToolscalePlugin()
        result = await _run(items, _StubAdapter(policy), catalog)
        assert result.details["errors"] == 1
        assert result.details["incomplete"] is True
        report = "\n".join(plugin.render_report_section(result))
        assert "*(request failed)* RuntimeError: server exploded" in report
        # An error is a miss, never a free pass: 3 of 4 items correct.
        assert result.score == 75.0
        assert result.details["grid_items"] == 4
        # But it is not the model declining to act: the call rate leaves it out.
        assert result.details["call_rate"] == 100.0
        assert result.details["cells"][0]["call_rate"] == 100.0
        assert result.details["cells"][0]["outcomes"]["error"] == 1

    async def test_transport_rejection_is_an_error(self) -> None:
        catalog = _catalog()
        items = _items(catalog, counts=[4], similarities=["distinct"], limit=1)

        def policy(query: str, names: list[str]) -> Any:
            return ChatCompletionResult(content="[server error 400]", transport_error_status=400)

        result = await _run(items, _StubAdapter(policy), catalog)
        assert result.details["errors"] == 1

    async def test_empty_grid_returns_a_zero_result(self) -> None:
        result = await ToolscalePlugin().run(_StubAdapter(), model="m", base_url="http://x")
        assert result.score == 0.0

    async def test_zero_concurrency_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            await ToolscalePlugin().run(
                _StubAdapter(), model="m", base_url="http://x", items=[], concurrency=0
            )

    async def test_details_are_json_and_private_catalogs_leak_no_items(self) -> None:
        data = _catalog_data()
        catalog = parse_catalog(json.dumps(data).encode(), source="custom")
        result = await _run(_items(catalog, limit=3), _StubAdapter(_near_confuser), catalog)
        stored = json.dumps(result.details)
        report = "\n".join(ToolscalePlugin().render_report_section(result))
        for tool in catalog.tools:
            assert tool.name not in stored
            assert tool.query not in stored
            assert tool.query not in report
            assert f"`{tool.name}`" not in report
        assert "withheld" in report

    async def test_report_has_every_section(self) -> None:
        catalog = _catalog()
        result = await _run(
            _items(catalog, trials=2, limit=5), _StubAdapter(_near_confuser), catalog
        )
        report = "\n".join(ToolscalePlugin().render_report_section(result))
        for heading in (
            "Accuracy When a Tool Was Called (literal cue)",
            "Call Rate (literal cue)",
            "Combined Accuracy (literal cue)",
            "Similarity Effect at Fixed N",
            "Count Effect at Fixed Similarity",
            "Error Mix",
            "Similarity Check",
            "Category L reference",
            "Accuracy by Target Position",
            "Failed Items",
        ):
            assert heading in report
        assert "±" in report  # trial spread appears with more than one trial


# ---------------------------------------------------------------------------
# Built-in catalog
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_builtin_catalog()


class TestBuiltinCatalog:
    def test_shape(self, catalog: Catalog) -> None:
        assert len(catalog.domains) == 17
        assert all(len(d.families) == 4 for d in catalog.domains)
        assert all(len(f.variants) == 4 for f in catalog.families)
        assert max_count(catalog) >= 64

    def test_lint_is_clean(self, catalog: Catalog) -> None:
        assert lint_catalog(catalog) == []

    def test_every_tool_has_an_implied_query(self, catalog: Catalog) -> None:
        assert all(t.query_implied for t in catalog.tools)

    def test_description_lengths_are_matched(self, catalog: Catalog) -> None:
        lengths = [len(t.description.split()) for t in catalog.tools]
        assert min(lengths) >= 12 and max(lengths) <= 30

    def test_measured_similarity_orders_the_tiers(self, catalog: Catalog) -> None:
        """For every tool: siblings > same-domain families > other domains."""
        sim = ToolSimilarity(t.to_openai_tool() for t in catalog.tools)
        defs = {t.name: t.to_openai_tool() for t in catalog.tools}

        def mean(target: str, others: list[str]) -> float:
            return sum(sim.between(defs[target], defs[o]) for o in others) / len(others)

        for family in catalog.families:
            domain = catalog.domain_of(family.variants[0])
            for tool in family.variants:
                near = [v.name for v in family.variants if v.name != tool.name]
                related = [v.name for f in domain.families if f.id != family.id for v in f.variants]
                distinct = [t.name for t in catalog.tools if t.domain != tool.domain]
                n, r, d = mean(tool.name, near), mean(tool.name, related), mean(tool.name, distinct)
                assert n > r > d, f"{tool.name}: near {n:.3f}, related {r:.3f}, distinct {d:.3f}"

    def test_default_grid_builds(self, catalog: Catalog) -> None:
        items = build_items(
            catalog,
            counts=[1, 4, 8, 16, 32, 64],
            similarities=["near", "related", "distinct"],
            seed=0,
            cues=["literal", "implied"],
        )
        assert len(items) == len(catalog.families) * (1 + 5 * 3) * 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> Any:
    from tool_eval_bench.cli.legacy_parser import make_parser
    from tool_eval_bench.cli.parser import parse_cli_args

    _, args = parse_cli_args(make_parser, argv)
    return args


class TestCliWiring:
    def test_registered(self) -> None:
        assert "toolscale" in available_plugins()
        assert get_plugin("toolscale").name == "toolscale"

    def test_flat_flags_and_defaults(self) -> None:
        args = _parse(["--toolscale-only"])
        assert args.toolscale_only is True
        assert args.toolscale_counts == "1,4,8,16,32,64"
        assert args.toolscale_similarity == "near,related,distinct"
        assert args.toolscale_cues == "literal,implied"
        assert args.toolscale_fillers == "sparse"
        assert args.toolscale_limit == 0
        assert args.toolscale_catalog is None

    def test_plugin_subcommand_translates_its_options(self) -> None:
        args = _parse(
            [
                "plugin",
                "toolscale",
                "--counts",
                "1,4",
                "--similarity",
                "near",
                "--cues",
                "implied",
                "--fillers",
                "sparse,crowded",
                "--limit",
                "3",
                "--catalog",
                "private.json",
                "--trials",
                "2",
            ]
        )
        assert args.toolscale_only is True
        assert (args.toolscale_counts, args.toolscale_similarity) == ("1,4", "near")
        assert args.toolscale_cues == "implied"
        assert args.toolscale_fillers == "sparse,crowded"
        assert (args.toolscale_limit, args.toolscale_catalog, args.trials) == (
            3,
            "private.json",
            2,
        )

    def test_toolscale_options_are_rejected_for_other_plugins(self) -> None:
        with pytest.raises(SystemExit):
            _parse(["plugin", "gsm8k", "--counts", "1,4"])

    def test_dispatch_runs_toolscale_and_stops(self) -> None:
        from rich.console import Console

        from tool_eval_bench.cli.plugin_runners import run_selected_plugins

        calls: list[str] = []
        stopped = run_selected_plugins(
            Console(),
            "model",
            "display",
            "http://server",
            None,
            _parse(["--toolscale-only"]),
            runners={"toolscale": lambda *a, **k: calls.append("toolscale")},
            extra_params=None,
            output_dir=None,
            run_context=None,
        )
        assert (calls, stopped) == (["toolscale"], True)


def _runner_args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "toolscale_counts": "1,4,8",
        "toolscale_similarity": "near,distinct",
        "toolscale_cues": "literal,implied",
        "toolscale_fillers": "sparse,crowded",
        "toolscale_descriptions": "clean",
        "toolscale_limit": 4,
        "toolscale_catalog": None,
        "trials": 1,
        "seed": 1,
        "parallel": 2,
        "temperature": 0.0,
        "timeout": 5.0,
        "label": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_console_prints_each_slice_at_its_own_largest_count() -> None:
    from rich.console import Console

    from tool_eval_bench.cli.plugin_runners import _print_toolscale_effects

    def effect(count: int, density: str) -> dict[str, Any]:
        return {
            "cue": "literal",
            "density": density,
            "descriptions": "clean",
            "versus": "implied",
            "count": count,
            "similarity": "near",
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "targets": 4,
        }

    console = Console(record=True, width=200)
    _print_toolscale_effects(
        console,
        {
            "cues": ["literal", "implied"],
            "densities": ["sparse", "crowded"],
            "descriptions": ["clean"],
            "cue_effects": [effect(16, "sparse"), effect(64, "sparse"), effect(128, "crowded")],
        },
    )
    output = console.export_text()
    assert "sparse, near, N=64" in output and "crowded, near, N=128" in output
    assert "N=16" not in output


def test_timeout_stretches_for_long_tool_lists() -> None:
    from tool_eval_bench.cli.plugin_runners import _toolscale_timeout

    assert _toolscale_timeout(300.0, 64) == 300.0  # an explicit longer timeout wins
    assert _toolscale_timeout(5.0, 64) > 120.0
    assert _toolscale_timeout(5.0, 248) > _toolscale_timeout(5.0, 64)


class TestToolscaleRunner:
    def _run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: argparse.Namespace
    ) -> tuple[str, list[dict]]:
        from rich.console import Console

        from tool_eval_bench.cli import plugin_runners

        monkeypatch.setattr(
            "tool_eval_bench.adapters.factory.build_adapter", lambda *a, **k: _StubAdapter()
        )
        monkeypatch.setattr(
            "tool_eval_bench.application.mode_runs.with_config_fingerprint", lambda value, *_: value
        )
        persisted: list[dict] = []
        monkeypatch.setattr("tool_eval_bench.application.run_queries.persist_run", persisted.append)
        console = Console(record=True, width=200)
        plugin_runners._run_toolscale_benchmark(
            console, "m", "Display", "http://x", None, args, output_dir=str(tmp_path)
        )
        return console.export_text(), persisted

    def test_private_catalog_run_persists_and_withholds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "private.json"
        path.write_text(json.dumps(_catalog_data()))
        output, persisted = self._run(
            monkeypatch, tmp_path, _runner_args(toolscale_catalog=str(path))
        )
        assert [p["run_type"] for p in persisted] == ["toolscale"]
        config = persisted[0]["config"]
        assert config["catalog_source"] == "custom"
        assert len(config["catalog_sha256"]) == 64
        assert config["counts"] == [1, 4, 8]
        assert config["cues"] == ["literal", "implied"]
        assert config["densities"] == ["sparse", "crowded"]
        assert "Accuracy when called" in output and "Call rate" in output
        assert "Accuracy when called / call rate (%)" in output
        reports = list(tmp_path.rglob("*.md"))
        assert reports, "a Markdown report is written"
        text = reports[0].read_text()
        assert "Toolscale" in text
        assert "op0_dom0_alpha" not in text

    def test_large_lists_warn_about_provider_tool_caps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_eval_bench.cli import plugin_runners

        monkeypatch.setattr(plugin_runners, "_OPENAI_TOOL_CAP", 4)
        path = tmp_path / "private.json"
        path.write_text(json.dumps(_catalog_data()))
        output, _ = self._run(monkeypatch, tmp_path, _runner_args(toolscale_catalog=str(path)))
        assert "some providers cap tools per request" in output

    def test_crowded_grid_without_cells_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "private.json"
        path.write_text(json.dumps(_catalog_data()))
        with pytest.raises(SystemExit):
            self._run(
                monkeypatch,
                tmp_path,
                _runner_args(
                    toolscale_catalog=str(path), toolscale_counts="4", toolscale_fillers="crowded"
                ),
            )

    def test_bad_counts_stop_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "private.json"
        path.write_text(json.dumps(_catalog_data()))
        with pytest.raises(SystemExit):
            self._run(
                monkeypatch,
                tmp_path,
                _runner_args(toolscale_catalog=str(path), toolscale_counts="1,2"),
            )
