"""Tests for the BenchmarkPlugin interface and registry."""

from __future__ import annotations

import pytest

from tool_eval_bench.domain.plugin import BenchmarkPlugin, BenchmarkResult


class TestBenchmarkResult:
    """Test BenchmarkResult serialization."""

    def test_to_dict(self):
        result = BenchmarkResult(
            plugin_name="test",
            score=85.5,
            score_label="85.5%",
            rating="★★★★ Good",
            details={"correct": 171, "total": 200},
            item_results=[{"index": 0, "correct": True}],
            metadata={"dataset": "test"},
            duration_seconds=123.456,
            total_tokens=50000,
        )
        d = result.to_dict()
        assert d["plugin_name"] == "test"
        assert d["score"] == 85.5
        assert d["rating"] == "★★★★ Good"
        assert d["duration_seconds"] == 123.46
        assert d["total_tokens"] == 50000

    def test_to_dict_defaults(self):
        result = BenchmarkResult(
            plugin_name="test",
            score=0.0,
            score_label="0%",
            rating="★ Poor",
        )
        d = result.to_dict()
        assert d["details"] == {}
        assert d["item_results"] == []
        assert d["duration_seconds"] == 0.0
        assert d["total_tokens"] == 0


class TestPluginRegistry:
    """Test the plugin registry."""

    def test_get_gsm8k_plugin(self):
        from tool_eval_bench.plugins.registry import get_plugin

        plugin = get_plugin("gsm8k")
        assert isinstance(plugin, BenchmarkPlugin)
        assert plugin.name == "gsm8k"
        assert len(plugin.description) > 0

    def test_unknown_plugin(self):
        from tool_eval_bench.plugins.registry import get_plugin

        with pytest.raises(KeyError, match="Unknown benchmark plugin"):
            get_plugin("nonexistent")

    def test_available_plugins(self):
        from tool_eval_bench.plugins.registry import available_plugins

        plugins = available_plugins()
        assert "gsm8k" in plugins
