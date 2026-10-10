"""CardData fixtures for the card renderer tests and visual review."""

from __future__ import annotations

from dataclasses import replace

from tool_eval_bench.domain.card import (
    AccuracyResult,
    CardData,
    CategoryScore,
    NeedleCell,
    NeedleSummary,
    PerfPoint,
    ToolEvalSummary,
)

CATEGORY_NAMES = (
    ("A", "Tool Selection"),
    ("B", "Parameter Precision"),
    ("C", "Multi-Step Chains"),
    ("D", "Restraint & Refusal"),
    ("E", "Error Recovery"),
    ("F", "Localization"),
    ("G", "Structured Reasoning"),
    ("H", "Instruction Following"),
    ("I", "Context & State Tracking"),
    ("J", "Code-Specific Patterns"),
    ("K", "Safety & Boundaries"),
    ("L", "Toolset Scale"),
    ("M", "Autonomous Planning"),
    ("N", "Creative Composition"),
    ("O", "Structured Output"),
    ("P", "Hard Mode"),
)

K = 1024


def _categories(percents: tuple[float, ...]) -> tuple[CategoryScore, ...]:
    return tuple(
        CategoryScore(code, name, pct)
        for (code, name), pct in zip(CATEGORY_NAMES, percents, strict=False)
    )


def _tool_eval(*, hardmode: bool = True, capped: bool = False, trials: int = 1) -> ToolEvalSummary:
    percents = (100, 92, 88, 83, 75, 100, 90, 86, 71, 94, 79, 62, 68, 80, 97, 41)
    cats = _categories(percents if hardmode else percents[:15])
    return ToolEvalSummary(
        score=86.4,
        rating="★★★ Adequate (safety-capped)" if capped else "★★★★ Good",
        passed=71 if hardmode else 56,
        partial=14 if hardmode else 8,
        failed=12 if hardmode else 5,
        total=97 if hardmode else 69,
        categories=cats,
        safety_capped=capped,
        hardmode=hardmode,
        trials=trials,
        suite_size=97 if hardmode else 69,
    )


def _needle_grid(
    contexts: tuple[int, ...],
    misses: set[tuple[int, float]],
    errored: frozenset[tuple[int, float]] = frozenset(),
    depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> NeedleSummary:
    """Every cell, errored ones included, the way card data builds the grid."""
    cells = tuple(
        NeedleCell(ctx, d, (ctx, d) not in misses and (ctx, d) not in errored, (ctx, d) in errored)
        for ctx in contexts
        for d in depths
    )
    answered = [c for c in cells if not c.error]
    found = sum(c.found for c in answered)
    effective = None
    for ctx in contexts:
        row = [c for c in answered if c.context_tokens == ctx]
        if row and all(c.found for c in row):
            effective = ctx
        else:
            break
    return NeedleSummary(
        cells=cells,
        accuracy=found / len(answered) if answered else 0.0,
        effective_context=effective,
        errors=len(errored),
    )


def full_long() -> CardData:
    """(a) A long-preset session with every section."""
    depth_rows = (
        (0, 2895.0, 151.2, 712.0, None),
        (4 * K, 2810.0, 147.9, 1490.0, 182.0),
        (8 * K, 2702.0, 145.1, 3060.0, 190.0),
        (16 * K, 2540.0, 139.6, 6510.0, 205.0),
        (32 * K, 2290.0, 131.0, 14300.0, 231.0),
        (64 * K, 1910.0, 118.4, 34200.0, 288.0),
        (128 * K, 1420.0, 102.7, 92300.0, 402.0),
    )
    perf = [
        PerfPoint(
            depth=d,
            concurrency=1,
            pp_tps=pp,
            tg_tps=tg,
            tg_tps_total=tg,
            ttft_ms=ttft,
            cached_ttft_ms=cached,
        )
        for d, pp, tg, ttft, cached in depth_rows
    ]
    for c, per in ((2, 139.0), (4, 121.5), (8, 98.2), (16, 71.4), (32, 46.9)):
        perf.append(
            PerfPoint(
                depth=0,
                concurrency=c,
                pp_tps=2895.0,
                tg_tps=per,
                tg_tps_total=per * c,
                ttft_ms=700.0 + 40 * c,
            )
        )
    needle = _needle_grid(
        (8 * K, 16 * K, 32 * K, 64 * K, 96 * K, 128 * K), {(128 * K, 0.5), (128 * K, 0.75)}
    )
    return CardData(
        model="Qwen3-Coder-30B-A3B-Instruct-FP8",
        tool_version="1.14.0",
        created_at="2026-10-10T13:42:07Z",
        engine="vLLM",
        engine_version="0.11.0",
        quantization="FP8",
        hardware="RTX PRO 6000 Blackwell 96GB",
        max_context=262144,
        perf_preset="long",
        perf_pp=2048,
        perf_tg=128,
        perf=tuple(perf),
        tool_eval=_tool_eval(),
        needle=needle,
        accuracy=(
            AccuracyResult("GSM8K", 0.921),
            AccuracyResult("MMLU", 0.712),
            AccuracyResult("IFEval", 0.843),
        ),
        seed=42,
        fingerprint="a1b2c3d4e5f6",
    )


def tool_eval_only() -> CardData:
    """(b) Only tool-calling scores."""
    return CardData(
        model="gemma-3-27b-it",
        tool_version="1.14.0",
        created_at="2026-10-09T08:00:00Z",
        engine="llama.cpp",
        engine_version="b6750",
        quantization="Q4_K_M",
        hardware=None,
        max_context=131072,
        tool_eval=_tool_eval(hardmode=False, trials=3),
        seed=7,
        fingerprint="0f9e8d7c6b5a",
    )


def perf_short() -> CardData:
    """(c) Perf only, short preset: depths stop at 8K."""
    perf = [
        PerfPoint(0, 1, 5120.0, 61.4, 61.4, 402.0),
        PerfPoint(4 * K, 1, 4980.0, 59.8, 59.8, 1210.0),
        PerfPoint(8 * K, 1, 4790.0, 58.1, 58.1, 2050.0),
        PerfPoint(0, 2, 5120.0, 57.0, 114.0, 455.0),
        PerfPoint(0, 4, 5120.0, 49.6, 198.4, 530.0),
    ]
    return CardData(
        model="meta-llama/Llama-3.3-70B-Instruct",
        tool_version="1.14.0",
        created_at="2026-10-08T21:10:00Z",
        engine="SGLang",
        engine_version="0.5.2",
        quantization="AWQ",
        hardware="2x RTX 4090",
        max_context=131072,
        perf_preset="short",
        perf_pp=2048,
        perf_tg=128,
        perf=tuple(perf),
        fingerprint="77aa88bb99cc",
    )


def extreme() -> CardData:
    """(d) Extreme values and a 120-character model name."""
    name = ("org-with-a-long-name/" + "Mega-Model-" * 10)[:120]
    perf = [
        PerfPoint(0, 1, 98_000.0, 9_000.0, 9_000.0, 40.0, 12.0, pp_estimated=True),
        PerfPoint(64 * K, 1, 21_000.0, 610.0, 610.0, 3_100.0, 45.0),
        PerfPoint(256 * K, 1, 2_400.0, 31.0, 31.0, 61_000.0, 380.0),
        PerfPoint(1024 * K, 1, 260.0, 2.0, 2.0, 400_000.0, 2_900.0),
        PerfPoint(0, 64, 98_000.0, 140.0, 8_960.0, 900.0),
        PerfPoint(0, 256, 98_000.0, 30.0, 7_680.0, 4_000.0),
    ]
    return CardData(
        model=name,
        tool_version="2.0.0.dev41+g1234abcd",
        created_at="2026-10-10T00:00:00+00:00",
        engine="vLLM",
        engine_version="0.12.0rc1.dev999+gdeadbeef",
        quantization="NVFP4 with a very long quantization description",
        hardware="8x H200 SXM 141GB in a rack somewhere with a long free text label",
        max_context=1024 * K,
        perf_preset="long",
        perf_pp=32768,
        perf_tg=4096,
        perf=tuple(perf),
        tool_eval=ToolEvalSummary(
            score=58.0,
            rating="★★ Weak (safety-capped)",
            passed=30,
            partial=10,
            failed=29,
            total=69,
            categories=_categories((100, 0, 55, 12, 33, 100, 50, 61, 70, 90, 20, 5, 44, 66, 99)),
            safety_capped=True,
            suite_size=69,
        ),
        needle=NeedleSummary(cells=(), accuracy=0.94, effective_context=96_000, errors=1),
        accuracy=(AccuracyResult("GSM8K", 1.0),),
        seed=2**31 - 1,
        fingerprint="ffffffffffffffffffffffff",
    )


def needle_misses() -> CardData:
    """(e) Needle misses at long context, one perf point, no tool eval."""
    contexts = (4 * K, 8 * K, 16 * K, 32 * K, 64 * K, 128 * K, 256 * K, 512 * K, 1024 * K)
    misses = {(256 * K, 0.5), (512 * K, 0.25), (512 * K, 0.5), (512 * K, 0.75)}
    misses |= {(1024 * K, d) for d in (0.0, 0.5, 0.75, 1.0)}
    errored = frozenset({(1024 * K, 0.25), (512 * K, 1.0)})
    return CardData(
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        tool_version="1.14.0",
        created_at="2026-10-07T10:00:00Z",
        engine="llama.cpp",
        quantization="Q8_0",
        max_context=1024 * K,
        perf=(PerfPoint(0, 1, 1830.0, 38.2, 38.2, 840.0),),
        needle=_needle_grid(contexts, misses, errored),
        accuracy=(AccuracyResult("MMLU", 0.69), AccuracyResult("IFEval", 0.81)),
    )


def real_sparse() -> CardData:
    """(f) Shaped like a real quick session: a 3-scenario subset, accuracy, nothing else."""
    return CardData(
        model="Step-3.7-Flash-NVFP4",
        tool_version="2.0.6",
        created_at="2026-06-08T09:12:00Z",
        engine="vLLM",
        engine_version="0.21.1rc1.dev292+g97e4022c6.d20260526",
        hardware="DGX Spark",
        max_context=256 * K,
        tool_eval=ToolEvalSummary(
            score=100.0,
            rating="★★★★★ Excellent",
            passed=3,
            partial=0,
            failed=0,
            total=3,
            categories=(CategoryScore("A", "Tool Selection", 100.0),),
            suite_size=69,
        ),
        accuracy=(
            AccuracyResult("GSM8K", 0.70, "10 of 1,319 q", items=10),
            AccuracyResult("MMLU", 0.25, "28 of 14,042 q", errors=24, items=28),
            AccuracyResult("IFEval", 0.0, "3 of 541 prompts", items=3),
        ),
        fingerprint="5e1f0c3a7b",
    )


def _long_perf(
    depths: tuple[int, ...], *, level: int = 1, estimated: bool = False
) -> list[PerfPoint]:
    out = []
    for i, d in enumerate(depths):
        tg = 120.0 - 6 * i
        out.append(
            PerfPoint(
                d, level, 3000.0 - 150 * i, tg, tg * level, 400.0 + d / 20, tg_estimated=estimated
            )
        )
    return out


def interrupted_long() -> CardData:
    """A long preset stopped at 16K by an interrupted depth pass, with failed cells."""
    base = full_long()
    return replace(
        base,
        perf=tuple(_long_perf((0, 4 * K, 8 * K, 16 * K))),
        needle=None,
        perf_incomplete=True,
        perf_failed_cells=3,
    )


def window_capped_long() -> CardData:
    """A long preset that finished but stopped at 16K because the server window is 32K."""
    base = full_long()
    return replace(
        base,
        max_context=32 * K,
        perf=tuple(_long_perf((0, 16 * K))),
        needle=None,
    )


def window_capped_deep() -> CardData:
    """A long preset capped at 64K by a 96K window: deep enough to chart."""
    base = full_long()
    return replace(
        base,
        max_context=96 * K,
        perf=tuple(_long_perf((0, 4 * K, 8 * K, 16 * K, 32 * K, 64 * K))),
    )


def _subset(**changes: object) -> ToolEvalSummary:
    te = ToolEvalSummary(
        score=100.0,
        rating="★★★★★ Excellent",
        passed=3,
        partial=0,
        failed=0,
        total=3,
        categories=(CategoryScore("A", "Tool Selection", 100.0),),
        suite_size=69,
    )
    return replace(te, **changes)  # type: ignore[arg-type]


def subset_3_of_69() -> CardData:
    return replace(real_sparse(), accuracy=())


def hardmode_only() -> CardData:
    te = ToolEvalSummary(
        score=47.3,
        rating="★★ Weak",
        passed=10,
        partial=6,
        failed=12,
        total=28,
        categories=_categories((0,) * 15 + (47.3,))[15:],
        hardmode=True,
        hardmode_only=True,
        suite_size=28,
    )
    return replace(full_long(), tool_eval=te)


def scenario_pack() -> CardData:
    te = replace(
        _tool_eval(hardmode=False),
        scenario_pack=True,
        suite_size=None,
        total=40,
        passed=31,
        partial=5,
        failed=4,
    )
    return replace(tool_eval_only(), tool_eval=te)


def excluded_scenarios() -> CardData:
    te = replace(_tool_eval(hardmode=False), total=66, passed=54, excluded=3)
    return replace(tool_eval_only(), tool_eval=te)


def all_errored_needle() -> CardData:
    contexts = (8 * K, 16 * K, 32 * K)
    errored = frozenset((c, d) for c in contexts for d in (0.0, 0.25, 0.5, 0.75, 1.0))
    return replace(full_long(), needle=_needle_grid(contexts, set(), errored))


def cjk_model() -> CardData:
    return replace(
        full_long(), model="通义千问/Qwen3-235B-A22B-思考版-FP8", hardware="昇腾 910B ×8"
    )


def compact_smarts() -> CardData:
    """Long preset, needle, accuracy with caveats, no tool eval: the old overlap case."""
    base = full_long()
    return replace(
        base,
        tool_eval=None,
        accuracy=(
            AccuracyResult("GSM8K", 0.75, "4 of 1,319 q", errors=24, items=28),
            AccuracyResult("MMLU", 0.712, "limited to 200 q", items=200),
            AccuracyResult("IFEval", 0.843),
        ),
    )


def sparse_perf_accuracy() -> CardData:
    """Perf and accuracy only: the sparse landscape that used to be mostly placeholders."""
    return replace(compact_smarts(), needle=None)


def no_single_stream() -> CardData:
    """Concurrency 2 and up only, per-request decode derived from the aggregate."""
    perf = _long_perf((0, 8 * K, 32 * K, 128 * K), level=2, estimated=True)
    perf += [
        PerfPoint(0, c, 3000.0, 100.0 - c, (100.0 - c) * c, 500.0, tg_estimated=True)
        for c in (4, 8)
    ]
    return replace(full_long(), perf=tuple(perf))


def needle_single_cell() -> CardData:
    return replace(full_long(), needle=_needle_grid((32 * K,), set(), depths=(0.5,)))


def needle_eleven_depths() -> CardData:
    depths = tuple(i / 10 for i in range(11))
    return replace(
        full_long(), needle=_needle_grid((8 * K, 32 * K), {(32 * K, 0.5)}, depths=depths)
    )


def accuracy_only() -> CardData:
    return replace(compact_smarts(), perf=(), needle=None, perf_preset=None)


ALL = {
    "a_full_long": full_long,
    "b_tool_eval_only": tool_eval_only,
    "c_perf_short": perf_short,
    "d_extreme": extreme,
    "e_needle_misses": needle_misses,
    "f_real_sparse": real_sparse,
    "g_interrupted_long": interrupted_long,
    "h_window_capped_long": window_capped_long,
    "i_window_capped_deep": window_capped_deep,
    "j_subset_3_of_69": subset_3_of_69,
    "k_hardmode_only": hardmode_only,
    "l_scenario_pack": scenario_pack,
    "m_excluded": excluded_scenarios,
    "n_all_errored_needle": all_errored_needle,
    "o_cjk_model": cjk_model,
    "p_compact_smarts": compact_smarts,
    "q_sparse_perf_accuracy": sparse_perf_accuracy,
    "r_no_single_stream": no_single_stream,
    "s_needle_single_cell": needle_single_cell,
    "t_needle_eleven_depths": needle_eleven_depths,
    "u_accuracy_only": accuracy_only,
}
