"""Cross-instrument genome insights: top-genome mining + aggregation."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("genome_insights", ROOT / "scripts" / "genome_insights.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_top_genomes_sorts_and_filters_non_finite():
    m = _mod()
    hist = [
        {"genome": {"name": "a"}, "fitness": 1.0},
        {"genome": {"name": "b"}, "fitness": 3.0},
        {"genome": {"name": "c"}, "fitness": float("nan")},   # dropped
        {"genome": {"name": "d"}, "fitness": 2.0},
    ]
    assert [g["name"] for g, _ in m.top_genomes(hist, 2)] == ["b", "d"]


def test_feature_kinds_and_trailing():
    m = _mod()
    g = {"features": [{"kind": "rsi"}, {"kind": "bb_upper"}, {"id": "x"}],
         "risk": {"atr_stop_mult": 2.0, "trail_atr_mult": 3.0}}
    assert m.feature_kinds(g) == {"rsi", "bb_upper"}
    assert m.has_trail(g) is True
    assert m.has_trail({"risk": {"atr_stop_mult": 2.0}}) is False


def test_summarize_aggregates_across_instruments():
    m = _mod()
    g_rsi = {"features": [{"kind": "rsi"}], "risk": {"atr_stop_mult": 2.0}, "entry_long": {"type": "cmp"}}
    g_bb = {"features": [{"kind": "bb_upper"}], "risk": {"atr_stop_mult": 3.0, "trail_atr_mult": 2.0},
            "entry_long": {"type": "cross"}}
    by = {
        "AAA": [{"genome": g_rsi, "fitness": 2.0}, {"genome": g_bb, "fitness": 1.0}],
        "BBB": [{"genome": g_rsi, "fitness": 3.0}],
    }
    s = m.summarize(by, top_k=5)
    assert s["n_winners"] == 3 and s["n_instruments"] == 2
    rsi = next(i for i in s["indicators"] if i["kind"] == "rsi")
    assert rsi["in_winners"] == 2 and rsi["instruments"] == 2     # recurs across both
    assert round(s["trailing_pct"]) == 33                          # 1 of 3 winners trails
    assert s["champions"]["AAA"][1] == 2.0                         # best AAA fitness
