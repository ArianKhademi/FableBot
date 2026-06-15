"""Composable strategy DSL: indicator palette, condition evaluator, genome
strategies, and the genome operators that define the search space."""

import numpy as np
import pandas as pd
import pytest

from fablebot import dsl, indicators as ind
from fablebot.backtest import run_backtest
from fablebot.config import InstrumentConfig
from fablebot.optimizer import run_walk_forward
from fablebot.strategies import build_strategy
from tests.conftest import make_ohlcv


def mixed_df(n=1200, seed=0):
    closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0.05, 1.2, n))
    return make_ohlcv(np.abs(closes) + 30, freq="1h")


# ── extended indicators ──────────────────────────────────────────────────────

def test_rsi_bounded_and_directional():
    up = make_ohlcv(np.linspace(100, 200, 200), freq="1h")
    r = ind.rsi(up["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 70  # a relentless uptrend is overbought


def test_macd_shapes_and_sign():
    df = make_ohlcv(np.linspace(100, 160, 300), freq="1h")
    line, signal, hist = ind.macd(df["close"])
    assert len(line) == len(signal) == len(hist) == len(df)
    assert line.dropna().iloc[-1] > 0  # uptrend -> positive MACD


def test_adx_components_in_range():
    df = mixed_df(400, seed=3)
    adx_, plus_di, minus_di = ind.adx(df, 14)
    a = adx_.dropna()
    assert (a >= 0).all() and (a <= 100).all()
    assert (plus_di.dropna() >= 0).all() and (minus_di.dropna() >= 0).all()


def test_bollinger_and_keltner_band_order():
    df = mixed_df(300, seed=1)
    bu, bm, bl = ind.bollinger(df["close"], 20, 2.0)
    assert (bu.dropna() >= bm.dropna()).all() and (bm.dropna() >= bl.dropna()).all()
    ku, km, kl = ind.keltner(df, 20, 2.0)
    valid = ku.notna() & kl.notna()
    assert (ku[valid] >= kl[valid]).all()


def test_volume_zscore_handles_missing_volume():
    df = mixed_df(120).copy()
    df["volume"] = np.linspace(1000, 5000, len(df)) + np.random.default_rng(0).normal(0, 100, len(df))
    vz = ind.volume_zscore(df, 20).dropna()
    assert len(vz) > 0 and vz.abs().max() < 50  # finite, sane on varying volume
    no_vol = df.drop(columns=["volume"])
    assert ind.volume_zscore(no_vol, 20).isna().all()


# ── condition evaluator ──────────────────────────────────────────────────────

def _feats(**series):
    return {k: pd.Series(v, dtype=float) for k, v in series.items()}


def test_cmp_operators():
    f = _feats(a=[1, 2, 3], b=[3, 2, 1])
    assert dsl.eval_condition({"type": "cmp", "op": "gt", "left": {"feature": "a"}, "right": {"feature": "b"}}, f)
    assert not dsl.eval_condition({"type": "cmp", "op": "lt", "left": {"feature": "a"}, "right": {"feature": "b"}}, f)
    assert dsl.eval_condition({"type": "cmp", "op": "gte", "left": {"feature": "a"}, "right": {"const": 3}}, f)


def test_cross_detection_uses_two_bars():
    f = _feats(a=[1.0, 5.0], b=[3.0, 3.0])  # a was below b, now above
    assert dsl.eval_condition({"type": "cross", "dir": "above", "left": {"feature": "a"}, "right": {"feature": "b"}}, f)
    assert not dsl.eval_condition({"type": "cross", "dir": "below", "left": {"feature": "a"}, "right": {"feature": "b"}}, f)


def test_boolean_combinators_and_nan_safety():
    f = _feats(a=[1, 2, 10], b=[0, 0, 5])
    node = {"type": "and", "nodes": [
        {"type": "cmp", "op": "gt", "left": {"feature": "a"}, "right": {"feature": "b"}},
        {"type": "not", "node": {"type": "cmp", "op": "lt", "left": {"feature": "a"}, "right": {"const": 0}}},
    ]}
    assert dsl.eval_condition(node, f)
    # missing feature -> comparison is False, never an exception
    assert not dsl.eval_condition({"type": "cmp", "op": "gt", "left": {"feature": "missing"}, "right": {"const": 0}}, f)
    assert not dsl.eval_condition(None, f)


# ── seed genomes reproduce the hand-written strategies exactly ───────────────

@pytest.mark.parametrize("name,params,genome", [
    ("mean_reversion", {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14, "atr_stop_mult": 2.0},
     dsl.seed_mean_reversion()),
    ("trend_following", {"fast_ema": 20, "slow_ema": 50, "atr_period": 14, "atr_stop_mult": 3.0, "trail_atr_mult": 3.0},
     dsl.seed_trend_following()),
    ("momentum_breakout", {"entry_channel": 20, "exit_channel": 10, "atr_period": 14, "atr_stop_mult": 2.5, "trail_atr_mult": 2.5},
     dsl.seed_momentum_breakout()),
])
def test_seed_genome_matches_handwritten_strategy(name, params, genome):
    df = mixed_df(800, seed=0)
    hard = run_backtest(df, build_strategy(name, params))
    gen = run_backtest(df, build_strategy(name, {}, genome=genome))
    assert gen.trades == hard.trades
    assert gen.total_pnl == pytest.approx(hard.total_pnl)
    assert gen.return_pct == pytest.approx(hard.return_pct)


# ── any genome on any symbol, through the unchanged risk engine ──────────────

def test_sampled_genomes_are_valid_and_runnable():
    rng = np.random.default_rng(7)
    df = mixed_df(450, seed=2)
    archetypes_seen = set()
    for _ in range(16):
        g = dsl.sample_genome(rng)
        dsl.validate_genome(g)  # never raises
        archetypes_seen.add(g["name"])
        result = run_backtest(df, build_strategy("genome", {}, genome=g))
        assert result.trades >= 0  # runs cleanly on an arbitrary symbol
    assert len(archetypes_seen) >= 4  # the palette is actually diverse


def test_mutation_and_crossover_stay_valid():
    rng = np.random.default_rng(5)
    df = mixed_df(400, seed=4)
    for _ in range(10):
        a, b = dsl.sample_genome(rng), dsl.sample_genome(rng)
        child = dsl.crossover_genome(a, b, rng)
        mutant = dsl.mutate_genome(child, rng)
        for g in (child, mutant):
            dsl.validate_genome(g)
            run_backtest(df, build_strategy("genome", {}, genome=g))  # no exceptions


def test_repair_keeps_fast_below_slow():
    g = dsl.seed_trend_following(fast_ema=20, slow_ema=50)
    by_id = {s["id"]: s for s in g["features"]}
    by_id["fast"]["params"]["period"] = 80  # break the invariant
    dsl._repair(g)
    assert by_id["fast"]["params"]["period"] < by_id["slow"]["params"]["period"]


def test_genome_param_paths_roundtrip():
    g = dsl.seed_mean_reversion(lookback=20, entry_z=2.0)
    paths = dsl.genome_param_paths(g)
    assert "feat.z.lookback" in paths and "risk.atr_stop_mult" in paths
    g2 = dsl.apply_genome_params(g, {"feat.z.lookback": 35})
    assert g2["features"][0]["params"]["lookback"] == 35
    assert g["features"][0]["params"]["lookback"] == 20  # original untouched


def test_genome_runs_through_optimizer_walk_forward():
    df = mixed_df(600, seed=6)
    inst = InstrumentConfig(
        key="ANY", symbol="ANY", name="Any", strategy="genome", timeframe="1h",
        params={}, param_bounds={}, genome=dsl.seed_momentum_breakout(),
    )
    trial = run_walk_forward(df, inst, {})
    assert trial is not None
    assert trial.oos_bars > 0
    assert not np.isnan(trial.oos_sharpe)
