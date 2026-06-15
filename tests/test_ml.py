"""ML signal models: the numpy gradient-boosted trees, leakage-free feature
engineering, the MLStrategy (incl. the exact precompute cache), and the
hardened-gate validation that keeps ML honest."""

import numpy as np
import pytest

from fablebot import dsl, ml
from fablebot.backtest import run_backtest
from fablebot.config import InstrumentConfig
from tests.conftest import make_ohlcv


def _df(n=900, seed=0, drift=0.12, noise=0.8):
    closes = 100 + np.cumsum(np.random.default_rng(seed).normal(drift, noise, n))
    return make_ohlcv(np.abs(closes) + 40, freq="1d")


def _inst():
    return InstrumentConfig(key="SPY", symbol="SPY", name="SPY", strategy="ml_gbt",
                            timeframe="1d", params={}, param_bounds={})


# ── gradient-boosted trees ───────────────────────────────────────────────────

def test_gbt_learns_a_nonlinear_function():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.where(X[:, 2] > 0, 1.0, -1.0) + 0.2 * rng.normal(0, 1, 300)
    gbt = ml.GradientBoostedTrees(n_estimators=60, max_depth=3, min_leaf=10, seed=1).fit(X, y)
    pred = gbt.predict(X)
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.6


def test_gbt_beats_the_mean_baseline():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (200, 2))
    y = X[:, 0] ** 2 + rng.normal(0, 0.1, 200)
    gbt = ml.GradientBoostedTrees(n_estimators=40, max_depth=3, seed=0).fit(X, y)
    sse_model = ((y - gbt.predict(X)) ** 2).sum()
    sse_mean = ((y - y.mean()) ** 2).sum()
    assert sse_model < 0.5 * sse_mean


# ── features / labels: no look-ahead ─────────────────────────────────────────

def test_features_use_only_past_data():
    df = _df(300, seed=3)
    full = ml.make_features(df)
    for k in (130, 200, 280):
        partial = ml.make_features(df.iloc[:k])
        for col in ("ret_5", "rsi", "ema_spread", "zscore", "macd_hist"):
            a, b = partial[col].iloc[-1], full[col].iloc[k - 1]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9


def test_labels_look_forward_and_drop_the_tail():
    df = _df(200, seed=1)
    lab = ml.make_labels(df, horizon=5)
    assert lab.tail(5).isna().all()  # no future data for the last `horizon` bars
    assert lab.iloc[:-5].notna().sum() > 0


# ── MLStrategy + the precompute cache ────────────────────────────────────────

def test_ml_strategy_runs_through_the_risk_engine():
    df = _df(900, seed=4)
    model = ml.MLModel(n_estimators=20, max_depth=2, seed=0)
    assert model.train(df.iloc[:675])
    oos = df.iloc[675:]
    result = run_backtest(oos, ml.MLStrategy(model).prepare(oos))
    assert result.trades >= 0  # sizing/stops handled by the broker, no exceptions


def test_prepare_cache_is_exact():
    """The O(n) precompute cache must give identical results to the O(n^2)
    expanding-window recompute — features are backward-looking."""
    df = _df(900, seed=5)
    model = ml.MLModel(n_estimators=20, max_depth=2, seed=0)
    assert model.train(df.iloc[:675])
    oos = df.iloc[675:]
    prepared = run_backtest(oos, ml.MLStrategy(model).prepare(oos))
    plain = run_backtest(oos, ml.MLStrategy(model))
    assert prepared.trades == plain.trades
    assert prepared.total_pnl == pytest.approx(plain.total_pnl)


def test_train_refuses_insufficient_data():
    df = _df(60, seed=0)
    assert ml.MLModel().train(df) is False


# ── walk-forward + hardened gate ─────────────────────────────────────────────

def test_ml_walk_forward_returns_trial():
    trial = ml.ml_walk_forward(_df(900, seed=6), _inst(), model_kwargs={"n_estimators": 20, "max_depth": 2})
    assert trial is not None
    assert trial.strategy == "ml_gbt"
    assert trial.oos_bars > 0 and not np.isnan(trial.oos_sharpe)
    assert len(trial.oos_equity_curve) > 0


def test_validate_ml_returns_full_verdict():
    df = _df(1200, seed=7)
    from fablebot.optimizer import split_search_holdout
    s_df, h_df = split_search_holdout(df)
    verdict = ml.validate_ml(
        s_df, h_df, _inst(), current_genome=dsl.seed_trend_following(),
        rng=np.random.default_rng(0), model_kwargs={"n_estimators": 15, "max_depth": 2},
    )
    assert set(verdict) >= {
        "qualified", "folds_won", "folds_valid", "deflated_sharpe",
        "holdout_candidate_sharpe", "reason",
    }
    assert isinstance(verdict["qualified"], bool)
