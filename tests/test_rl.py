"""RL agent: constrained tabular Q-learning, the direction-only RLStrategy
(sizing stays with the broker), the exact precompute cache, and the
hardened-gate validation."""

import numpy as np
import pytest

from fablebot import dsl, rl
from fablebot.backtest import run_backtest
from fablebot.config import InstrumentConfig
from fablebot.strategies.base import Action
from tests.conftest import make_ohlcv


def _uptrend(n=600, seed=0):
    closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0.25, 0.4, n))
    return make_ohlcv(np.abs(closes) + 20, freq="1d")


def _mixed(n=900, seed=1):
    closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0.05, 1.0, n))
    return make_ohlcv(np.abs(closes) + 40, freq="1d")


def _inst():
    return InstrumentConfig(key="SPY", symbol="SPY", name="SPY", strategy="rl_qlearn",
                            timeframe="1d", params={}, param_bounds={})


# ── state discretisation ─────────────────────────────────────────────────────

def test_discretise_buckets():
    assert rl.discretise(1.0, 70, 2.0, None) == (1, 2, 2, 0)
    assert rl.discretise(-1.0, 30, -2.0, "long") == (-1, 0, 0, 1)
    assert rl.discretise(0.0, 50, 0.0, "short") == (0, 1, 1, 2)
    assert rl.discretise(float("nan"), float("nan"), float("nan"), None) == (0, 1, 1, 0)


# ── learning ─────────────────────────────────────────────────────────────────

def test_agent_learns_to_be_long_in_an_uptrend():
    df = _uptrend(600, seed=0)
    agent = rl.QLearningAgent(epochs=30, epsilon=0.1, seed=0)
    assert agent.train(df.iloc[:450])
    oos = df.iloc[450:]
    res = run_backtest(oos, rl.RLStrategy(agent).prepare(oos))
    assert res.return_pct > 0  # learned to ride the trend long


def test_agent_train_refuses_thin_data():
    assert rl.QLearningAgent().train(_uptrend(40)) is False


# ── strategy contract: direction only, risk stays with the broker ───────────

def test_rl_strategy_emits_only_direction_signals():
    df = _mixed(700, seed=2)
    agent = rl.QLearningAgent(epochs=15, seed=0)
    assert agent.train(df.iloc[:525])
    strat = rl.RLStrategy(agent).prepare(df)
    actions = {strat.evaluate(df.iloc[: i + 1], None).action for i in range(strat.min_bars(), len(df))}
    assert actions <= {Action.OPEN_LONG, Action.OPEN_SHORT, Action.HOLD}
    # the strategy never sets units/stops itself — only the risk engine does
    res = run_backtest(df.iloc[525:], rl.RLStrategy(agent).prepare(df.iloc[525:]))
    assert res.trades >= 0


def test_prepare_cache_is_exact():
    df = _mixed(900, seed=3)
    agent = rl.QLearningAgent(epochs=12, seed=0)
    assert agent.train(df.iloc[:675])
    oos = df.iloc[675:]
    prepared = run_backtest(oos, rl.RLStrategy(agent).prepare(oos))
    plain = run_backtest(oos, rl.RLStrategy(agent))
    assert prepared.trades == plain.trades
    assert prepared.total_pnl == pytest.approx(plain.total_pnl)


# ── walk-forward + hardened gate ─────────────────────────────────────────────

def test_rl_walk_forward_returns_trial():
    trial = rl.rl_walk_forward(_mixed(900, seed=4), _inst(), agent_kwargs={"epochs": 12})
    assert trial is not None
    assert trial.strategy == "rl_qlearn"
    assert trial.oos_bars > 0 and not np.isnan(trial.oos_sharpe)


def test_validate_rl_returns_full_verdict():
    df = _mixed(1200, seed=5)
    from fablebot.optimizer import split_search_holdout
    s_df, h_df = split_search_holdout(df)
    verdict = rl.validate_rl(
        s_df, h_df, _inst(), current_genome=dsl.seed_trend_following(),
        rng=np.random.default_rng(0), agent_kwargs={"epochs": 10},
    )
    assert set(verdict) >= {"qualified", "folds_won", "deflated_sharpe", "holdout_candidate_sharpe", "reason"}
    assert isinstance(verdict["qualified"], bool)
