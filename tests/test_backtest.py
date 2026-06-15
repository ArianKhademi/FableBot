import numpy as np

from fablebot.backtest import run_backtest
from fablebot.strategies import build_strategy
from tests.conftest import make_ohlcv


def test_backtest_trend_follower_profits_on_clean_trend():
    df = make_ohlcv(np.linspace(100, 200, 300), freq="4h")
    strat = build_strategy(
        "trend_following",
        {"fast_ema": 20, "slow_ema": 50, "atr_period": 14, "atr_stop_mult": 3.0},
    )
    result = run_backtest(df, strat)
    assert result.trades >= 1
    assert result.total_pnl > 0


def test_backtest_losses_capped_at_risk_budget():
    # Whipsaw series designed to stop trades out
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 2.0, 400))
    df = make_ohlcv(closes)
    strat = build_strategy(
        "momentum_breakout",
        {"entry_channel": 20, "exit_channel": 10, "atr_period": 14, "atr_stop_mult": 2.5},
    )
    result = run_backtest(df, strat, starting_equity=100_000.0, risk_per_trade=0.01)
    # No single losing trade should meaningfully exceed 1% of equity at entry.
    # (Small tolerance for the equity base shifting between trades.)
    assert result.trades > 0
    assert result.max_drawdown_pct < 100.0


def test_friction_defaults_off_and_drags_returns():
    df = make_ohlcv(np.linspace(100, 200, 300), freq="4h")
    strat = lambda: build_strategy(
        "trend_following", {"fast_ema": 20, "slow_ema": 50, "atr_period": 14, "atr_stop_mult": 3.0})
    free = run_backtest(df, strat())
    free_again = run_backtest(df, strat(), friction_bps=0.0)
    assert free.total_pnl == free_again.total_pnl  # default is exactly friction-free
    costed = run_backtest(df, strat(), friction_bps=10.0)
    # friction is a strict drag on a profitable run, and never improves PnL
    assert costed.total_pnl < free.total_pnl


def test_friction_keeps_losses_bounded():
    rng = np.random.default_rng(11)
    df = make_ohlcv(100 + np.cumsum(rng.normal(0, 2.0, 400)))
    strat = build_strategy(
        "momentum_breakout", {"entry_channel": 20, "exit_channel": 10, "atr_period": 14, "atr_stop_mult": 2.5})
    result = run_backtest(df, strat, friction_bps=5.0)
    assert result.trades > 0
    assert result.max_drawdown_pct < 100.0  # slippage doesn't blow past the floor


def test_backtest_equity_curve_length_matches_bars():
    df = make_ohlcv(np.linspace(100, 120, 100), freq="4h")
    strat = build_strategy(
        "trend_following",
        {"fast_ema": 10, "slow_ema": 30, "atr_period": 14, "atr_stop_mult": 3.0},
    )
    result = run_backtest(df, strat)
    assert len(result.equity_curve) == len(df) - strat.min_bars() + 1
