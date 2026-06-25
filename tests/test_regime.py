"""Regime switch: ADX routes to momentum in a trend, mean-reversion in a range —
both the existing sub-strategies, just selected by trend strength."""

import numpy as np

from fablebot.strategies import build_strategy
from fablebot.strategies.mean_reversion import MeanReversion
from fablebot.strategies.momentum_breakout import MomentumBreakout
from fablebot.strategies.regime import RegimeSwitch
from tests.conftest import make_ohlcv

PARAMS = {
    "lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14, "atr_stop_mult": 2.0,
    "entry_channel": 20, "exit_channel": 10, "adx_period": 14, "adx_threshold": 25.0,
}


def _strong_trend(n=120):
    return make_ohlcv(np.linspace(100, 170, n), freq="1h")          # steady up -> high ADX


def _range(n=160):
    return make_ohlcv(100 + 3 * np.sin(np.linspace(0, 14 * np.pi, n)), freq="1h")  # chop -> low ADX


def test_registered_in_the_factory():
    assert isinstance(build_strategy("regime", PARAMS), RegimeSwitch)


def test_trend_routes_to_momentum():
    rs = RegimeSwitch(PARAMS)
    strat, adx_val, trending = rs._active(_strong_trend())
    assert trending and isinstance(strat, MomentumBreakout)
    assert "[trend" in rs.evaluate(_strong_trend(), None).reason


def test_range_routes_to_mean_reversion():
    rs = RegimeSwitch(PARAMS)
    strat, adx_val, trending = rs._active(_range())
    assert not trending and isinstance(strat, MeanReversion)
    assert "[range" in rs.evaluate(_range(), None).reason


def test_context_reports_regime_and_adx():
    ctx = RegimeSwitch(PARAMS).context(_strong_trend())
    assert ctx.get("regime") in ("trend", "range")
    assert isinstance(ctx.get("adx"), float)
