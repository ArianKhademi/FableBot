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


def test_three_zone_mid_band_blocks_new_entries():
    """With adx_momentum_threshold set, the ADX band between the two thresholds
    takes no NEW positions (live: 9/9 momentum entries at ADX 25-32 lost) but a
    held position is still managed to its exit."""
    rs = RegimeSwitch({**PARAMS, "adx_momentum_threshold": 40.0})
    dip = make_ohlcv(list(100 + np.random.default_rng(3).normal(0, 1, 40)) + [92.0], freq="1h")
    rs._adx = lambda df: 30.0                      # pin ADX into the mid band
    sig = rs.evaluate(dip, None)
    assert not sig.opens and "[mid" in sig.reason  # entry suppressed
    # a held long still gets exit management (MR closes it at the mean)
    reverted = make_ohlcv(list(100 + np.random.default_rng(3).normal(0, 1, 40)) + [92.0, 100.0], freq="1h")
    sig2 = rs.evaluate(reverted, "long")
    assert sig2.action.value == "close"


def test_three_zone_outer_bands_route_normally():
    rs = RegimeSwitch({**PARAMS, "adx_momentum_threshold": 40.0})
    frame = _range()
    rs._adx = lambda df: 20.0
    assert "[range" in rs.evaluate(frame, None).reason      # below 25 -> MR
    rs._adx = lambda df: 55.0
    assert "[trend" in rs.evaluate(frame, None).reason      # >= 40 -> momentum


def test_two_zone_default_unchanged():
    """Without adx_momentum_threshold, ADX 30 still routes to momentum (old behavior)."""
    rs = RegimeSwitch(PARAMS)
    rs._adx = lambda df: 30.0
    assert "[trend" in rs.evaluate(_range(), None).reason
