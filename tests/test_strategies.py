import numpy as np

from fablebot.strategies import build_strategy
from fablebot.strategies.base import Action
from tests.conftest import make_ohlcv

MR_PARAMS = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14, "atr_stop_mult": 2.0}
MB_PARAMS = {"entry_channel": 20, "exit_channel": 10, "atr_period": 14, "atr_stop_mult": 2.5}
TF_PARAMS = {"fast_ema": 20, "slow_ema": 50, "atr_period": 14, "atr_stop_mult": 3.0}


def noisy_flat(n, level=100.0, amp=1.0, seed=7):
    rng = np.random.default_rng(seed)
    return level + rng.normal(0, amp, n)


# -- mean reversion ----------------------------------------------------------

def test_mean_reversion_fades_a_spike_up():
    strat = build_strategy("mean_reversion", MR_PARAMS)
    closes = list(noisy_flat(40)) + [108.0]  # sharp overextension up
    signal = strat.evaluate(make_ohlcv(closes, freq="15min"), None)
    assert signal.action is Action.OPEN_SHORT


def test_mean_reversion_fades_a_spike_down():
    strat = build_strategy("mean_reversion", MR_PARAMS)
    closes = list(noisy_flat(40)) + [92.0]
    signal = strat.evaluate(make_ohlcv(closes, freq="15min"), None)
    assert signal.action is Action.OPEN_LONG


def test_mean_reversion_exits_on_reversion():
    strat = build_strategy("mean_reversion", MR_PARAMS)
    closes = list(noisy_flat(40)) + [92.0, 100.0]  # snap back to the mean
    signal = strat.evaluate(make_ohlcv(closes, freq="15min"), "long")
    assert signal.action is Action.CLOSE


def test_mean_reversion_holds_inside_band():
    strat = build_strategy("mean_reversion", MR_PARAMS)
    signal = strat.evaluate(make_ohlcv(noisy_flat(40), freq="15min"), None)
    assert signal.action is Action.HOLD


def test_trend_filter_skips_fades_against_the_trend():
    """With trend_filter_ma set, don't short into an uptrend or buy into a
    downtrend; the same setups fire normally without the filter."""
    up = list(np.linspace(100, 130, 40)) + [140.0]     # uptrend + overbought spike
    down = list(np.linspace(130, 100, 40)) + [90.0]    # downtrend + oversold spike
    up_df, down_df = make_ohlcv(up, freq="15min"), make_ohlcv(down, freq="15min")

    plain = build_strategy("mean_reversion", MR_PARAMS)
    assert plain.evaluate(up_df, None).action is Action.OPEN_SHORT
    assert plain.evaluate(down_df, None).action is Action.OPEN_LONG

    filt = build_strategy("mean_reversion", {**MR_PARAMS, "trend_filter_ma": 30})
    assert filt.evaluate(up_df, None).action is Action.HOLD       # won't short an uptrend
    assert filt.evaluate(down_df, None).action is Action.HOLD     # won't buy a downtrend


# -- momentum breakout -------------------------------------------------------

def test_breakout_long_on_new_high():
    strat = build_strategy("momentum_breakout", MB_PARAMS)
    closes = list(noisy_flat(40)) + [110.0]  # clears the 20-bar channel + spread
    signal = strat.evaluate(make_ohlcv(closes), None)
    assert signal.action is Action.OPEN_LONG
    assert signal.atr > 0


def test_breakout_short_on_new_low():
    strat = build_strategy("momentum_breakout", MB_PARAMS)
    closes = list(noisy_flat(40)) + [90.0]
    signal = strat.evaluate(make_ohlcv(closes), None)
    assert signal.action is Action.OPEN_SHORT


def test_breakout_rides_then_exits_on_channel_break():
    strat = build_strategy("momentum_breakout", MB_PARAMS)
    rising = list(np.linspace(100, 130, 45))
    assert strat.evaluate(make_ohlcv(rising), "long").action is Action.HOLD
    # collapse below the 10-bar low ends the ride
    collapsed = rising + [110.0]
    assert strat.evaluate(make_ohlcv(collapsed), "long").action is Action.CLOSE


def test_breakout_holds_inside_channel():
    strat = build_strategy("momentum_breakout", MB_PARAMS)
    signal = strat.evaluate(make_ohlcv(noisy_flat(40)), None)
    assert signal.action is Action.HOLD


# -- trend following ---------------------------------------------------------

def test_trend_following_longs_an_uptrend():
    strat = build_strategy("trend_following", TF_PARAMS)
    df = make_ohlcv(np.linspace(100, 150, 80), freq="4h")
    assert strat.evaluate(df, None).action is Action.OPEN_LONG


def test_trend_following_shorts_a_downtrend():
    strat = build_strategy("trend_following", TF_PARAMS)
    df = make_ohlcv(np.linspace(150, 100, 80), freq="4h")
    assert strat.evaluate(df, None).action is Action.OPEN_SHORT


def test_trend_following_flips_on_cross():
    strat = build_strategy("trend_following", TF_PARAMS)
    up_then_down = list(np.linspace(100, 150, 80)) + list(np.linspace(150, 90, 60))
    df = make_ohlcv(up_then_down, freq="4h")
    assert strat.evaluate(df, "long").action is Action.CLOSE


def test_trend_following_holds_with_trend():
    strat = build_strategy("trend_following", TF_PARAMS)
    df = make_ohlcv(np.linspace(100, 150, 80), freq="4h")
    assert strat.evaluate(df, "long").action is Action.HOLD
