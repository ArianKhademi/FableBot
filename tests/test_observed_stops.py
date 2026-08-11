"""Observed-price stop mode: trails ratchet off the engine's own polled price
stream (ordering known) and arm immediately; a stop moved mid-bar is tested
against observed closes only until the bar rolls."""

import numpy as np
import pandas as pd
import pytest

from fablebot.config import AccountConfig, Config, CorrelationConfig, InstrumentConfig, ScheduleConfig
from fablebot.engine import TradingEngine
from tests.conftest import make_ohlcv
from tests.test_engine import FakeFeed, SilentNotifier, noisy_flat

MR = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14,
      "atr_stop_mult": 2.0, "trail_atr_mult": 1.0}


def _engine(stop_mode="observed"):
    cfg = Config(
        account=AccountConfig(starting_equity=100_000.0, risk_per_trade=0.01, stop_mode=stop_mode),
        schedule=ScheduleConfig(),
        correlation=CorrelationConfig(groups={}, max_same_direction_per_group=2),
        instruments={"SPX": InstrumentConfig("SPX", "^GSPC", "S&P", "mean_reversion", "15m", MR)},
    )
    return TradingEngine(cfg, state_path=None, fetch=FakeFeed({}), notifier=SilentNotifier())


def _entry_frame():
    return make_ohlcv(list(noisy_flat(40, seed=1)) + [92.0], freq="15min")


def _same_bar(frame, close, high=None, low=None):
    """The same forming bar, observed later: same timestamps, updated last row."""
    out = frame.copy()
    out.iloc[-1, out.columns.get_loc("close")] = close
    out.iloc[-1, out.columns.get_loc("high")] = high if high is not None else max(
        float(out["high"].iloc[-1]), close)
    if low is not None:
        out.iloc[-1, out.columns.get_loc("low")] = low
    return out


def test_observed_trail_ratchets_intrabar_off_observed_close_not_bar_high():
    engine = _engine()
    inst = engine.config.instruments["SPX"]
    frame = _entry_frame()
    log = engine.evaluate_instrument(inst, frame)
    assert any("opened long" in l for l in log)
    stop0 = engine.broker.positions["SPX"].stop_price

    # same bar, later poll: observed close 96; the bar HIGH spikes to 120 but
    # that extreme was never observed post-entry — the trail must use 96.
    later = _same_bar(frame, close=96.0, high=120.0)
    log2 = engine.evaluate_instrument(inst, later)
    pos = engine.broker.positions["SPX"]
    assert any("trailed (observed)" in l for l in log2)
    assert pos.stop_price > stop0
    assert pos.stop_price < 96.0            # anchored to observed 96...
    assert pos.stop_price < 120.0 - 5.0     # ...not to the unobserved 120 spike


def test_stop_moved_this_bar_is_tested_against_observed_closes_only():
    engine = _engine()
    inst = engine.config.instruments["SPX"]
    frame = _entry_frame()
    engine.evaluate_instrument(inst, frame)
    engine.evaluate_instrument(inst, _same_bar(frame, close=96.0))   # trail up
    stop = engine.broker.positions["SPX"].stop_price
    assert stop > 92.0

    # same bar: LOW crashes far below the fresh stop, but the observed close
    # stays above it -> must survive (that low may predate the stop)
    engine.evaluate_instrument(inst, _same_bar(frame, close=stop + 0.3, low=60.0))
    assert "SPX" in engine.broker.positions

    # an OBSERVED close beyond the stop fires it
    log = engine.evaluate_instrument(inst, _same_bar(frame, close=stop - 0.2, low=60.0))
    assert any("STOPPED OUT" in l for l in log)
    assert "SPX" not in engine.broker.positions


def test_next_bar_full_range_tests_the_trailed_stop():
    engine = _engine()
    inst = engine.config.instruments["SPX"]
    frame = _entry_frame()
    engine.evaluate_instrument(inst, frame)
    engine.evaluate_instrument(inst, _same_bar(frame, close=96.0))   # trail armed
    stop = engine.broker.positions["SPX"].stop_price

    # a NEW bar whose low dips through the stop fires it even though its close
    # is back above — the whole new bar is post-stop, wicks count again.
    closes = list(noisy_flat(40, seed=1)) + [92.0, stop + 0.5]
    nxt = make_ohlcv(closes, freq="15min")
    nxt.iloc[-1, nxt.columns.get_loc("low")] = stop - 0.5
    log = engine.evaluate_instrument(inst, nxt)
    assert any("STOPPED OUT" in l for l in log)


def test_bars_mode_default_and_invalid_mode_rejected():
    assert _engine(stop_mode="bars").stop_mode == "bars"
    assert AccountConfig().stop_mode == "bars"          # default untouched
    with pytest.raises(ValueError):
        _engine(stop_mode="ticks")
