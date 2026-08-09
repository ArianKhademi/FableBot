"""Trailing stop behavior: stops ratchet toward price, never away."""

import numpy as np
import pytest

from fablebot.backtest import run_backtest
from fablebot.portfolio import PaperBroker, Position
from fablebot.strategies import build_strategy
from tests.conftest import make_ohlcv
from tests.test_engine import SilentNotifier, FakeFeed, noisy_flat
from fablebot.config import (
    AccountConfig,
    Config,
    CorrelationConfig,
    InstrumentConfig,
    ScheduleConfig,
)
from fablebot.engine import TradingEngine


def make_position(direction="long"):
    return Position(
        instrument="BTC",
        direction=direction,
        units=0.5,
        entry_price=60_000.0,
        stop_price=58_000.0 if direction == "long" else 62_000.0,
        risk_amount=1000.0,
        strategy="momentum_breakout",
        reason="test",
    )


def test_long_trail_ratchets_up_never_down():
    pos = make_position("long")
    # Price moves up: stop follows at 2.5 * ATR(400) behind the high
    assert pos.trail(high=62_000.0, low=61_000.0, atr=400.0, trail_mult=2.5)
    assert pos.stop_price == pytest.approx(61_000.0)
    # Price falls back: a worse candidate must never loosen the stop
    assert not pos.trail(high=60_500.0, low=59_800.0, atr=400.0, trail_mult=2.5)
    assert pos.stop_price == pytest.approx(61_000.0)


def test_short_trail_ratchets_down_never_up():
    pos = make_position("short")
    assert pos.trail(high=59_000.0, low=58_000.0, atr=400.0, trail_mult=2.5)
    assert pos.stop_price == pytest.approx(59_000.0)
    assert not pos.trail(high=60_000.0, low=59_500.0, atr=400.0, trail_mult=2.5)
    assert pos.stop_price == pytest.approx(59_000.0)


def test_trail_can_lock_in_profit_past_entry():
    pos = make_position("long")  # entry 60,000
    pos.trail(high=65_000.0, low=64_000.0, atr=400.0, trail_mult=2.5)
    assert pos.stop_price > pos.entry_price  # worst case is now a winner


def test_broker_persists_trailed_stop(tmp_path):
    state = tmp_path / "portfolio.json"
    broker = PaperBroker(100_000.0, state_path=state)
    broker.open_position(make_position())
    broker.trail_stop("BTC", high=62_000.0, low=61_000.0, atr=400.0, trail_mult=2.5)
    reloaded = PaperBroker(0.0, state_path=state)
    assert reloaded.positions["BTC"].stop_price == pytest.approx(61_000.0)


def test_backtest_trailing_protects_trend_profits():
    """A clean uptrend that collapses in one bar: the static stop gives back
    everything to the gap check; the trailing stop exits near the top."""
    closes = list(np.linspace(100, 200, 200)) + [110.0]
    df = make_ohlcv(closes, freq="4h")
    params = {"fast_ema": 20, "slow_ema": 50, "atr_period": 14, "atr_stop_mult": 3.0}

    static = run_backtest(df, build_strategy("trend_following", params))
    trailing = run_backtest(
        df, build_strategy("trend_following", {**params, "trail_atr_mult": 3.0})
    )
    assert trailing.total_pnl > static.total_pnl
    assert trailing.total_pnl > 0


def test_engine_trails_stop_and_journals_it():
    config = Config(
        account=AccountConfig(starting_equity=100_000.0, risk_per_trade=0.01),
        schedule=ScheduleConfig(),
        correlation=CorrelationConfig(groups={}, max_same_direction_per_group=2),
        instruments={
            "BTC": InstrumentConfig(
                "BTC",
                "BTC-USD",
                "Bitcoin",
                "momentum_breakout",
                "1h",
                {
                    "entry_channel": 20,
                    "exit_channel": 10,
                    "atr_period": 14,
                    "atr_stop_mult": 2.5,
                    "trail_atr_mult": 2.5,
                },
            ),
        },
    )
    base = list(noisy_flat(40, seed=4))
    frames = {"BTC-USD": make_ohlcv(base + [110.0])}
    engine = TradingEngine(config, state_path=None, fetch=FakeFeed(frames), notifier=SilentNotifier())

    engine.run_cycle()  # breakout entry
    entry_stop = engine.broker.positions["BTC"].stop_price
    # Entry bar must not trail — the stop is exactly where sizing placed it
    pos = engine.broker.positions["BTC"]
    assert pos.units * (pos.entry_price - entry_stop) == pytest.approx(1000.0)

    frames["BTC-USD"] = make_ohlcv(base + [110.0, 118.0])  # trend continues
    engine.run_cycle()
    # Deferred trailing: bar B computes the candidate but the stop must NOT
    # move while B is still the latest bar (its low may predate the stop).
    assert engine.broker.positions["BTC"].stop_price == entry_stop
    assert engine._pending_trail.get("BTC") is not None

    frames["BTC-USD"] = make_ohlcv(base + [110.0, 118.0, 119.0])  # next bar arrives
    engine.run_cycle()
    trailed_stop = engine.broker.positions["BTC"].stop_price
    assert trailed_stop > entry_stop

    trail_events = [
        e
        for e in engine.journal.read()
        if e["event"] == "decision" and e["action"] == "trail"
    ]
    assert len(trail_events) == 1
