"""End-to-end engine tests with a fake data feed — no network."""

import numpy as np
import pandas as pd
import pytest

from fablebot.config import (
    AccountConfig,
    Config,
    CorrelationConfig,
    InstrumentConfig,
    ScheduleConfig,
)
from fablebot.engine import TradingEngine
from fablebot.notify import Notifier
from tests.conftest import make_ohlcv

MR_PARAMS = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14, "atr_stop_mult": 2.0}
MB_PARAMS = {"entry_channel": 20, "exit_channel": 10, "atr_period": 14, "atr_stop_mult": 2.5}


def make_config():
    return Config(
        account=AccountConfig(starting_equity=100_000.0, risk_per_trade=0.01),
        schedule=ScheduleConfig(),
        correlation=CorrelationConfig(
            groups={"risk_on": ["SPX", "NDX", "BTC"]}, max_same_direction_per_group=2
        ),
        instruments={
            "SPX": InstrumentConfig("SPX", "^GSPC", "S&P 500", "mean_reversion", "15m", MR_PARAMS),
            "NDX": InstrumentConfig("NDX", "^IXIC", "Nasdaq", "mean_reversion", "15m", MR_PARAMS),
            "BTC": InstrumentConfig("BTC", "BTC-USD", "Bitcoin", "momentum_breakout", "1h", MB_PARAMS),
        },
    )


def noisy_flat(n, level=100.0, seed=3):
    rng = np.random.default_rng(seed)
    return level + rng.normal(0, 1.0, n)


class FakeFeed:
    """Returns a canned DataFrame per symbol."""

    def __init__(self, frames):
        self.frames = frames

    def __call__(self, symbol, timeframe):
        return self.frames[symbol]


class SilentNotifier(Notifier):
    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, title, body):
        self.sent.append((title, body))


def engine_with(frames, state_path=None):
    return TradingEngine(
        make_config(),
        state_path=state_path,
        fetch=FakeFeed(frames),
        notifier=SilentNotifier(),
    )


def dip(seed):
    """Flat-noise series ending in a sharp dip — a long signal for MR."""
    return make_ohlcv(list(noisy_flat(40, seed=seed)) + [92.0], freq="15min")


def flat(seed):
    return make_ohlcv(noisy_flat(41, seed=seed))


def test_cycle_opens_sized_position_with_stop():
    frames = {"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)}
    engine = engine_with(frames)
    log = engine.run_cycle()
    assert any("SPX: opened long" in line for line in log)
    pos = engine.broker.positions["SPX"]
    # Hard 1% risk budget, stop priced to lose exactly that
    assert pos.risk_amount == pytest.approx(1000.0)
    assert pos.units * (pos.entry_price - pos.stop_price) == pytest.approx(1000.0)


def test_trade_events_carry_a_live_account_summary():
    """Opening/closing/trailing log lines must surface equity + live unrealized,
    so the terminal tells you the account state at each event (not just the bot)."""
    engine = engine_with({"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)})
    log = engine.run_cycle()
    opened = next(line for line in log if "SPX: opened long" in line)
    assert "acct $" in opened and "real" in opened and "unreal" in opened


def test_write_dashboard_emits_html_and_never_throws(tmp_path):
    state = tmp_path / "portfolio.json"
    engine = engine_with({"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)}, state_path=state)
    engine.run_cycle()                      # opens a position, marks prices
    engine._write_dashboard()
    out = tmp_path / "dashboard.html"
    assert out.exists()
    html = out.read_text()
    assert html.startswith("<!DOCTYPE html>") and "FableBot" in html
    # stateless engine (no path) is a safe no-op, not a crash
    engine_with({"^GSPC": flat(1), "^IXIC": flat(2), "BTC-USD": flat(3)}, state_path=None)._write_dashboard()


def test_write_dashboard_works_for_a_fresh_book(tmp_path):
    """A brand-new book (the second bot) has no portfolio.json yet; the dashboard
    must still render at starting equity from the in-memory broker."""
    state = tmp_path / "portfolio.json"
    engine = engine_with({"^GSPC": flat(1), "^IXIC": flat(2), "BTC-USD": flat(3)}, state_path=state)
    assert not state.exists()                       # nothing traded -> no file on disk yet
    engine._write_dashboard()
    html = (tmp_path / "dashboard.html").read_text()
    assert "FableBot" in html and "$100,000" in html  # fresh $100k book renders


def test_experiment_config_is_a_valid_mean_reversion_book():
    from pathlib import Path
    from fablebot.config import load_config
    cfg = load_config(Path(__file__).resolve().parent.parent / "config_experiment.yaml")
    assert cfg.account.starting_equity == 100_000.0
    assert set(cfg.instruments) == {"SPY", "QQQ", "IWM", "DIA", "XLK"}
    assert all(i.strategy == "mean_reversion" and i.timeframe == "15m" for i in cfg.instruments.values())


def _tz_aware(df, end):
    """Re-index a frame onto a real-time (tz-aware UTC) clock ending at `end`."""
    out = df.copy()
    out.index = pd.date_range(end=end, periods=len(df), freq="15min", tz="UTC")
    return out


def test_feed_is_stale_thresholds():
    from fablebot.engine import feed_is_stale
    now = pd.Timestamp("2026-06-26 12:00", tz="UTC")
    # 15m: ~40-min threshold — a 50-min-old bar is stale; the in-progress bar or a
    # single missed poll (<=~30 min) is fine, so it doesn't false-trip intraday
    assert feed_is_stale(now - pd.Timedelta(minutes=50), "15m", now=now)
    assert not feed_is_stale(now - pd.Timedelta(minutes=30), "15m", now=now)
    # daily tolerates a long weekend but not a dead week
    assert not feed_is_stale(now - pd.Timedelta(days=3), "1d", now=now)
    assert feed_is_stale(now - pd.Timedelta(days=7), "1d", now=now)


def test_no_same_bar_reentry_after_exit():
    """After exiting on a bar, the same-direction signal on that SAME bar must not
    re-open (the stop-out -> re-enter loop). A new bar clears the block."""
    fresh_end = pd.Timestamp.now(tz="UTC")
    frame = _tz_aware(dip(1), fresh_end)        # oversold -> long signal, fresh
    engine = engine_with({"^GSPC": frame, "^IXIC": flat(2), "BTC-USD": flat(3)})
    inst = engine.config.instruments["SPX"]
    engine._last_exit["SPX"] = (frame.index[-1], "long")   # just exited a long here
    log = engine.evaluate_instrument(inst, frame)
    assert "SPX" not in engine.broker.positions
    assert any("no same-bar re-entry" in line for line in log)
    # a new bar (later timestamp) clears the block -> the same signal opens
    frame2 = _tz_aware(dip(1), fresh_end + pd.Timedelta(minutes=15))
    log2 = engine.evaluate_instrument(inst, frame2)
    assert "SPX" in engine.broker.positions
    assert any("opened long" in line for line in log2)


def test_cycle_idles_on_a_stale_feed_instead_of_reentry_loop():
    """The overnight bug: when the market is closed the feed returns the same
    'oversold' bar every poll, and the bot opened a position whose stop that bar's
    own low already breached — stop-out and re-enter forever. A stale bar must now
    open nothing; a fresh copy of the same dip trades normally."""
    fresh_end = pd.Timestamp.now(tz="UTC")
    stale_end = fresh_end - pd.Timedelta(hours=6)        # market closed: 6h-old 15m bar

    engine = engine_with({"^GSPC": _tz_aware(dip(1), stale_end),
                          "^IXIC": _tz_aware(flat(2), stale_end),
                          "BTC-USD": _tz_aware(flat(3), stale_end)})
    log = engine.run_cycle()
    assert any("SPX" in line and "stale" in line for line in log)
    assert "SPX" not in engine.broker.positions          # no degenerate entry

    fresh = engine_with({"^GSPC": _tz_aware(dip(1), fresh_end),
                         "^IXIC": _tz_aware(flat(2), fresh_end),
                         "BTC-USD": _tz_aware(flat(3), fresh_end)})
    log2 = fresh.run_cycle()
    assert not any("stale" in line for line in log2)
    assert any("SPX: opened long" in line for line in log2)


def test_engine_runs_a_configured_genome_through_the_risk_engine():
    """An instrument carrying a composable-DSL `genome` must trade that genome
    live (not its named strategy), still sized/stopped by the risk engine."""
    from fablebot import dsl
    from fablebot.dsl import GenomeStrategy

    cfg = make_config()
    cfg.instruments["SPX"] = InstrumentConfig(
        "SPX", "^GSPC", "S&P 500", "mean_reversion", "15m", {}, genome=dsl.seed_mean_reversion()
    )
    engine = TradingEngine(
        cfg, state_path=None,
        fetch=FakeFeed({"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)}),
        notifier=SilentNotifier(),
    )
    assert isinstance(engine.strategies["SPX"], GenomeStrategy)
    log = engine.run_cycle()
    assert any("SPX: opened long" in line for line in log)
    pos = engine.broker.positions["SPX"]
    assert pos.risk_amount == pytest.approx(1000.0)
    assert pos.units * (pos.entry_price - pos.stop_price) == pytest.approx(1000.0)


def test_engine_trains_and_trades_a_live_model():
    """An instrument with a `model` block trains a live ML model on deep history
    at init, trades it (through the unchanged risk engine), and retrains on call.
    """
    from fablebot.ml import MLStrategy

    cfg = make_config()
    cfg.instruments["BTC"] = InstrumentConfig(
        "BTC", "BTC-USD", "Bitcoin", "momentum_breakout", "1h", {},
        model={"type": "ml_gbt", "n_estimators": 15, "max_depth": 2},
    )
    deep = make_ohlcv(
        np.abs(100 + np.cumsum(np.random.default_rng(0).normal(0.05, 1.0, 800))) + 40, freq="1h"
    )
    engine = TradingEngine(
        cfg, state_path=None,
        fetch=FakeFeed({"^GSPC": flat(2), "^IXIC": flat(3), "BTC-USD": deep.tail(200)}),
        notifier=SilentNotifier(),
        history_fetch=lambda symbol, timeframe: deep,
    )
    assert isinstance(engine.strategies["BTC"], MLStrategy)
    assert isinstance(engine.run_cycle(), list)  # trades the model with no error
    engine.retrain_models()  # daily retrain rebuilds it
    assert isinstance(engine.strategies["BTC"], MLStrategy)


def test_model_instrument_falls_back_when_untrainable():
    """Too little data to train -> the bot never goes dark; it falls back to the
    instrument's named strategy."""
    from fablebot.strategies.momentum_breakout import MomentumBreakout

    cfg = make_config()
    cfg.instruments["BTC"] = InstrumentConfig(
        "BTC", "BTC-USD", "Bitcoin", "momentum_breakout", "1h", MB_PARAMS,
        model={"type": "ml_gbt"},
    )
    tiny = make_ohlcv(np.linspace(100, 110, 50), freq="1h")
    engine = TradingEngine(
        cfg, state_path=None,
        fetch=FakeFeed({"^GSPC": flat(2), "^IXIC": flat(3), "BTC-USD": tiny}),
        notifier=SilentNotifier(),
        history_fetch=lambda symbol, timeframe: tiny,
    )
    assert isinstance(engine.strategies["BTC"], MomentumBreakout)


def test_zero_atr_signal_does_not_crash_or_open():
    """A degenerate ATR must not raise out of sizing or open an unsized trade."""
    from fablebot.strategies.base import Action, Signal, Strategy

    class ZeroAtrOpener(Strategy):
        name = "zatr"
        def min_bars(self):
            return 2
        def evaluate(self, df, position_direction):
            return Signal(Action.OPEN_LONG, "wants in but atr=0", 0.0)

    engine = engine_with({"^GSPC": flat(1), "^IXIC": flat(2), "BTC-USD": flat(3)})
    engine.strategies["SPX"] = ZeroAtrOpener({})
    engine.run_cycle()  # must not raise
    assert "SPX" not in engine.broker.positions  # no ATR -> no size -> no position


def test_one_instrument_error_does_not_crash_the_cycle():
    """A single instrument blowing up must be logged and skipped, not fatal —
    the 24/7 loop keeps trading everything else."""
    from fablebot.strategies.base import Strategy

    class Boom(Strategy):
        name = "boom"
        def min_bars(self):
            return 2
        def evaluate(self, df, position_direction):
            raise RuntimeError("kaboom")

    engine = engine_with({"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)})
    engine.strategies["NDX"] = Boom({})
    log = engine.run_cycle()  # must not raise
    assert any("NDX: evaluation error" in line for line in log)
    assert "SPX" in engine.broker.positions  # other instruments still processed


def test_correlation_filter_blocks_third_risk_on_long():
    breakout = make_ohlcv(list(noisy_flat(40, seed=4)) + [110.0])
    frames = {"^GSPC": dip(1), "^IXIC": dip(2), "BTC-USD": breakout}
    engine = engine_with(frames)
    log = engine.run_cycle()
    assert "SPX" in engine.broker.positions
    assert "NDX" in engine.broker.positions
    assert "BTC" not in engine.broker.positions
    assert any("BTC" in line and "blocked" in line for line in log)


def test_data_failure_is_isolated():
    frames = {"^GSPC": dip(1), "^IXIC": flat(2)}  # BTC-USD missing -> KeyError
    engine = engine_with(frames)
    log = engine.run_cycle()
    assert any("BTC: data fetch failed" in line for line in log)
    assert "SPX" in engine.broker.positions  # others unaffected


def test_morning_and_evening_reports_render():
    frames = {"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)}
    engine = engine_with(frames)
    engine.run_cycle()
    engine.send_morning_report()
    engine.send_evening_report()
    titles = [t for t, _ in engine.notifier.sent]
    assert titles == ["Morning briefing", "Evening recap"]
    morning_body = engine.notifier.sent[0][1]
    assert "S&P 500" in morning_body and "watching" in morning_body
    evening_body = engine.notifier.sent[1][1]
    assert "Equity:" in evening_body


def test_blocked_signal_opens_shadow_position():
    breakout = make_ohlcv(list(noisy_flat(40, seed=4)) + [110.0])
    frames = {"^GSPC": dip(1), "^IXIC": dip(2), "BTC-USD": breakout}
    engine = engine_with(frames)
    engine.run_cycle()
    assert "BTC" not in engine.broker.positions
    assert "BTC" in engine.shadow.positions
    shadow_pos = engine.shadow.positions["BTC"]
    # Same sizing discipline as a real trade
    assert shadow_pos.risk_amount == pytest.approx(1000.0)
    blocked = [
        e
        for e in engine.journal.read()
        if e["event"] == "decision" and e["instrument"] == "BTC" and e["action"] == "blocked"
    ]
    assert len(blocked) == 1
    assert "correlation" in blocked[0]["reason"]


def test_shadow_trade_resolves_and_is_journaled():
    breakout = make_ohlcv(list(noisy_flat(40, seed=4)) + [110.0])
    frames = {"^GSPC": dip(1), "^IXIC": dip(2), "BTC-USD": breakout}
    engine = engine_with(frames)
    engine.run_cycle()
    assert "BTC" in engine.shadow.positions

    # The breakout collapses far through the shadow stop
    frames["BTC-USD"] = make_ohlcv(list(noisy_flat(40, seed=4)) + [110.0, 80.0])
    engine.run_cycle()
    assert "BTC" not in engine.shadow.positions
    shadow_trades = [
        e for e in engine.journal.read() if e["event"] == "trade_closed" and e["shadow"]
    ]
    assert len(shadow_trades) == 1
    assert shadow_trades[0]["trade"]["instrument"] == "BTC"
    assert shadow_trades[0]["trade"]["pnl"] < 0
    # The counterfactual never touches real equity
    assert engine.broker.realized_pnl == 0.0


def test_every_decision_is_journaled_including_holds():
    frames = {"^GSPC": flat(1), "^IXIC": flat(2), "BTC-USD": flat(3)}
    engine = engine_with(frames)
    engine.run_cycle()
    decisions = [e for e in engine.journal.read() if e["event"] == "decision"]
    assert len(decisions) == 3
    assert all(d["action"] == "hold" for d in decisions)
    assert all("context" in d and d["context"]["atr"] > 0 for d in decisions)


def test_journal_captures_strategy_context_on_entry():
    frames = {"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)}
    engine = engine_with(frames)
    engine.run_cycle()
    opens = [
        e
        for e in engine.journal.read()
        if e["event"] == "decision" and e["action"] == "open_long"
    ]
    assert len(opens) == 1
    assert opens[0]["context"]["zscore"] < -2.0  # the world as the strategy saw it


def test_state_survives_restart(tmp_path):
    state = tmp_path / "portfolio.json"
    frames = {"^GSPC": dip(1), "^IXIC": flat(2), "BTC-USD": flat(3)}
    engine_with(frames, state_path=state).run_cycle()

    revived = engine_with(frames, state_path=state)
    assert "SPX" in revived.broker.positions
