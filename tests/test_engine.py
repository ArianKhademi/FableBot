"""End-to-end engine tests with a fake data feed — no network."""

import numpy as np
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
