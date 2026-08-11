"""Prop-firm rule layer: trailing drawdown (EOD vs realtime), buffer lock,
profit target, consistency, persistence, contract sizing, engine gating."""

import numpy as np
import pytest

from fablebot.prop_rules import PropAccount
from fablebot.risk import size_position


# -- drawdown modes ----------------------------------------------------------

def test_realtime_floor_trails_and_blows_intraday():
    acct = PropAccount(50_000, max_drawdown=2_000, drawdown_mode="realtime")
    acct.update(51_500, "d1")                       # watermark 51.5k -> floor 49.5k
    assert acct.state.floor == pytest.approx(49_500)
    note = acct.update(49_400, "d1")                # intraday dip through the floor
    assert acct.blown and "BLOWN" in note


def test_eod_mode_ignores_intraday_dips():
    acct = PropAccount(50_000, max_drawdown=2_000, drawdown_mode="eod")
    acct.update(51_500, "d1")
    assert acct.state.floor == pytest.approx(48_000)  # floor only moves at EOD
    assert acct.update(47_500, "d1") is None          # intraday breach: survives
    assert not acct.blown
    note = acct.end_of_day(47_900, "d1")              # but an EOD close below blows
    assert acct.blown and "EOD" in note


def test_eod_watermark_ratchets_only_at_close():
    acct = PropAccount(50_000, max_drawdown=2_000, drawdown_mode="eod")
    acct.end_of_day(51_000, "d1")                     # EOD high 51k -> floor 49k
    assert acct.state.floor == pytest.approx(49_000)
    assert acct.end_of_day(49_100, "d2") is None      # above floor: alive
    assert not acct.blown


def test_buffer_locks_the_floor_at_start():
    acct = PropAccount(50_000, max_drawdown=2_000, drawdown_mode="realtime", buffer=2_100)
    acct.update(52_200, "d1")                         # cleared start+buffer
    assert acct.state.floor_locked
    assert acct.state.floor == pytest.approx(50_000)  # locked at start, stops trailing
    acct.update(60_000, "d2")
    assert acct.state.floor == pytest.approx(50_000)


def test_profit_target_and_consistency_report():
    acct = PropAccount(50_000, max_drawdown=2_000, drawdown_mode="eod",
                       profit_target=3_000, consistency_pct=50)
    acct.end_of_day(52_000, "d1")                     # +2000 day
    acct.end_of_day(53_200, "d2")                     # +1200 day
    note = acct.update(53_200, "d3")
    assert acct.passed and "TARGET REACHED" in note
    assert "VIOLATED" in acct.consistency_note()      # best day 2000/3200 = 63% > 50%


def test_blown_state_persists_across_restart(tmp_path):
    path = tmp_path / "prop.json"
    acct = PropAccount(50_000, 2_000, "realtime", state_path=path)
    acct.update(47_000, "d1")
    assert acct.blown
    revived = PropAccount(50_000, 2_000, "realtime", state_path=path)
    assert revived.blown                              # a blown account stays blown


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        PropAccount(50_000, 2_000, drawdown_mode="intraday")


# -- integer contract sizing -------------------------------------------------

def test_futures_sizing_whole_contracts_and_exact_units():
    s = size_position(equity=50_000, risk_per_trade=0.004, entry_price=7_775.0,
                      atr=10.0, atr_stop_mult=2.0, direction="long",
                      contract_multiplier=5.0)
    # $200 budget / (20pt stop x $5) = 2 contracts; units = 2 x 5 = 10
    assert s.contracts == 2 and s.units == pytest.approx(10.0)
    assert s.risk_amount == pytest.approx(200.0)      # actual, not budget


def test_futures_sizing_zero_contracts_when_too_expensive():
    s = size_position(equity=50_000, risk_per_trade=0.004, entry_price=29_741.0,
                      atr=60.0, atr_stop_mult=2.0, direction="long",
                      contract_multiplier=2.0)
    # one MNQ contract risks 120pt x $2 = $240 > $200 budget -> skip, never round up
    assert s.contracts == 0 and s.units == 0.0


def test_futures_sizing_respects_contract_cap():
    s = size_position(equity=50_000, risk_per_trade=0.02, entry_price=3_000.0,
                      atr=4.0, atr_stop_mult=2.0, direction="long",
                      contract_multiplier=5.0, max_contracts=3)
    assert s.contracts == 3                            # 25 affordable, capped at 3


# -- engine integration ------------------------------------------------------

def _futures_engine(tmp_path=None):
    from fablebot.config import (AccountConfig, Config, CorrelationConfig,
                                 InstrumentConfig, PropConfig, ScheduleConfig)
    from fablebot.engine import TradingEngine
    from tests.test_engine import FakeFeed, SilentNotifier, noisy_flat
    from tests.conftest import make_ohlcv
    cfg = Config(
        account=AccountConfig(starting_equity=50_000.0, risk_per_trade=0.004),
        schedule=ScheduleConfig(),
        correlation=CorrelationConfig(groups={}, max_same_direction_per_group=2),
        instruments={"MES": InstrumentConfig(
            "MES", "MES=F", "Micro S&P", "mean_reversion", "15m",
            {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14, "atr_stop_mult": 2.0},
            contract_multiplier=5.0)},
        prop=PropConfig(max_drawdown=2_000, drawdown_mode="realtime", max_contracts=50),
    )
    dip = make_ohlcv(list(noisy_flat(40, seed=1)) + [92.0], freq="15min")
    return TradingEngine(cfg, state_path=None, fetch=FakeFeed({"MES=F": dip}),
                         notifier=SilentNotifier()), dip


def test_engine_opens_whole_contracts_and_blocks_when_blown():
    engine, dip = _futures_engine()
    log = engine.run_cycle()
    assert any("opened long" in l and "contract(s)" in l for l in log)
    pos = engine.broker.positions["MES"]
    assert pos.units % 5.0 == pytest.approx(0.0)      # contracts x multiplier
    engine.broker.close_position("MES", 92.0, "cleanup")

    engine.prop.state.blown = True                     # account dies
    log2 = engine.run_cycle()
    assert any("prop account blown" in l for l in log2)
    assert "MES" not in engine.broker.positions
