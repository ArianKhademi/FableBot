"""Stop sweep: backtest a (hard stop × trailing stop) grid for a strategy."""

import importlib.util
from pathlib import Path

import numpy as np

from fablebot.config import InstrumentConfig
from tests.conftest import make_ohlcv

ROOT = Path(__file__).resolve().parent.parent

INST = InstrumentConfig(
    key="GOLD", symbol="GC=F", name="Gold", strategy="trend_following", timeframe="4h",
    params={"fast_ema": 20, "slow_ema": 50, "atr_period": 14, "atr_stop_mult": 3.0},
)


def _mod():
    spec = importlib.util.spec_from_file_location("stop_sweep", ROOT / "scripts" / "stop_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _trending_df(n=1000, seed=2):
    closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0.15, 0.7, n))
    return make_ohlcv(np.abs(closes) + 40, freq="4h")


def test_sweep_covers_the_full_grid():
    m = _mod()
    rows = m.sweep(_trending_df(), INST, [2.0, 4.0], [0.0, 2.0])
    assert len(rows) == 4
    assert {(r["stop"], r["trail"]) for r in rows} == {(2.0, 0.0), (2.0, 2.0), (4.0, 0.0), (4.0, 2.0)}
    assert all("oos_sharpe" in r and "oos_trades" in r for r in rows)


def test_sweep_models_the_trailing_stop():
    """trail 0 (no trail) and trail 2 should be evaluated as distinct runs."""
    m = _mod()
    rows = m.sweep(_trending_df(), INST, [3.0], [0.0, 2.0])
    assert len(rows) == 2
    by_trail = {r["trail"]: r["oos_sharpe"] for r in rows}
    assert 0.0 in by_trail and 2.0 in by_trail
