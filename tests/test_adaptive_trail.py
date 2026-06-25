"""Adaptive mean-reversion trailing stop: loose far from the mean, tightening as
price nears it — plus the stop sweep's --near dimension that tunes it."""

import importlib.util
from pathlib import Path

import numpy as np

from fablebot.config import InstrumentConfig, load_config
from fablebot.strategies.mean_reversion import MeanReversion
from fablebot.strategies.momentum_breakout import MomentumBreakout
from tests.conftest import make_ohlcv

ROOT = Path(__file__).resolve().parent.parent
ADAPT = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14,
         "atr_stop_mult": 2.0, "trail_atr_mult": 4.0, "trail_near_mult": 1.0}


def _ending(last, seed=1):
    body = list(100 + np.random.default_rng(seed).normal(0, 1, 39))
    return make_ohlcv(body + [last], freq="1h")


def test_non_mr_trail_is_constant():
    mb = MomentumBreakout({"entry_channel": 20, "exit_channel": 10, "atr_period": 14,
                           "atr_stop_mult": 2.0, "trail_atr_mult": 2.5})
    assert mb.current_trail_mult(_ending(105), "long") == 2.5
    no_trail = MomentumBreakout({"entry_channel": 20, "exit_channel": 10, "atr_period": 14, "atr_stop_mult": 2.0})
    assert no_trail.current_trail_mult(_ending(105), "long") is None


def test_mr_trail_loose_far_tight_near():
    mr = MeanReversion(ADAPT)
    far, near = _ending(92.0), _ending(100.1)          # big stretch vs near the mean
    assert abs(mr._z(far)) > mr.entry_z
    assert abs(mr._z(near)) < mr.exit_z
    assert mr.current_trail_mult(far, "long") == 4.0   # loose = trail_atr_mult
    assert mr.current_trail_mult(near, "long") == 1.0  # tight = trail_near_mult


def test_mr_without_near_is_constant():
    mr = MeanReversion({k: v for k, v in ADAPT.items() if k != "trail_near_mult"})
    assert mr.current_trail_mult(_ending(92.0), "long") == 4.0
    assert mr.current_trail_mult(_ending(100.1), "long") == 4.0   # no narrowing


def _sweep():
    spec = importlib.util.spec_from_file_location("stop_sweep", ROOT / "scripts" / "stop_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sweep_grids_over_the_near_dimension():
    m = _sweep()
    inst = InstrumentConfig(key="SPY", symbol="SPY", name="S&P", strategy="mean_reversion", timeframe="4h",
                            params={"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14, "atr_stop_mult": 2.0})
    df = make_ohlcv(np.abs(100 + np.cumsum(np.random.default_rng(3).normal(0, 0.8, 900))) + 40, freq="4h")
    rows = m.sweep(df, inst, [3.0], [2.0], near_mults=(1.0, 2.0))
    assert len(rows) == 2
    assert {r["near"] for r in rows} == {1.0, 2.0}


def test_adaptive_config_loads():
    cfg = load_config(ROOT / "config_experiment_adaptive.yaml")
    assert cfg.name == "Experiment — 15m MR + adaptive trail"
    assert all(i.params.get("trail_atr_mult") and i.params.get("trail_near_mult")
               for i in cfg.instruments.values())
