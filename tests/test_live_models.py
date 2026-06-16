"""Building trained, live model-backed strategies from a config `model` spec."""

import numpy as np
import pytest

from fablebot.live_models import build_model_strategy
from fablebot.ml import MLStrategy
from fablebot.rl import RLStrategy
from tests.conftest import make_ohlcv


def _df(n=700, seed=0):
    closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0.05, 1.0, n))
    return make_ohlcv(np.abs(closes) + 40, freq="1h")


def test_build_ml_strategy_trains():
    strat = build_model_strategy({"type": "ml_gbt", "n_estimators": 15, "max_depth": 2}, _df())
    assert isinstance(strat, MLStrategy)
    assert strat.model.gbt is not None  # actually trained


def test_build_rl_strategy_trains():
    strat = build_model_strategy({"type": "rl_qlearn", "epochs": 10}, _df())
    assert isinstance(strat, RLStrategy)
    assert strat.agent.Q  # learned a policy table


def test_returns_none_on_insufficient_data():
    assert build_model_strategy({"type": "ml_gbt"}, _df(40)) is None
    assert build_model_strategy({"type": "rl_qlearn"}, _df(40)) is None


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        build_model_strategy({"type": "transformer"}, _df())
