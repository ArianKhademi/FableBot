"""Skipped statistical checks must be visible without changing qualification rules."""

from types import SimpleNamespace

import numpy as np
import pytest

from fablebot import dsl, ml, optimizer, rl, search
from fablebot.config import InstrumentConfig
from tests.conftest import make_ohlcv


@pytest.mark.parametrize("kind", ["ml", "rl", "genome"])
@pytest.mark.parametrize("trials,score,expected", [(6, 0.2, True), (10, 0.2, False), (10, 0.99, True)])
def test_verdict_distinguishes_skipped_and_computed_dsr(monkeypatch, kind, trials, score, expected):
    df = make_ohlcv(np.linspace(100, 110, 100))
    candidate = SimpleNamespace(oos_sharpe=2.0, oos_equity_curve=list(range(100, 120)))
    baseline = SimpleNamespace(oos_sharpe=1.0)
    calls = []
    monkeypatch.setattr(optimizer, "validation_folds", lambda data: [data] * 4)
    monkeypatch.setattr(optimizer, "deflated_sharpe_ratio", lambda *args: calls.append(args) or score)
    if kind == "genome":
        genome = dsl.seed_mean_reversion()
        current = dsl.seed_trend_following()
        monkeypatch.setattr(search, "_perturb_genome", lambda value, rng: value)
        monkeypatch.setattr(optimizer, "run_walk_forward",
                            lambda data, inst, *args: baseline if inst.genome == current else candidate)
        verdict = search.validate_genome(df, None, "SYNTH", "1h", genome, current,
                                         np.random.default_rng(0), n_trials=trials)
    else:
        module = ml if kind == "ml" else rl
        monkeypatch.setattr(optimizer, "run_walk_forward", lambda *args: baseline)
        monkeypatch.setattr(module, kind + "_walk_forward", lambda *args, **kwargs: candidate)
        fit_name = "_fit_and_backtest" if kind == "ml" else "_train_and_backtest"
        monkeypatch.setattr(module, fit_name, lambda *args: (2.0, 20))
        inst = InstrumentConfig("SYNTH", "SYNTH", "Synthetic", kind, "1h")
        verdict = getattr(module, "validate_" + kind)(
            df, None, inst, dsl.seed_trend_following(), np.random.default_rng(0), n_trials=trials)
    assert verdict["qualified"] is expected
    assert verdict["deflated_sharpe_checked"] is (trials >= 10)
    assert len(calls) == (1 if trials >= 10 else 0)
    assert verdict["holdout_checked"] is False
    assert "held-out confirmation not checked" in verdict["reason"]
    assert ("DSR not checked" in verdict["reason"]) is (trials < 10)
    if trials >= 10:
        assert verdict["deflated_sharpe"] == score
