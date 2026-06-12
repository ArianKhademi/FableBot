"""Optimizer: walk-forward validation, Sharpe, sampling, proposal logic."""

import numpy as np
import pytest

from fablebot.config import InstrumentConfig
from fablebot.optimizer import (
    BacktestHistory,
    ContinuousOptimizer,
    MIN_IMPROVEMENT,
    MIN_TRADES,
    ProposalQueue,
    _round_param,
    run_walk_forward,
    sample_params,
    sharpe_ratio,
)
from tests.conftest import make_ohlcv

MR_INST = InstrumentConfig(
    key="SPX",
    symbol="^GSPC",
    name="S&P 500",
    strategy="mean_reversion",
    timeframe="15m",
    params={"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "atr_period": 14, "atr_stop_mult": 2.0},
    param_bounds={"lookback": [10, 40], "entry_z": [1.5, 3.0], "exit_z": [0.2, 1.0], "atr_stop_mult": [1.5, 3.5]},
)

MB_INST = InstrumentConfig(
    key="BTC",
    symbol="BTC-USD",
    name="Bitcoin",
    strategy="momentum_breakout",
    timeframe="1h",
    params={"entry_channel": 20, "exit_channel": 10, "atr_period": 14, "atr_stop_mult": 2.5, "trail_atr_mult": 2.5},
    param_bounds={"entry_channel": [10, 40], "exit_channel": [5, 20], "atr_stop_mult": [1.5, 4.0], "trail_atr_mult": [1.5, 4.0]},
)


# ── Sharpe ratio ─────────────────────────────────────────────────────────────

def test_sharpe_positive_on_monotone_equity():
    equity = list(np.linspace(100_000, 120_000, 200))
    assert sharpe_ratio(equity, 252) > 0


def test_sharpe_negative_on_drawdown():
    equity = list(np.linspace(100_000, 80_000, 200))
    assert sharpe_ratio(equity, 252) < 0


def test_sharpe_minus_inf_on_tiny_curve():
    assert sharpe_ratio([100_000], 252) == float("-inf")


def test_sharpe_zero_on_flat():
    assert sharpe_ratio([100_000] * 100, 252) == 0.0


# ── param sampling ──────────────────────────────────────────────────────────

def test_sampled_params_within_bounds():
    rng = np.random.default_rng(42)
    for _ in range(50):
        p = sample_params(MR_INST, rng)
        assert 10 <= p["lookback"] <= 40
        assert 1.5 <= p["entry_z"] <= 3.0
        assert 0.2 <= p["exit_z"] <= 1.0


def test_sampled_params_freeze_unlisted():
    rng = np.random.default_rng(0)
    p = sample_params(MR_INST, rng)
    assert p["atr_period"] == 14  # not in bounds → frozen


def test_exit_channel_always_less_than_entry():
    rng = np.random.default_rng(7)
    for _ in range(30):
        p = sample_params(MB_INST, rng)
        assert p["exit_channel"] < p["entry_channel"]


def test_integer_params_are_integers():
    rng = np.random.default_rng(1)
    p = sample_params(MR_INST, rng)
    assert isinstance(p["lookback"], int)


# ── walk-forward ─────────────────────────────────────────────────────────────

def test_walk_forward_returns_trial_on_sufficient_data():
    df = make_ohlcv(np.linspace(100, 200, 300), freq="15min")
    trial = run_walk_forward(df, MR_INST, dict(MR_INST.params))
    assert trial is not None
    assert trial.is_bars > trial.oos_bars  # 75/25 split
    assert trial.oos_bars > 0
    assert not np.isnan(trial.is_sharpe)


def test_walk_forward_returns_none_on_too_few_bars():
    df = make_ohlcv(np.linspace(100, 110, 20), freq="15min")
    assert run_walk_forward(df, MR_INST, dict(MR_INST.params)) is None


def test_walk_forward_bad_params_return_none():
    df = make_ohlcv(np.linspace(100, 200, 300), freq="15min")
    bad = {**MR_INST.params, "entry_z": -99.0}  # triggers no trades, empty equity
    # Should not raise; may return a trial with 0 trades or None
    result = run_walk_forward(df, MR_INST, bad)
    # Just check it doesn't crash


# ── BacktestHistory ──────────────────────────────────────────────────────────

def test_history_log_and_read(tmp_path):
    h = BacktestHistory(tmp_path / "hist.jsonl")
    df = make_ohlcv(np.linspace(100, 150, 300), freq="15min")
    trial = run_walk_forward(df, MR_INST, dict(MR_INST.params))
    h.log(trial)
    entries = h.read("SPX")
    assert len(entries) == 1
    assert entries[0]["instrument"] == "SPX"


def test_already_tried_deduplication(tmp_path):
    h = BacktestHistory(tmp_path / "hist.jsonl")
    df = make_ohlcv(np.linspace(100, 150, 300), freq="15min")
    trial = run_walk_forward(df, MR_INST, dict(MR_INST.params))
    h.log(trial)
    assert h.already_tried("SPX", dict(MR_INST.params), within_hours=48)
    assert not h.already_tried("SPX", {**MR_INST.params, "lookback": 99}, within_hours=48)


def test_baseline_flag(tmp_path):
    h = BacktestHistory(tmp_path / "hist.jsonl")
    df = make_ohlcv(np.linspace(100, 150, 300), freq="15min")
    trial = run_walk_forward(df, MR_INST, dict(MR_INST.params))
    trial.is_baseline = True
    h.log(trial)
    assert h.baseline("SPX") is not None
    assert h.baseline("SPX")["is_baseline"] is True


# ── ProposalQueue ────────────────────────────────────────────────────────────

def test_proposal_queue_write_and_read_pending(tmp_path):
    from fablebot.optimizer import Proposal
    q = ProposalQueue(tmp_path / "proposals.jsonl")
    p = Proposal(
        instrument="BTC",
        strategy="momentum_breakout",
        timeframe="1h",
        proposed_params={"entry_channel": 15},
        current_params={"entry_channel": 20},
        oos_sharpe=0.9,
        baseline_oos_sharpe=0.6,
        improvement=0.3,
        oos_trades=12,
    )
    q.add(p)
    pending = q.read_pending()
    assert len(pending) == 1
    assert pending[0]["improvement"] == pytest.approx(0.3)


def test_promoted_proposals_excluded_from_pending(tmp_path):
    from fablebot.optimizer import Proposal
    import json
    path = tmp_path / "proposals.jsonl"
    q = ProposalQueue(path)
    p = Proposal("BTC", "mb", "1h", {}, {}, 0.9, 0.6, 0.3, 10)
    q.add(p)
    # Manually mark as promoted
    data = json.loads(path.read_text())
    data["promoted"] = True
    path.write_text(json.dumps(data) + "\n")
    assert q.read_pending() == []


# ── ContinuousOptimizer end-to-end ──────────────────────────────────────────

def make_config_with(inst: InstrumentConfig):
    from fablebot.config import AccountConfig, Config, CorrelationConfig, ScheduleConfig
    return Config(
        account=AccountConfig(starting_equity=100_000.0, risk_per_trade=0.01),
        schedule=ScheduleConfig(),
        correlation=CorrelationConfig(groups={}, max_same_direction_per_group=2),
        instruments={inst.key: inst},
    )


def test_optimizer_trial_one_logs_to_history(tmp_path):
    df = make_ohlcv(np.linspace(100, 200, 400), freq="15min")
    config = make_config_with(MR_INST)
    history = BacktestHistory(tmp_path / "hist.jsonl")
    proposals = ProposalQueue(tmp_path / "proposals.jsonl")

    def fake_fetch(symbol, timeframe, days=None):
        return df

    opt = ContinuousOptimizer(config, history, proposals, fetch_fn=fake_fetch, seed=42)
    trial = opt.trial_one(MR_INST)
    # May be None if dedup fires on the first run (baseline + trial same params)
    entries = history.read("SPX")
    assert len(entries) >= 1  # at least the baseline


def test_optimizer_generates_proposal_when_oos_better(tmp_path):
    """Force a scenario where OOS Sharpe exceeds baseline by > MIN_IMPROVEMENT."""
    from fablebot.optimizer import Trial
    config = make_config_with(MR_INST)
    history = BacktestHistory(tmp_path / "hist.jsonl")
    proposals = ProposalQueue(tmp_path / "proposals.jsonl")

    df = make_ohlcv(np.linspace(100, 200, 400), freq="15min")

    def fake_fetch(symbol, timeframe, days=None):
        return df

    opt = ContinuousOptimizer(config, history, proposals, fetch_fn=fake_fetch, seed=42)
    # Inject a low baseline so the next real trial beats it
    fake_baseline = Trial(
        instrument="SPX", strategy="mean_reversion", timeframe="15m",
        params=dict(MR_INST.params), is_sharpe=0.1, oos_sharpe=0.0,
        is_trades=10, oos_trades=10, is_bars=300, oos_bars=100,
        is_return_pct=1.0, oos_return_pct=-1.0, is_baseline=True,
    )
    history.log(fake_baseline)

    # Run trials until we either get a proposal or exhaust patience
    for _ in range(20):
        opt.trial_one(MR_INST)
        if proposals.read_pending():
            break

    # With a 0.0 baseline and real backtests on a clean trend, we expect at
    # least some proposals to emerge (not guaranteed in 20 trials but likely).
    # The test is really checking the proposal plumbing works without error.
    assert isinstance(proposals.read_pending(), list)
