"""Optimizer: walk-forward validation, Sharpe, sampling, proposal logic."""

import numpy as np
import pytest

from fablebot.config import InstrumentConfig
from fablebot.optimizer import (
    BacktestHistory,
    ContinuousOptimizer,
    MIN_IMPROVEMENT,
    MIN_TRADES,
    MIN_TRIALS_FOR_DSR,
    Trial,
    _round_param,
    deflated_sharpe_ratio,
    equity_returns,
    expected_max_sharpe,
    perturb_params,
    perturbation_passes,
    probabilistic_sharpe_ratio,
    run_walk_forward,
    sample_params,
    sharpe_ratio,
    split_search_holdout,
    validation_folds,
)
from fablebot.pipeline import CANDIDATE, ProposalStore
from tests.conftest import make_ohlcv


def _fake_trial(instrument="SPX", oos_sharpe=0.0, params=None, oos_curve=None, **over):
    base = dict(
        instrument=instrument, strategy="mean_reversion", timeframe="15m",
        params=params if params is not None else dict(MR_INST.params),
        is_sharpe=0.0, oos_sharpe=oos_sharpe, is_trades=30, oos_trades=30,
        is_bars=300, oos_bars=100, is_return_pct=0.0, oos_return_pct=0.0,
    )
    base.update(over)
    t = Trial(**base)
    if oos_curve is not None:
        t.oos_equity_curve = list(oos_curve)
    return t

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


# ── validation building blocks ──────────────────────────────────────────────

def test_validation_folds_cover_different_regimes():
    df = make_ohlcv(np.linspace(100, 200, 400), freq="15min")
    folds = validation_folds(df, n_folds=4)
    assert len(folds) == 4
    # Windows roll forward: each fold starts later than the previous
    starts = [f.index[0] for f in folds]
    assert starts == sorted(starts)
    assert starts[0] != starts[-1]
    # Each window is a real chunk, not the whole history
    assert all(len(f) < len(df) for f in folds)


def test_perturb_params_stays_within_bounds_and_constraints():
    rng = np.random.default_rng(3)
    for _ in range(30):
        p = perturb_params(dict(MB_INST.params), MB_INST, rng)
        assert 10 <= p["entry_channel"] <= 40
        assert p["exit_channel"] < p["entry_channel"]
        assert 1.5 <= p["atr_stop_mult"] <= 4.0
        assert p["atr_period"] == 14  # frozen params untouched


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
    proposals = ProposalStore(tmp_path / "proposals.jsonl")

    def fake_fetch(symbol, timeframe, days=None):
        return df

    opt = ContinuousOptimizer(config, history, proposals, fetch_fn=fake_fetch, seed=42)
    opt.trial_one(MR_INST)
    entries = history.read("SPX")
    assert len(entries) >= 1  # at least the baseline


def test_optimizer_candidates_enter_pipeline_and_get_validated(tmp_path):
    """Low injected baseline -> trials become candidates -> validate_one
    transitions them out of candidate status with a recorded verdict."""
    from fablebot.optimizer import Trial
    config = make_config_with(MR_INST)
    history = BacktestHistory(tmp_path / "hist.jsonl")
    proposals = ProposalStore(tmp_path / "proposals.jsonl")

    df = make_ohlcv(np.linspace(100, 200, 400), freq="15min")

    def fake_fetch(symbol, timeframe, days=None):
        return df

    opt = ContinuousOptimizer(config, history, proposals, fetch_fn=fake_fetch, seed=42)
    fake_baseline = Trial(
        instrument="SPX", strategy="mean_reversion", timeframe="15m",
        params=dict(MR_INST.params), is_sharpe=0.1, oos_sharpe=-5.0,
        is_trades=10, oos_trades=10, is_bars=300, oos_bars=100,
        is_return_pct=1.0, oos_return_pct=-1.0, is_baseline=True,
    )
    history.log(fake_baseline)

    for _ in range(25):
        opt.trial_one(MR_INST)
        if proposals.by_status(CANDIDATE):
            break

    candidates = proposals.by_status(CANDIDATE)
    if candidates:  # candidate emergence depends on sampled params hitting trades
        verdict = opt.validate_one()
        assert verdict is not None
        assert "reason" in verdict
        # The candidate moved on: either qualified or rejected, never stuck
        state = proposals.latest()[candidates[0]["proposal_id"]]
        assert state["status"] in ("qualified", "rejected")
    # Plumbing check regardless: validate_one with no candidates is a no-op
    while opt.validate_one() is not None:
        pass
    assert proposals.by_status(CANDIDATE) == []


# ── Workstream A: hardened validation gate ──────────────────────────────────

# -- trade-count floor was raised off the noisy MIN_TRADES=8 ------------------

def test_min_trades_floor_is_substantial():
    assert MIN_TRADES >= 20


# -- deflated / probabilistic Sharpe (multiple-comparisons correction) --------

def test_psr_high_for_consistent_gains_low_for_noise():
    rng = np.random.default_rng(1)
    strong = rng.normal(0.01, 0.002, 300)  # steady gains, tiny vol -> high SR
    assert probabilistic_sharpe_ratio(strong, 0.0) > 0.99
    noise = rng.normal(0.0, 0.01, 300)  # zero-drift random returns
    assert probabilistic_sharpe_ratio(noise, 0.0) < 0.95


def test_psr_degenerate_streams_score_zero():
    assert probabilistic_sharpe_ratio(np.array([0.01, 0.01]), 0.0) == 0.0  # too short
    assert probabilistic_sharpe_ratio(np.full(50, 0.01), 0.0) == 0.0       # zero variance


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(5000, 0.02) > expected_max_sharpe(20, 0.02)
    assert expected_max_sharpe(1, 0.02) == 0.0   # undefined for a single draw
    assert expected_max_sharpe(50, 0.0) == 0.0   # no dispersion -> no benchmark


def test_deflation_inputs_caps_effective_trials():
    from fablebot.optimizer import DSR_EFFECTIVE_TRIALS_CAP, BARS_PER_YEAR, deflation_inputs
    sharpes = list(np.random.default_rng(0).normal(0.0, 2.0, 50_000))
    n_eff, sr_std_perbar = deflation_inputs(sharpes, BARS_PER_YEAR["1h"])
    assert n_eff == DSR_EFFECTIVE_TRIALS_CAP   # capped, not 50,000 (that made the gate unpassable)
    assert sr_std_perbar > 0


def test_deflation_inputs_robust_to_outliers():
    from fablebot.optimizer import deflation_inputs
    base = list(np.random.default_rng(1).normal(0.5, 0.3, 300))
    clean = deflation_inputs(base, 252)[1]
    with_mirages = deflation_inputs(base + [500.0, -500.0, 800.0], 252)[1]
    # winsorising keeps a few overfit single-split mirages from inflating the bar
    assert with_mirages < 2.0 * clean


def test_deflation_penalises_more_comparisons():
    """The same edge is less convincing when it was the best of many draws."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0009, 0.01, 250)  # a modest real-looking edge
    few = deflated_sharpe_ratio(returns, n_trials=3, sr_std=0.03)
    many = deflated_sharpe_ratio(returns, n_trials=5000, sr_std=0.03)
    assert few > many


def test_deflated_sharpe_gate_rejects_noise_keeps_real_edge(tmp_path):
    """The entry gate must reject a noisy OOS curve once enough configs have
    been tried, while still passing a genuine edge."""
    config = make_config_with(MR_INST)
    history = BacktestHistory(tmp_path / "h.jsonl")
    proposals = ProposalStore(tmp_path / "p.jsonl")
    opt = ContinuousOptimizer(config, history, proposals, fetch_fn=lambda *a, **k: None, seed=0)

    # Seed a broad search so the multiple-comparison correction is in force.
    rng = np.random.default_rng(7)
    for _ in range(MIN_TRIALS_FOR_DSR + 40):
        history.log(_fake_trial(oos_sharpe=float(rng.normal(0.5, 1.5))))

    noisy_curve = 100_000 * np.cumprod(1 + rng.normal(0.0, 0.01, 200))
    assert opt._passes_deflated_sharpe(MR_INST, _fake_trial(oos_curve=noisy_curve)) is False

    strong_curve = np.linspace(100_000, 130_000, 200)
    assert opt._passes_deflated_sharpe(MR_INST, _fake_trial(oos_curve=list(strong_curve))) is True


def test_deflated_sharpe_gate_skipped_until_enough_trials(tmp_path):
    """Too few prior trials -> multiple-comparison risk is low and the variance
    estimate is unreliable, so the gate abstains (returns True)."""
    config = make_config_with(MR_INST)
    history = BacktestHistory(tmp_path / "h.jsonl")
    proposals = ProposalStore(tmp_path / "p.jsonl")
    opt = ContinuousOptimizer(config, history, proposals, fetch_fn=lambda *a, **k: None, seed=0)
    history.log(_fake_trial(oos_sharpe=0.5))  # only one prior trial
    noisy_curve = 100_000 * np.cumprod(1 + np.random.default_rng(3).normal(0.0, 0.01, 200))
    assert opt._passes_deflated_sharpe(MR_INST, _fake_trial(oos_curve=noisy_curve)) is True
    # ...but a curve too short to assess is always rejected
    assert opt._passes_deflated_sharpe(MR_INST, _fake_trial(oos_curve=[100_000, 100_100])) is False


def test_equity_returns_scrubs_non_finite():
    assert equity_returns([100.0, 110.0, 121.0]).tolist() == pytest.approx([0.1, 0.1])
    assert equity_returns([100.0]).size == 0


# -- held-out split: selection and confirmation on separate data --------------

def test_split_search_holdout_disjoint_and_embargoed():
    df = make_ohlcv(np.linspace(100, 200, 1000), freq="1h")
    search, holdout = split_search_holdout(df, holdout_frac=0.15, embargo_frac=0.02)
    assert holdout is not None
    assert len(holdout) == 150
    # the holdout tail starts strictly after the search slice, with an embargo gap
    assert search.index[-1] < holdout.index[0]
    gap = df.index.get_loc(holdout.index[0]) - df.index.get_loc(search.index[-1]) - 1
    assert gap >= 1


def test_split_search_holdout_none_when_too_small():
    df = make_ohlcv(np.linspace(100, 110, 40), freq="1h")
    search, holdout = split_search_holdout(df)
    assert holdout is None
    assert len(search) == len(df)


# -- non-overlapping, purged folds --------------------------------------------

def test_validation_folds_are_non_overlapping_and_purged():
    df = make_ohlcv(np.linspace(100, 300, 1200), freq="4h")
    folds = validation_folds(df, n_folds=4)
    assert len(folds) >= 2
    for earlier, later in zip(folds, folds[1:]):
        # no shared bars, and a purge gap between adjacent folds
        assert earlier.index[-1] < later.index[0]


# -- perturbation bar now keys off the candidate's OWN Sharpe ------------------

def test_perturbation_requires_stability_near_own_sharpe():
    # within 20% of its own 1.0 (floor 0.8) and above baseline 0.5 -> passes
    assert perturbation_passes(0.85, candidate_oos=1.0, baseline_oos=0.5)
    # above the old "beat baseline" bar (0.6 > 0.5) but collapses below 0.8 floor
    # -> the tightened bar correctly rejects it
    assert not perturbation_passes(0.6, candidate_oos=1.0, baseline_oos=0.5)
    # stable near its own value but no longer beats the baseline -> rejected
    assert not perturbation_passes(0.95, candidate_oos=1.0, baseline_oos=0.97)


# -- baseline bug fix: keyed to params, scanned across full history -----------

def test_baseline_survives_more_than_100_later_trials(tmp_path):
    """The 101-trial window bug: a baseline must stay findable no matter how
    many non-baseline trials are logged after it."""
    h = BacktestHistory(tmp_path / "h.jsonl")
    h.log(_fake_trial(oos_sharpe=1.234, is_baseline=True))
    for i in range(150):
        h.log(_fake_trial(oos_sharpe=0.1, params={"lookback": i}))
    found = h.baseline("SPX", params=dict(MR_INST.params))
    assert found is not None
    assert found["oos_sharpe"] == pytest.approx(1.234)


def test_baseline_is_keyed_to_params(tmp_path):
    h = BacktestHistory(tmp_path / "h.jsonl")
    h.log(_fake_trial(oos_sharpe=1.0, is_baseline=True))
    # a different config has no baseline yet -> forces a fresh measurement
    assert h.baseline("SPX", params={**MR_INST.params, "lookback": 99}) is None
    # the unkeyed lookup still returns the most recent baseline
    assert h.baseline("SPX") is not None


def test_ensure_baseline_does_not_rechurn(tmp_path):
    """With a baseline already present for the current params, _ensure_baseline
    reuses it and never appends a duplicate — even after 150 later trials. This
    is the regression for the 101-trial re-baselining churn."""
    df = make_ohlcv(np.linspace(100, 200, 500), freq="15min")
    config = make_config_with(MR_INST)
    h = BacktestHistory(tmp_path / "h.jsonl")
    p = ProposalStore(tmp_path / "p.jsonl")
    opt = ContinuousOptimizer(config, h, p, fetch_fn=lambda *a, **k: df, seed=0)
    search_df, _ = split_search_holdout(df)

    s1 = opt._ensure_baseline(MR_INST, search_df)
    for i in range(150):
        h.log(_fake_trial(oos_sharpe=0.0, params={"lookback": i}))
    s2 = opt._ensure_baseline(MR_INST, search_df)

    assert s1 == s2
    n_baselines = sum(1 for t in h.read("SPX", limit=None) if t.get("is_baseline"))
    assert n_baselines == 1


def test_ensure_baseline_remeasures_when_params_change(tmp_path):
    """A promotion that changes config.yaml must trigger a fresh baseline rather
    than grading new candidates against the superseded config."""
    df = make_ohlcv(np.linspace(100, 180, 500), freq="15min")
    config = make_config_with(MR_INST)
    h = BacktestHistory(tmp_path / "h.jsonl")
    p = ProposalStore(tmp_path / "p.jsonl")
    opt = ContinuousOptimizer(config, h, p, fetch_fn=lambda *a, **k: df, seed=0)
    search_df, _ = split_search_holdout(df)
    opt._ensure_baseline(MR_INST, search_df)

    # simulate a promotion: live params change
    changed = InstrumentConfig(**{**MR_INST.__dict__, "params": {**MR_INST.params, "entry_z": 2.5}})
    opt.config.instruments["SPX"] = changed
    opt._ensure_baseline(changed, search_df)

    baselines = [t for t in h.read("SPX", limit=None) if t.get("is_baseline")]
    assert len(baselines) == 2  # one per distinct config
    assert {b["params"]["entry_z"] for b in baselines} == {2.0, 2.5}


# ── Workstream F: parallel trial execution ──────────────────────────────────

def _trial_signature(entries):
    """Fields that must be identical regardless of worker count (timestamps and
    in-memory curves excluded)."""
    return [
        (e["instrument"], e["params"], round(e["oos_sharpe"], 6), e["oos_trades"], e.get("is_baseline", False))
        for e in entries
    ]


def test_trial_batch_logs_and_dedups(tmp_path):
    df = make_ohlcv(np.linspace(100, 220, 700), freq="15min")
    config = make_config_with(MR_INST)
    history = BacktestHistory(tmp_path / "h.jsonl")
    proposals = ProposalStore(tmp_path / "p.jsonl")
    opt = ContinuousOptimizer(config, history, proposals, fetch_fn=lambda *a, **k: df, seed=11)
    trials = opt.trial_batch(MR_INST, n=5)
    assert len(trials) >= 1
    # no duplicate parameter sets within a batch
    seen = [t.params for t in trials]
    assert len(seen) == len({tuple(sorted(p.items())) for p in seen})


def test_trial_batch_parallel_matches_serial(tmp_path):
    """The parallel path must produce byte-for-byte the same history and
    candidates as the serial path for a given seed — only the CPU-bound
    backtests move off the main process."""
    from concurrent.futures import ProcessPoolExecutor

    df = make_ohlcv(np.linspace(100, 240, 700), freq="15min")
    config = make_config_with(MR_INST)

    hist_s = BacktestHistory(tmp_path / "s.jsonl")
    opt_s = ContinuousOptimizer(config, hist_s, ProposalStore(tmp_path / "ps.jsonl"),
                                fetch_fn=lambda *a, **k: df, seed=123)
    serial = opt_s.trial_batch(MR_INST, n=4, pool=None)

    hist_p = BacktestHistory(tmp_path / "p.jsonl")
    opt_p = ContinuousOptimizer(config, hist_p, ProposalStore(tmp_path / "pp.jsonl"),
                                fetch_fn=lambda *a, **k: df, seed=123)
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = opt_p.trial_batch(MR_INST, n=4, pool=pool)

    assert [t.params for t in serial] == [t.params for t in parallel]
    assert [round(t.oos_sharpe, 6) for t in serial] == [round(t.oos_sharpe, 6) for t in parallel]
    assert _trial_signature(hist_s.read("SPX", limit=None)) == _trial_signature(hist_p.read("SPX", limit=None))
