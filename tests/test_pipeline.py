"""Proposal pipeline: append-only store, status transitions, validation gates."""

import numpy as np
import pytest

from fablebot.config import InstrumentConfig
from fablebot.optimizer import validate_candidate
from fablebot.pipeline import (
    CANDIDATE,
    PROMOTED,
    QUALIFIED,
    REJECTED,
    Proposal,
    ProposalStore,
    proposal_id,
)
from tests.conftest import make_ohlcv


def make_proposal(**overrides):
    base = dict(
        instrument="BTC",
        strategy="momentum_breakout",
        timeframe="1h",
        proposed_params={"entry_channel": 15, "exit_channel": 8},
        current_params={"entry_channel": 20, "exit_channel": 10},
        oos_sharpe=0.9,
        baseline_oos_sharpe=0.6,
        improvement=0.3,
        oos_trades=12,
    )
    base.update(overrides)
    return Proposal(**base)


# ── store mechanics ──────────────────────────────────────────────────────────

def test_candidate_enters_as_candidate(tmp_path):
    store = ProposalStore(tmp_path / "p.jsonl")
    pid = store.add_candidate(make_proposal())
    assert store.by_status(CANDIDATE)[0]["proposal_id"] == pid
    assert store.by_status(QUALIFIED) == []


def test_duplicate_candidates_not_readded(tmp_path):
    store = ProposalStore(tmp_path / "p.jsonl")
    store.add_candidate(make_proposal())
    store.add_candidate(make_proposal())  # identical params -> same id
    assert len(store.latest()) == 1


def test_status_transitions_keep_data_and_history(tmp_path):
    store = ProposalStore(tmp_path / "p.jsonl")
    pid = store.add_candidate(make_proposal())
    store.set_status(pid, QUALIFIED, reason="won 4/4 folds")
    store.set_status(pid, PROMOTED, reason="shipped in PR #9")

    state = store.latest()[pid]
    assert state["status"] == PROMOTED
    assert state["oos_sharpe"] == pytest.approx(0.9)  # original data preserved
    # Full audit trail: candidate -> qualified -> promoted
    statuses = [e["status"] for e in store.history(pid)]
    assert statuses == [CANDIDATE, QUALIFIED, PROMOTED]


def test_unknown_status_or_id_rejected(tmp_path):
    store = ProposalStore(tmp_path / "p.jsonl")
    pid = store.add_candidate(make_proposal())
    with pytest.raises(ValueError):
        store.set_status(pid, "shipped")
    with pytest.raises(KeyError):
        store.set_status("nonexistent", QUALIFIED)


def test_proposal_id_is_deterministic():
    a = proposal_id("BTC", {"entry_channel": 15})
    b = proposal_id("BTC", {"entry_channel": 15})
    c = proposal_id("BTC", {"entry_channel": 16})
    assert a == b != c


# ── validation gates ─────────────────────────────────────────────────────────

TF_INST = InstrumentConfig(
    key="GOLD",
    symbol="GC=F",
    name="Gold",
    strategy="trend_following",
    timeframe="4h",
    params={"fast_ema": 20, "slow_ema": 50, "atr_period": 14, "atr_stop_mult": 3.0},
    param_bounds={"fast_ema": [10, 30], "slow_ema": [30, 80], "atr_stop_mult": [2.0, 5.0]},
)


def trending_df(n=1000, seed=5):
    # Non-overlapping purged folds carve the history into VALIDATION_FOLDS
    # contiguous blocks; each block must still hold enough bars to warm up a
    # slow_ema=50 strategy and run a 75/25 walk-forward, so the fixture is
    # sized so all four folds stay valid.
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0.15, 1.0, n))  # drift + noise
    return make_ohlcv(closes, freq="4h")


def test_validate_candidate_returns_complete_verdict():
    df = trending_df()
    rng = np.random.default_rng(0)
    verdict = validate_candidate(
        df, TF_INST,
        candidate_params={**TF_INST.params, "fast_ema": 15},
        current_params=dict(TF_INST.params),
        rng=rng,
    )
    assert set(verdict) >= {
        "qualified", "folds_won", "folds_valid", "perturbed_sharpes",
        "median_perturbed_sharpe", "baseline_oos_sharpe", "reason",
    }
    assert verdict["folds_valid"] >= 3
    assert len(verdict["perturbed_sharpes"]) > 0


def test_identical_params_cannot_qualify():
    """A candidate identical to the current params can't beat itself in any
    fold, so the regime test must reject it."""
    df = trending_df()
    rng = np.random.default_rng(0)
    verdict = validate_candidate(
        df, TF_INST,
        candidate_params=dict(TF_INST.params),
        current_params=dict(TF_INST.params),
        rng=rng,
    )
    assert verdict["qualified"] is False
    assert verdict["folds_won"] == 0
    assert "regime test failed" in verdict["reason"]


def test_validation_records_fold_details():
    df = trending_df()
    rng = np.random.default_rng(1)
    verdict = validate_candidate(
        df, TF_INST,
        candidate_params={**TF_INST.params, "fast_ema": 12, "slow_ema": 40},
        current_params=dict(TF_INST.params),
        rng=rng,
    )
    real_folds = [f for f in verdict["fold_details"] if not f.get("skipped")]
    assert len(real_folds) == verdict["folds_valid"]
    for fold in real_folds:
        assert "candidate_oos" in fold and "current_oos" in fold


# ── held-out confirmation (selection and validation on separate data) ────────

def _heldout_scenario():
    """A candidate that clears every in-sample stage (folds 4/4, stable jitter,
    beats baseline) on a clean uptrend, plus two holdouts: a continuation trend
    it survives and a rally-then-crash it does not. Only the holdout differs, so
    it is provably the binding gate. The steadier slow EMA is the candidate; the
    choppier fast EMA is the incumbent."""
    candidate = dict(TF_INST.params)                       # fast 20 / slow 50
    current = {**TF_INST.params, "fast_ema": 10, "slow_ema": 20}
    search = make_ohlcv(100 + np.cumsum(np.random.default_rng(3).normal(0.18, 0.7, 1100)), freq="4h")
    start = float(search["close"].iloc[-1])
    favorable = make_ohlcv(
        start + np.cumsum(np.random.default_rng(20).normal(0.18, 0.7, 420)), freq="4h"
    )
    r = np.random.default_rng(21)
    up = np.cumsum(r.normal(0.5, 0.8, 220))
    crash = up[-1] + np.cumsum(r.normal(-0.9, 0.8, 220))
    adverse = make_ohlcv(start + np.concatenate([up, crash]), freq="4h")
    return candidate, current, search, favorable, adverse


def _proposals_cli():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("proposals_cli", root / "scripts" / "proposals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_overfitting_gap_flags_is_oos_divergence():
    mod = _proposals_cli()
    trials = [
        {"is_sharpe": 3.0, "oos_sharpe": 0.5},
        {"is_sharpe": 2.0, "oos_sharpe": 0.0},
        {"is_sharpe": 4.0, "oos_sharpe": 1.0},
        {"is_baseline": True, "is_sharpe": 1.0, "oos_sharpe": 1.0},  # baseline ignored
    ]
    assert mod.overfitting_gap(trials) == 2.5  # median of {2.5, 2.0, 3.0}
    assert mod.overfitting_gap([]) != mod.overfitting_gap([])  # NaN when empty


def test_overfitting_gap_ignores_non_finite():
    mod = _proposals_cli()
    trials = [
        {"is_sharpe": 3.0, "oos_sharpe": 1.0},                       # gap 2.0
        {"is_sharpe": float("inf"), "oos_sharpe": float("-inf")},    # degenerate → ignored
        {"is_sharpe": float("nan"), "oos_sharpe": 0.0},              # NaN → ignored
    ]
    assert mod.overfitting_gap(trials) == 2.0  # only the finite row counts


def test_shipping_slate_picks_one_best_per_instrument():
    mod = _proposals_cli()

    qualified = [
        {"proposal_id": "a", "instrument": "BTC", "improvement": 0.2, "oos_trades": 40},
        {"proposal_id": "b", "instrument": "BTC", "improvement": 0.5, "oos_trades": 60},  # best BTC
        {"proposal_id": "c", "instrument": "GOLD", "improvement": 0.3, "oos_trades": 30},
        {"proposal_id": "d", "instrument": "BTC", "improvement": 0.5, "oos_trades": 55},  # ties improv, fewer trades
    ]
    slate = mod.best_per_instrument(qualified)
    assert set(slate) == {"BTC", "GOLD"}            # one per instrument
    assert slate["BTC"]["proposal_id"] == "b"        # highest improvement, then trades
    assert slate["GOLD"]["proposal_id"] == "c"


def test_in_sample_pass_qualifies_without_a_holdout():
    candidate, current, search, _, _ = _heldout_scenario()
    v = validate_candidate(search, TF_INST, candidate, current, np.random.default_rng(0))
    assert v["qualified"] is True
    assert v["folds_won"] == v["folds_valid"]  # 4/4, a clean in-sample pass
    assert v["holdout_checked"] is False


def test_holdout_confirms_a_genuine_edge():
    candidate, current, search, favorable, _ = _heldout_scenario()
    v = validate_candidate(
        search, TF_INST, candidate, current, np.random.default_rng(0), holdout_df=favorable
    )
    assert v["holdout_checked"] is True
    assert v["holdout_candidate_sharpe"] > v["holdout_current_sharpe"]
    assert v["qualified"] is True


def test_holdout_rejects_an_edge_that_does_not_survive_unseen_data():
    """Identical in-sample result (folds 4/4); only the holdout changes. The
    candidate that wins in-sample but loses on the embargoed slice is rejected —
    the gate selection bias cannot game."""
    candidate, current, search, _, adverse = _heldout_scenario()
    v = validate_candidate(
        search, TF_INST, candidate, current, np.random.default_rng(0), holdout_df=adverse
    )
    assert v["folds_won"] == v["folds_valid"]  # in-sample still passes 4/4
    assert v["holdout_candidate_sharpe"] < v["holdout_current_sharpe"]
    assert v["qualified"] is False
    assert "held-out" in v["reason"]
