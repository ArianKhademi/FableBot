"""Dynamic universe rotation: the pure book-selection policy and scoring."""

import json

import numpy as np

from fablebot import rotation
from tests.conftest import make_ohlcv


# ── propose_rotation: the guardrailed decision policy ────────────────────────

def test_picks_top_scorers_and_drops_stale():
    current = ["A", "B", "C"]
    scores = {"A": 1.0, "B": 0.5, "C": 0.2, "D": 2.0, "E": 1.5}
    plan = rotation.propose_rotation(current, scores, open_keys=set(), max_active=3,
                                     min_score=0.3, churn_margin=0.0)
    assert set(plan["add"]) == {"D", "E"}
    assert set(plan["drop"]) == {"B", "C"}   # C below floor, B outranked
    assert plan["keep"] == ["A"]


def test_caps_concurrency():
    scores = {k: float(v) for k, v in {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}.items()}
    plan = rotation.propose_rotation(["A"], scores, open_keys=set(), max_active=2, min_score=0.0)
    assert len(plan["target"]) == 2
    assert set(plan["target"]) == {"A", "B"}


def test_open_position_is_sticky_and_never_dropped():
    current = ["A", "B"]
    scores = {"A": 2.0, "B": -1.0, "C": 1.0}
    plan = rotation.propose_rotation(current, scores, open_keys={"B"}, max_active=2,
                                     min_score=0.3, churn_margin=0.0)
    assert "B" not in plan["drop"]          # open position kept despite worst score
    assert "B" in plan["keep"]
    assert "B" in plan["kept_sticky"]       # carried past the cap on purpose
    assert "C" in plan["add"]


def test_hysteresis_resists_churn():
    current, scores = ["A"], {"A": 1.0, "B": 1.2}
    # margin 0.5: A's effective score (1.5) beats challenger B (1.2) -> no swap
    keep = rotation.propose_rotation(current, scores, set(), max_active=1, min_score=0.3, churn_margin=0.5)
    assert keep["keep"] == ["A"] and keep["add"] == []
    # no margin: B (1.2) > A (1.0) -> swap
    swap = rotation.propose_rotation(current, scores, set(), max_active=1, min_score=0.3, churn_margin=0.0)
    assert swap["add"] == ["B"] and swap["drop"] == ["A"]


def test_min_score_floor_blocks_weak_newcomers():
    plan = rotation.propose_rotation(["A"], {"A": 1.0, "D": 0.1}, set(), max_active=3, min_score=0.3)
    assert "D" not in plan["add"]
    assert plan["keep"] == ["A"]


# ── realized-P&L tilt ────────────────────────────────────────────────────────

def _trades(instrument, pnl, risk=100.0, n=10):
    return [{"instrument": instrument, "pnl": pnl, "risk_amount": risk,
             "closed_at": f"2026-01-{d+1:02d}T00:00:00"} for d in range(n)]


def test_performance_tilt_rewards_winners_and_penalises_losers():
    trades = _trades("WIN", 100.0) + _trades("LOSE", -50.0)
    tilt = rotation.performance_tilt(trades, min_trades=5, scale=0.4, cap=0.5)
    assert tilt["WIN"] == 0.4    # +1R expectancy * 0.4
    assert tilt["LOSE"] == -0.2  # -0.5R expectancy * 0.4
    # cap binds for an extreme winner
    assert rotation.performance_tilt(_trades("HUGE", 1000.0), min_trades=5)["HUGE"] == 0.5


def test_performance_tilt_needs_enough_trades():
    tilt = rotation.performance_tilt(_trades("THIN", 100.0, n=3), min_trades=6)
    assert "THIN" not in tilt  # too few trades -> no tilt (noise)


def test_performance_tilt_skips_zero_risk_trades():
    # all risk_amount 0 -> no R can be formed -> no tilt
    tilt = rotation.performance_tilt(_trades("ZR", 50.0, risk=0.0, n=10), min_trades=6)
    assert "ZR" not in tilt


def test_losing_incumbent_demoted_past_hysteresis():
    """A losing incumbent that churn_margin alone would keep gets dropped once
    its negative realized-P&L tilt overcomes the margin."""
    current, scores = ["A"], {"A": 1.0, "B": 1.2}
    # without tilt, margin 0.5 keeps A (1.5 > 1.2) — see test_hysteresis_resists_churn
    plan = rotation.propose_rotation(current, scores, set(), max_active=1, min_score=0.3,
                                     churn_margin=0.5, performance={"A": -0.5})
    assert plan["drop"] == ["A"] and plan["add"] == ["B"]  # A effective 1.0 < B 1.2


def test_profitable_incumbent_kept_against_a_better_challenger():
    """A profitable incumbent stays even when a challenger outscores it on
    backtest tradeability, because its realized edge tips the ranking."""
    current, scores = ["A"], {"A": 1.0, "B": 1.2}
    plan = rotation.propose_rotation(current, scores, set(), max_active=1, min_score=0.3,
                                     churn_margin=0.0, performance={"A": 0.4})
    assert plan["keep"] == ["A"] and plan["add"] == []  # A effective 1.4 > B 1.2
    assert plan["performance"] == {"A": 0.4}


# ── open-position reading ────────────────────────────────────────────────────

def test_read_open_positions(tmp_path):
    p = tmp_path / "portfolio.json"
    p.write_text(json.dumps({"positions": {"OIL": {}, "GOLD": {}}}))
    assert rotation.read_open_positions(p) == {"OIL", "GOLD"}
    assert rotation.read_open_positions(tmp_path / "missing.json") == set()


def test_read_trades(tmp_path):
    p = tmp_path / "portfolio.json"
    p.write_text(json.dumps({"trades": [{"instrument": "OIL", "pnl": 5.0}]}))
    assert rotation.read_trades(p) == [{"instrument": "OIL", "pnl": 5.0}]
    assert rotation.read_trades(tmp_path / "missing.json") == []


# ── scoring a symbol ─────────────────────────────────────────────────────────

def test_score_symbol_returns_a_number():
    closes = 100 + np.cumsum(np.random.default_rng(3).normal(0.12, 0.7, 800))
    df = make_ohlcv(np.abs(closes) + 40, freq="4h")
    score = rotation.score_symbol("GOLD", "4h", df, friction=3.0, min_trades=5)
    assert isinstance(score, float)
    assert score > float("-inf")  # a trending series yields a qualifying edge
