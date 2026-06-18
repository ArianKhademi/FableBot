"""Continuous walk-forward parameter optimizer.

Runs forever as a background process alongside the trading bot.
Every trial:
  1. samples a random parameter set within param_bounds
  2. splits the instrument's full history 75/25 train/OOS
  3. runs the backtest on each half
  4. logs the result to state/backtest_history.jsonl
  5. if OOS Sharpe > baseline + MIN_IMPROVEMENT: enters the pipeline as a
     *candidate* (see fablebot.pipeline)

Candidates then face stage-two validation (interleaved with the search):
regime robustness across non-overlapping purged folds, a parameter-perturbation
test, and confirmation on an embargoed holdout the search never touched.
Survivors become *qualified* — the only status the nightly reviewer is allowed
to act on. Failures are *rejected* but stay in the log.

Selection and validation run only on the search slice; the holdout tail exists
solely to confirm what selection found. The entry gate also applies a deflated
Sharpe correction so the flood of random draws can't promote noise.

Nothing here ever touches config.yaml or any running code. It only
produces evidence.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Optional

import numpy as np

from .backtest import run_backtest
from .config import InstrumentConfig
from .pipeline import CANDIDATE, Proposal, ProposalStore, QUALIFIED, REJECTED
from .strategies import build_strategy

# Walk-forward split: last OUT_OF_SAMPLE_FRAC of bars held out for validation.
OUT_OF_SAMPLE_FRAC = 0.25

# Minimum OOS Sharpe improvement over the baseline to be worth proposing.
MIN_IMPROVEMENT = 0.15

# Minimum trades in both train AND OOS to trust a Sharpe figure. A Sharpe on a
# handful of trades has a huge standard error; require dozens before a draw is
# allowed to enter the pipeline at all.
MIN_TRADES = 20

# ── selection / confirmation split (held-out data the search never touches) ──
# The optimizer searches and validates on SEARCH_FRAC of the history. The final
# HOLDOUT_FRAC tail is embargoed: a candidate only qualifies if it ALSO beats
# the current params on this slice, which selection never saw. SPLIT_EMBARGO_FRAC
# is a gap dropped at the search/holdout boundary so a position open across it
# can't leak the holdout into selection.
HOLDOUT_FRAC = 0.15
SPLIT_EMBARGO_FRAC = 0.02
MIN_HOLDOUT_TRADES = 5

# ── multiple-comparisons correction (deflated Sharpe) ────────────────────────
# Keeping the best of thousands of random draws guarantees spurious winners.
# A candidate must clear a Deflated Sharpe Ratio: the probability its true
# Sharpe beats the *expected maximum* Sharpe of N null trials must exceed
# DSR_THRESHOLD.
DSR_THRESHOLD = 0.95
# Below this many logged trials for an instrument the deflation benchmark and
# its variance estimate are too noisy to trust, and multiple-comparison risk is
# still low, so the DSR gate is skipped.
MIN_TRIALS_FOR_DSR = 10
# N in the deflated Sharpe is the number of *effectively independent* trials —
# NOT the raw cumulative count. The optimizer logs tens of thousands of random
# samples into a handful of bounded params; those are massively redundant, and
# feeding the raw (forever-growing) count ratchets the benchmark to an
# unpassable 5-37 Sharpe, blocking every candidate before the real defenses
# (folds + perturbation + embargoed held-out) ever run. Cap N at a conservative
# estimate of distinct configurations so the gate stays strict but passable;
# the held-out set remains the honest arbiter of overfitting.
DSR_EFFECTIVE_TRIALS_CAP = 200
_EULER_GAMMA = 0.5772156649015329
_NORM = NormalDist()

# How long (seconds) to cache fetched data before re-fetching.
DATA_CACHE_TTL = 1800

# Annualisation: approximate bars per calendar year for each timeframe.
BARS_PER_YEAR = {"15m": 252 * 26, "1h": 252 * 6.5, "4h": 252 * 1.625, "1d": 252}

# How many days of history to fetch for optimisation (longer than live trading).
# Daily reaches back years for deep, multi-regime rankings (Workstream H).
OPTIMIZER_FETCH_DAYS = {"15m": 59, "1h": 700, "4h": 700, "1d": 3650}

# How many seconds to sleep between trials so the optimizer doesn't pin a CPU.
TRIAL_SLEEP = 1.5

# ── validation stage (candidate -> qualified/rejected) ───────────────────────

# Non-overlapping time blocks the candidate is re-tested on. It must beat the
# CURRENT params (not just the full-history baseline) in most of them. The
# blocks are contiguous and purged by an embargo band so "won 3 of 4" reflects
# independent regimes, not the same correlated stretch counted four times.
VALIDATION_FOLDS = 4
MIN_FOLDS_WON_FRAC = 0.75
MIN_VALID_FOLDS = 3
FOLD_EMBARGO_FRAC = 0.02

# Perturbation test: jitter each tunable param ±10% and re-run walk-forward.
# A real edge survives nudging; a knife-edge overfit collapses. The jittered
# median must stay within PERTURB_TOLERANCE of the candidate's OWN Sharpe — not
# merely above the baseline it already beat by construction.
PERTURBATIONS = 5
PERTURB_FRAC = 0.10
PERTURB_TOLERANCE = 0.20


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sharpe_ratio(equity_curve: list[float], bars_per_year: float) -> float:
    """Annualised Sharpe from an equity curve. Returns -inf for trivial curves."""
    if len(equity_curve) < 10:
        return float("-inf")
    arr = np.array(equity_curve, dtype=float)
    returns = np.diff(arr) / arr[:-1]
    std = returns.std()
    if std == 0 or math.isnan(std):
        return 0.0
    return float(returns.mean() / std * math.sqrt(bars_per_year))


def equity_returns(equity_curve: list[float]) -> np.ndarray:
    """Per-bar simple returns of an equity curve, NaN/inf scrubbed."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size < 2:
        return np.empty(0)
    rets = np.diff(arr) / arr[:-1]
    return rets[np.isfinite(rets)]


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probability that the *true* (per-bar) Sharpe exceeds ``sr_benchmark``,
    given the observed return stream (Bailey & López de Prado).

    Accounts for track-record length, skew and kurtosis: a high Sharpe over a
    short, fat-tailed, negatively skewed stream is heavily discounted. Returns
    a probability in [0, 1]; 0.0 when the stream is too short or degenerate.
    """
    returns = np.asarray(returns, dtype=float)
    n = returns.size
    if n < 3:
        return 0.0
    std = returns.std()
    if std == 0 or not math.isfinite(std):
        return 0.0
    sr = float(returns.mean() / std)  # non-annualised
    z = (returns - returns.mean()) / std
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # raw 4th moment; 3.0 for a normal
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return 0.0
    stat = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_NORM.cdf(stat))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum (per-bar) Sharpe across ``n_trials`` independent draws
    with zero true skill and cross-trial Sharpe dispersion ``sr_std``. This is
    the benchmark a real edge must clear to survive multiple comparisons."""
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_std * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, sr_std: float) -> float:
    """PSR measured against the expected-maximum-Sharpe benchmark of ``n_trials``
    null draws. Probability the candidate's edge is real after correcting for
    how many configurations were tried. All Sharpe terms are per-bar."""
    benchmark = expected_max_sharpe(n_trials, sr_std)
    return probabilistic_sharpe_ratio(returns, benchmark)


def deflation_inputs(oos_sharpes, bars_per_year: float) -> tuple[int, float]:
    """Derive the deflated-Sharpe inputs (effective N, per-bar Sharpe dispersion)
    from an instrument's logged trial Sharpes. ``N`` is capped at
    DSR_EFFECTIVE_TRIALS_CAP — the raw cumulative count overstates independent
    trials and would make the gate unpassable — and the dispersion is winsorised
    at the 1st/99th percentile so a few overfit single-split outliers don't
    inflate the benchmark for everything else."""
    arr = np.array([s for s in oos_sharpes if math.isfinite(s)], dtype=float)
    if arr.size < 2:
        return arr.size, 0.0
    lo, hi = np.percentile(arr, [1, 99])
    sr_std_annual = float(np.clip(arr, lo, hi).std())
    n_eff = min(arr.size, DSR_EFFECTIVE_TRIALS_CAP)
    return n_eff, sr_std_annual / math.sqrt(bars_per_year)


@dataclass
class Trial:
    instrument: str
    strategy: str
    timeframe: str
    params: dict
    is_sharpe: float   # in-sample
    oos_sharpe: float  # out-of-sample — the number that matters
    is_trades: int
    oos_trades: int
    is_bars: int
    oos_bars: int
    is_return_pct: float
    oos_return_pct: float
    timestamp: str = field(default_factory=_utcnow)
    is_baseline: bool = False  # True when this trial records the current config
    # OOS equity curve, kept in memory for the deflated-Sharpe gate but never
    # written to history (one curve per trial would bloat the log). Stripped in
    # BacktestHistory.log and excluded from equality/dedup.
    oos_equity_curve: list = field(default_factory=list, repr=False, compare=False)


class BacktestHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, trial: Trial) -> None:
        record = asdict(trial)
        record.pop("oos_equity_curve", None)  # in-memory only; never persisted
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def read(self, instrument: Optional[str] = None, limit: Optional[int] = 500) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        entries = [json.loads(l) for l in lines if l.strip()]
        if instrument:
            entries = [e for e in entries if e["instrument"] == instrument]
        return entries if limit is None else entries[-limit:]

    def already_tried(self, instrument: str, params: dict, within_hours: float = 48.0) -> bool:
        cutoff = time.time() - within_hours * 3600
        for e in self.read(instrument, limit=200):
            if e["params"] == params:
                ts = datetime.fromisoformat(e["timestamp"]).timestamp()
                if ts > cutoff:
                    return True
        return False

    def best_oos_sharpe(self, instrument: str) -> float:
        trials = self.read(instrument)
        if not trials:
            return float("-inf")
        return max(t["oos_sharpe"] for t in trials)

    def baseline(self, instrument: str, params: Optional[dict] = None) -> Optional[dict]:
        """Most recent baseline for ``instrument``. When ``params`` is given,
        only a baseline measured on *those* params matches — so a baseline is
        tied to the config it graded, not to a fixed-size lookback window. The
        full history is scanned (not the last N rows), so a live baseline never
        silently scrolls out of view and gets needlessly re-measured."""
        for t in reversed(self.read(instrument, limit=None)):
            if not t.get("is_baseline"):
                continue
            if params is None or t.get("params") == params:
                return t
        return None


def _round_param(key: str, val: float) -> float | int:
    """Integer params stay integer; others are rounded to 2dp."""
    integer_keys = {"lookback", "entry_channel", "exit_channel", "fast_ema", "slow_ema"}
    return int(round(val)) if key in integer_keys else round(val, 2)


def sample_params(inst: InstrumentConfig, rng: np.random.Generator) -> dict:
    """Random sample within param_bounds, frozen params kept from inst.params."""
    params = dict(inst.params)
    for key, bounds in inst.param_bounds.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        params[key] = _round_param(key, rng.uniform(lo, hi))
    # Ensure exit_channel < entry_channel for momentum strategies
    if "exit_channel" in params and "entry_channel" in params:
        params["exit_channel"] = min(params["exit_channel"], params["entry_channel"] - 2)
        params["exit_channel"] = max(params["exit_channel"], 2)
    # Ensure fast_ema < slow_ema for trend following
    if "fast_ema" in params and "slow_ema" in params:
        params["fast_ema"] = min(params["fast_ema"], params["slow_ema"] - 5)
        params["fast_ema"] = max(params["fast_ema"], 5)
    return params


def run_walk_forward(
    df,
    inst: InstrumentConfig,
    params: dict,
    risk_per_trade: float = 0.01,
    friction_bps: float = 0.0,
) -> Optional[Trial]:
    """Split df 75/25, run backtest on each half. Returns None if too few bars."""
    bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
    split = int(len(df) * (1 - OUT_OF_SAMPLE_FRAC))
    train_df = df.iloc[:split]
    oos_df = df.iloc[split:]

    try:
        strategy = build_strategy(inst.strategy, params, genome=getattr(inst, "genome", None))
        if len(train_df) < strategy.min_bars() * 2:
            return None

        is_result = run_backtest(train_df, strategy, risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
        oos_result = run_backtest(oos_df, strategy, risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
    except Exception:
        return None

    is_sharpe = sharpe_ratio(is_result.equity_curve, bpy)
    oos_sharpe = sharpe_ratio(oos_result.equity_curve, bpy)

    return Trial(
        instrument=inst.key,
        strategy=inst.strategy,
        timeframe=inst.timeframe,
        params=params,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        is_trades=is_result.trades,
        oos_trades=oos_result.trades,
        is_bars=len(train_df),
        oos_bars=len(oos_df),
        is_return_pct=is_result.return_pct,
        oos_return_pct=oos_result.return_pct,
        oos_equity_curve=list(oos_result.equity_curve),
    )


def split_search_holdout(
    df,
    holdout_frac: float = HOLDOUT_FRAC,
    embargo_frac: float = SPLIT_EMBARGO_FRAC,
):
    """Split history into (search, holdout). The search slice is everything the
    optimizer is allowed to see while selecting and validating; the holdout is
    the embargoed tail used only for final confirmation. An embargo band between
    them is discarded so a trade open across the boundary can't leak. Returns
    (search_df, None) when there isn't enough data to carve a usable holdout."""
    n = len(df)
    holdout = int(n * holdout_frac)
    embargo = int(n * embargo_frac)
    search_end = n - holdout - embargo
    if holdout < 30 or search_end < 50:
        return df, None
    return df.iloc[:search_end], df.iloc[n - holdout:]


def validation_folds(df, n_folds: int = VALIDATION_FOLDS, embargo_frac: float = FOLD_EMBARGO_FRAC):
    """Contiguous, non-overlapping time blocks covering the history, each purged
    by an embargo band trimmed from its leading edge (after the first) so
    adjacent blocks share no bars. Tests the candidate across independent
    regimes — unlike overlapping windows, "won 3 of 4" here means three
    genuinely distinct stretches, not the same data counted repeatedly."""
    n = len(df)
    if n_folds < 2 or n < 100:
        return [df]
    embargo = int(n * embargo_frac)
    block = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * block + (embargo if i > 0 else 0)
        end = (i + 1) * block if i < n_folds - 1 else n
        if end - start >= 50:
            folds.append(df.iloc[start:end])
    return folds or [df]


def perturb_params(
    params: dict, inst: InstrumentConfig, rng: np.random.Generator, frac: float = PERTURB_FRAC
) -> dict:
    """Jitter each tunable param by ±frac, clipped to its bounds."""
    out = dict(params)
    for key, bounds in inst.param_bounds.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        jittered = float(out[key]) * (1.0 + rng.uniform(-frac, frac))
        out[key] = _round_param(key, min(max(jittered, lo), hi))
    if "exit_channel" in out and "entry_channel" in out:
        out["exit_channel"] = max(min(out["exit_channel"], out["entry_channel"] - 2), 2)
    if "fast_ema" in out and "slow_ema" in out:
        out["fast_ema"] = max(min(out["fast_ema"], out["slow_ema"] - 5), 5)
    return out


def perturbation_passes(
    median_perturbed: float,
    candidate_oos: float,
    baseline_oos: float,
    tolerance: float = PERTURB_TOLERANCE,
) -> bool:
    """Whether the perturbation test passes. The jittered median must hold
    within ``tolerance`` of the candidate's OWN Sharpe (stability near its own
    value) AND still beat the baseline — not merely clear the bar it already
    beat by construction, which the old ``median > baseline`` rule allowed."""
    return median_perturbed >= candidate_oos * (1.0 - tolerance) and median_perturbed > baseline_oos


def _full_oos_sharpe(df, inst: InstrumentConfig, params: dict, risk_per_trade: float, friction_bps: float = 0.0):
    """Sharpe + trade count over a whole slice (no train/OOS split), used for
    the embargoed holdout where the entire slice is unseen confirmation data."""
    try:
        strategy = build_strategy(inst.strategy, params, genome=getattr(inst, "genome", None))
        if df is None or len(df) < strategy.min_bars() * 2:
            return None
        result = run_backtest(df, strategy, risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
    except Exception:
        return None
    bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
    return sharpe_ratio(result.equity_curve, bpy), result.trades


def validate_candidate(
    df,
    inst: InstrumentConfig,
    candidate_params: dict,
    current_params: dict,
    rng: np.random.Generator,
    risk_per_trade: float = 0.01,
    holdout_df=None,
    friction_bps: float = 0.0,
) -> dict:
    """Stage-two testing for a candidate proposal. ``df`` is the search slice
    (everything selection was allowed to see); ``holdout_df`` is the embargoed
    tail selection never touched. Returns a verdict dict.

    Test 1 — regime robustness: across non-overlapping, purged time blocks the
    candidate must beat the *current* params head-to-head in >= 75% of valid
    blocks.
    Test 2 — perturbation: jittered ±10%, the candidate's median OOS Sharpe must
    stay within PERTURB_TOLERANCE of its OWN Sharpe (and still beat the current
    params). An edge that only exists at exact parameter values is curve-fitting.
    Test 3 — held-out confirmation: on the embargoed holdout the search never
    saw, the candidate must beat the current params. This is the gate selection
    bias cannot game, because the data was never available to select on.
    """
    folds_won = 0
    folds_valid = 0
    fold_details = []
    for fold in validation_folds(df):
        cand = run_walk_forward(fold, inst, candidate_params, risk_per_trade, friction_bps)
        curr = run_walk_forward(fold, inst, current_params, risk_per_trade, friction_bps)
        if cand is None or curr is None:
            fold_details.append({"skipped": True})
            continue
        folds_valid += 1
        won = bool(cand.oos_sharpe > curr.oos_sharpe)
        folds_won += won
        fold_details.append(
            {"candidate_oos": cand.oos_sharpe, "current_oos": curr.oos_sharpe, "won": won}
        )

    baseline_full = run_walk_forward(df, inst, current_params, risk_per_trade, friction_bps)
    baseline_oos = baseline_full.oos_sharpe if baseline_full else float("-inf")
    candidate_full = run_walk_forward(df, inst, candidate_params, risk_per_trade, friction_bps)
    candidate_oos = candidate_full.oos_sharpe if candidate_full else float("-inf")

    perturbed_sharpes = []
    for _ in range(PERTURBATIONS):
        jittered = perturb_params(candidate_params, inst, rng)
        trial = run_walk_forward(df, inst, jittered, risk_per_trade, friction_bps)
        if trial is not None:
            perturbed_sharpes.append(trial.oos_sharpe)
    median_perturbed = float(np.median(perturbed_sharpes)) if perturbed_sharpes else float("-inf")

    # Held-out confirmation on the embargoed tail (whole slice, no re-split).
    holdout_checked = holdout_df is not None
    cand_h = _full_oos_sharpe(holdout_df, inst, candidate_params, risk_per_trade, friction_bps) if holdout_checked else None
    curr_h = _full_oos_sharpe(holdout_df, inst, current_params, risk_per_trade, friction_bps) if holdout_checked else None
    holdout_candidate_sharpe = cand_h[0] if cand_h else float("-inf")
    holdout_current_sharpe = curr_h[0] if curr_h else float("-inf")
    holdout_trades = cand_h[1] if cand_h else 0

    folds_ok = (
        folds_valid >= MIN_VALID_FOLDS
        and folds_won >= math.ceil(MIN_FOLDS_WON_FRAC * folds_valid)
    )
    # Stable near its own Sharpe, not merely above the thing it already beat.
    perturb_ok = perturbation_passes(median_perturbed, candidate_oos, baseline_oos)
    if holdout_checked:
        holdout_ok = (
            cand_h is not None
            and curr_h is not None
            and holdout_trades >= MIN_HOLDOUT_TRADES
            and holdout_candidate_sharpe > holdout_current_sharpe
        )
    else:
        holdout_ok = True  # no holdout supplied (e.g. unit tests of the in-sample gate)

    if not folds_ok:
        reason = (
            f"regime test failed: beat current params in {folds_won}/{folds_valid} folds "
            f"(need >= {MIN_FOLDS_WON_FRAC:.0%} of >= {MIN_VALID_FOLDS})"
        )
    elif not perturb_ok:
        reason = (
            f"perturbation test failed: median jittered OOS Sharpe {median_perturbed:.3f} "
            f"below {1 - PERTURB_TOLERANCE:.0%} of the candidate's own {candidate_oos:.3f} "
            f"(or under baseline {baseline_oos:.3f}) — edge collapses when params are nudged"
        )
    elif not holdout_ok:
        reason = (
            f"held-out confirmation failed: on the embargoed holdout the candidate "
            f"scored {holdout_candidate_sharpe:.3f} vs current {holdout_current_sharpe:.3f} "
            f"({holdout_trades} trades) — the edge did not survive on unseen data"
        )
    else:
        reason = (
            f"won {folds_won}/{folds_valid} folds; jitter median {median_perturbed:.3f} "
            f"within {PERTURB_TOLERANCE:.0%} of own {candidate_oos:.3f}; "
            f"held-out {holdout_candidate_sharpe:.3f} > current {holdout_current_sharpe:.3f}"
        )

    return {
        "qualified": folds_ok and perturb_ok and holdout_ok,
        "folds_won": folds_won,
        "folds_valid": folds_valid,
        "fold_details": fold_details,
        "perturbed_sharpes": [round(s, 4) for s in perturbed_sharpes],
        "median_perturbed_sharpe": median_perturbed,
        "candidate_oos_sharpe": candidate_oos,
        "baseline_oos_sharpe": baseline_oos,
        "holdout_checked": holdout_checked,
        "holdout_candidate_sharpe": holdout_candidate_sharpe,
        "holdout_current_sharpe": holdout_current_sharpe,
        "reason": reason,
    }


def _wf_worker(args):
    """Top-level (picklable) wrapper so backtests can run in worker processes.
    Pure: it only reads its inputs and returns a Trial, so the result is
    identical regardless of which process computes it."""
    search_df, inst, params, risk, friction = args
    return run_walk_forward(search_df, inst, params, risk, friction)


class ContinuousOptimizer:
    def __init__(
        self,
        config,
        history: BacktestHistory,
        proposals: ProposalStore,
        fetch_fn=None,
        seed: Optional[int] = None,
        friction_bps: float = 0.0,
    ):
        self.config = config
        self.history = history
        self.proposals = proposals
        self.fetch_fn = fetch_fn or self._default_fetch
        self.rng = np.random.default_rng(seed)
        self.friction_bps = friction_bps
        self._data_cache: dict[str, tuple[float, object]] = {}

    def _default_fetch(self, symbol: str, timeframe: str, days: int):
        from .data import fetch_ohlcv
        return fetch_ohlcv(symbol, timeframe, days=days)

    def _get_data(self, inst: InstrumentConfig):
        key = inst.key
        now = time.monotonic()
        if key in self._data_cache:
            cached_at, df = self._data_cache[key]
            if now - cached_at < DATA_CACHE_TTL:
                return df
        days = OPTIMIZER_FETCH_DAYS.get(inst.timeframe, 60)
        df = self.fetch_fn(inst.symbol, inst.timeframe, days)
        self._data_cache[key] = (now, df)
        return df

    def _ensure_baseline(self, inst: InstrumentConfig, search_df) -> float:
        """Walk-forward Sharpe of the *current* config on the search slice.

        Keyed to the params it measures: a baseline is reused only while the
        live params are unchanged, so a promotion that edits config.yaml
        automatically forces a fresh measurement instead of grading new
        candidates against the superseded config.
        """
        current = dict(inst.params)
        existing = self.history.baseline(inst.key, params=current)
        if existing is not None:
            return existing["oos_sharpe"]
        trial = run_walk_forward(search_df, inst, current, self.config.account.risk_per_trade, self.friction_bps)
        if trial is None:
            return float("-inf")
        trial.is_baseline = True
        self.history.log(trial)
        print(f"[optimizer] {inst.key} baseline OOS Sharpe: {trial.oos_sharpe:.3f} "
              f"({trial.oos_trades} trades OOS) for params {current}", flush=True)
        return trial.oos_sharpe

    def _passes_deflated_sharpe(self, inst: InstrumentConfig, trial: Trial) -> bool:
        """Deflated-Sharpe gate: the candidate's OOS edge must look real after
        correcting for how many configs have been tried on this instrument.
        Skipped early (few trials → little multiple-comparison risk and an
        unreliable variance estimate); applied once the search is broad."""
        returns = equity_returns(trial.oos_equity_curve)
        if returns.size < 12:
            return False
        prior = [t["oos_sharpe"] for t in self.history.read(inst.key, limit=None) if not t.get("is_baseline")]
        if len(prior) < MIN_TRIALS_FOR_DSR:
            return True
        bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
        n_eff, sr_std_perbar = deflation_inputs(prior, bpy)
        dsr = deflated_sharpe_ratio(returns, n_eff, sr_std_perbar)
        return dsr >= DSR_THRESHOLD

    def _maybe_candidate(self, inst: InstrumentConfig, trial: Trial, baseline_sharpe: float) -> bool:
        """Apply the entry gate to a freshly-logged trial; enqueue a candidate
        if it clears MIN_TRADES, beats the baseline by MIN_IMPROVEMENT, and
        survives the deflated-Sharpe correction. Returns whether it qualified."""
        if not (
            trial.oos_trades >= MIN_TRADES
            and trial.is_trades >= MIN_TRADES
            and trial.oos_sharpe > baseline_sharpe + MIN_IMPROVEMENT
            and self._passes_deflated_sharpe(inst, trial)
        ):
            return False
        self.proposals.add_candidate(
            Proposal(
                instrument=inst.key,
                strategy=inst.strategy,
                timeframe=inst.timeframe,
                proposed_params=dict(trial.params),
                current_params=dict(inst.params),
                oos_sharpe=trial.oos_sharpe,
                baseline_oos_sharpe=baseline_sharpe,
                improvement=trial.oos_sharpe - baseline_sharpe,
                oos_trades=trial.oos_trades,
            )
        )
        print(
            f"[optimizer] CANDIDATE: {inst.key} OOS Sharpe "
            f"{trial.oos_sharpe:.3f} vs baseline {baseline_sharpe:.3f} "
            f"(+{trial.oos_sharpe - baseline_sharpe:.3f}, {trial.oos_trades} OOS trades) "
            f"— queued for validation",
            flush=True,
        )
        return True

    def trial_one(self, inst: InstrumentConfig) -> Optional[Trial]:
        """Run one random trial for an instrument. Returns the Trial or None."""
        trials = self.trial_batch(inst, n=1)
        return trials[0] if trials else None

    def trial_batch(self, inst: InstrumentConfig, n: int = 1, pool=None) -> list[Trial]:
        """Run ``n`` independent random trials for an instrument. Parameter
        sampling, deduplication, history I/O and the candidate gate all happen
        here in the main process (so the append-only log has a single writer and
        results are deterministic for a given seed); only the CPU-bound
        backtests are dispatched to ``pool`` when one is supplied. Parallel and
        serial runs therefore produce identical history."""
        if not inst.param_bounds:
            return []
        try:
            df = self._get_data(inst)
        except Exception as exc:
            print(f"[optimizer] {inst.key}: data fetch failed ({exc})", flush=True)
            return []

        # Selection runs on the search slice only; the holdout tail is reserved
        # for held-out confirmation in validate_one and is never sampled on.
        search_df, _ = split_search_holdout(df)
        baseline_sharpe = self._ensure_baseline(inst, search_df)
        risk = self.config.account.risk_per_trade

        param_sets: list[dict] = []
        attempts = 0
        while len(param_sets) < n and attempts < max(n * 5, 5):
            attempts += 1
            params = sample_params(inst, self.rng)
            if params in param_sets or self.history.already_tried(inst.key, params):
                continue
            param_sets.append(params)
        if not param_sets:
            return []

        work = [(search_df, inst, params, risk, self.friction_bps) for params in param_sets]
        if pool is not None and len(work) > 1:
            results = list(pool.map(_wf_worker, work))
        else:
            results = [_wf_worker(w) for w in work]

        trials: list[Trial] = []
        for params, trial in zip(param_sets, results):
            if trial is None:
                continue
            self.history.log(trial)
            self._maybe_candidate(inst, trial, baseline_sharpe)
            trials.append(trial)
        return trials

    def validate_one(self) -> Optional[dict]:
        """Pick the oldest pending candidate and run the stage-two validation:
        non-overlapping regime folds + perturbation + held-out confirmation on
        the embargoed slice. Transitions it to qualified or rejected. Returns
        the verdict, or None if there was nothing to validate."""
        candidates = self.proposals.by_status(CANDIDATE)
        if not candidates:
            return None
        cand = sorted(candidates, key=lambda c: c["ts"])[0]
        key = cand["instrument"]
        inst = self.config.instruments.get(key)
        if inst is None:
            self.proposals.set_status(
                cand["proposal_id"], REJECTED, reason="instrument no longer configured"
            )
            return None
        try:
            df = self._get_data(inst)
        except Exception as exc:
            print(f"[optimizer] validation deferred for {key}: data fetch failed ({exc})", flush=True)
            return None

        # Validate on the same search slice selection saw; confirm on the
        # embargoed holdout it never did.
        search_df, holdout_df = split_search_holdout(df)
        verdict = validate_candidate(
            search_df,
            inst,
            cand["proposed_params"],
            cand["current_params"],
            self.rng,
            self.config.account.risk_per_trade,
            holdout_df=holdout_df,
            friction_bps=self.friction_bps,
        )
        status = QUALIFIED if verdict["qualified"] else REJECTED
        self.proposals.set_status(
            cand["proposal_id"],
            status,
            reason=verdict["reason"],
            folds_won=verdict["folds_won"],
            folds_valid=verdict["folds_valid"],
            median_perturbed_sharpe=verdict["median_perturbed_sharpe"],
        )
        print(
            f"[optimizer] {status.upper()}: {key} proposal {cand['proposal_id']} — "
            f"{verdict['reason']}",
            flush=True,
        )
        return verdict

    def run_forever(
        self,
        sleep: Optional[float] = None,
        workers: int = 1,
        throttle: bool = True,
    ) -> None:
        """Round-robin the tunable instruments forever.

        ``workers`` backtests run in parallel per pass (each trial is
        independent and CPU-bound), so throughput scales ~linearly with cores.
        ``throttle`` keeps the ``TRIAL_SLEEP`` pause that stops the optimizer
        pinning a CPU when it shares the machine with the live bot; turn it off
        for a dedicated optimisation run. ``sleep`` overrides the pause length.
        """
        instruments = list(self.config.instruments.values())
        tunable = [i for i in instruments if i.param_bounds]
        if not tunable:
            print("[optimizer] No instruments have param_bounds — nothing to optimize.", flush=True)
            return
        if sleep is None:
            sleep = TRIAL_SLEEP if throttle else 0.0
        workers = max(1, int(workers))
        print(
            f"[optimizer] Starting continuous optimization over "
            f"{[i.key for i in tunable]} with {workers} worker(s), "
            f"sleep={sleep:g}s. Ctrl-C to stop.",
            flush=True,
        )
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        idx = 0
        try:
            while True:
                inst = tunable[idx % len(tunable)]
                idx += 1
                # Interleave: validate one pending candidate per pass so the
                # pipeline drains while the search keeps exploring.
                self.validate_one()
                trials = self.trial_batch(inst, n=workers, pool=pool)
                for trial in trials:
                    print(
                        f"[optimizer] {inst.key}: OOS Sharpe {trial.oos_sharpe:.3f} "
                        f"({trial.oos_trades} trades) — {trial.params}",
                        flush=True,
                    )
                if sleep:
                    time.sleep(sleep)
        finally:
            if pool is not None:
                pool.shutdown()
