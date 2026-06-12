"""Continuous walk-forward parameter optimizer.

Runs forever as a background process alongside the trading bot.
Every trial:
  1. samples a random parameter set within param_bounds
  2. splits the instrument's full history 75/25 train/OOS
  3. runs the backtest on each half
  4. logs the result to state/backtest_history.jsonl
  5. if OOS Sharpe > baseline + MIN_IMPROVEMENT: adds to state/proposals.jsonl

The nightly reviewer reads both files so it can see everything that has
been tried and promote genuinely better configurations.

Nothing here ever touches config.yaml or any running code. It only
produces evidence.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .backtest import run_backtest
from .config import InstrumentConfig
from .strategies import build_strategy

# Walk-forward split: last OUT_OF_SAMPLE_FRAC of bars held out for validation.
OUT_OF_SAMPLE_FRAC = 0.25

# Minimum OOS Sharpe improvement over the baseline to be worth proposing.
MIN_IMPROVEMENT = 0.15

# Minimum trades in both train AND OOS to trust a Sharpe figure.
MIN_TRADES = 8

# How long (seconds) to cache fetched data before re-fetching.
DATA_CACHE_TTL = 1800

# Annualisation: approximate bars per calendar year for each timeframe.
BARS_PER_YEAR = {"15m": 252 * 26, "1h": 252 * 6.5, "4h": 252 * 1.625}

# How many days of history to fetch for optimisation (longer than live trading).
OPTIMIZER_FETCH_DAYS = {"15m": 59, "1h": 700, "4h": 700}

# How many seconds to sleep between trials so the optimizer doesn't pin a CPU.
TRIAL_SLEEP = 1.5


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


@dataclass
class Proposal:
    instrument: str
    strategy: str
    timeframe: str
    proposed_params: dict
    current_params: dict
    oos_sharpe: float
    baseline_oos_sharpe: float
    improvement: float
    oos_trades: int
    timestamp: str = field(default_factory=_utcnow)
    promoted: bool = False  # set True once the reviewer acts on it


class BacktestHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, trial: Trial) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(trial)) + "\n")

    def read(self, instrument: Optional[str] = None, limit: int = 500) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        entries = [json.loads(l) for l in lines if l.strip()]
        if instrument:
            entries = [e for e in entries if e["instrument"] == instrument]
        return entries[-limit:]

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

    def baseline(self, instrument: str) -> Optional[dict]:
        for t in reversed(self.read(instrument, limit=100)):
            if t.get("is_baseline"):
                return t
        return None


class ProposalQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, proposal: Proposal) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(proposal)) + "\n")

    def read_pending(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(l)
            for l in self.path.read_text().splitlines()
            if l.strip() and not json.loads(l).get("promoted")
        ]


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
) -> Optional[Trial]:
    """Split df 75/25, run backtest on each half. Returns None if too few bars."""
    bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
    split = int(len(df) * (1 - OUT_OF_SAMPLE_FRAC))
    train_df = df.iloc[:split]
    oos_df = df.iloc[split:]

    try:
        strategy = build_strategy(inst.strategy, params)
        if len(train_df) < strategy.min_bars() * 2:
            return None

        is_result = run_backtest(train_df, strategy, risk_per_trade=risk_per_trade, instrument=inst.key)
        oos_result = run_backtest(oos_df, strategy, risk_per_trade=risk_per_trade, instrument=inst.key)
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
    )


class ContinuousOptimizer:
    def __init__(
        self,
        config,
        history: BacktestHistory,
        proposals: ProposalQueue,
        fetch_fn=None,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.history = history
        self.proposals = proposals
        self.fetch_fn = fetch_fn or self._default_fetch
        self.rng = np.random.default_rng(seed)
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

    def _ensure_baseline(self, inst: InstrumentConfig, df) -> float:
        """Compute the current config's walk-forward Sharpe once per run."""
        existing = self.history.baseline(inst.key)
        if existing is not None:
            return existing["oos_sharpe"]
        trial = run_walk_forward(df, inst, dict(inst.params), self.config.account.risk_per_trade)
        if trial is None:
            return float("-inf")
        trial.is_baseline = True
        self.history.log(trial)
        print(f"[optimizer] {inst.key} baseline OOS Sharpe: {trial.oos_sharpe:.3f} "
              f"({trial.oos_trades} trades OOS)", flush=True)
        return trial.oos_sharpe

    def trial_one(self, inst: InstrumentConfig) -> Optional[Trial]:
        """Run one random trial for an instrument. Returns the Trial or None."""
        if not inst.param_bounds:
            return None
        try:
            df = self._get_data(inst)
        except Exception as exc:
            print(f"[optimizer] {inst.key}: data fetch failed ({exc})", flush=True)
            return None

        baseline_sharpe = self._ensure_baseline(inst, df)
        params = sample_params(inst, self.rng)

        if self.history.already_tried(inst.key, params):
            return None

        trial = run_walk_forward(df, inst, params, self.config.account.risk_per_trade)
        if trial is None:
            return None

        self.history.log(trial)

        if (
            trial.oos_trades >= MIN_TRADES
            and trial.is_trades >= MIN_TRADES
            and trial.oos_sharpe > baseline_sharpe + MIN_IMPROVEMENT
            and trial.oos_sharpe > trial.oos_sharpe * 0.0  # always true; placeholder for future filters
        ):
            proposal = Proposal(
                instrument=inst.key,
                strategy=inst.strategy,
                timeframe=inst.timeframe,
                proposed_params=params,
                current_params=dict(inst.params),
                oos_sharpe=trial.oos_sharpe,
                baseline_oos_sharpe=baseline_sharpe,
                improvement=trial.oos_sharpe - baseline_sharpe,
                oos_trades=trial.oos_trades,
            )
            self.proposals.add(proposal)
            print(
                f"[optimizer] PROPOSAL: {inst.key} OOS Sharpe "
                f"{trial.oos_sharpe:.3f} vs baseline {baseline_sharpe:.3f} "
                f"(+{trial.oos_sharpe - baseline_sharpe:.3f}, {trial.oos_trades} OOS trades)",
                flush=True,
            )
        return trial

    def run_forever(self, sleep: float = TRIAL_SLEEP) -> None:
        instruments = list(self.config.instruments.values())
        tunable = [i for i in instruments if i.param_bounds]
        if not tunable:
            print("[optimizer] No instruments have param_bounds — nothing to optimize.", flush=True)
            return
        print(
            f"[optimizer] Starting continuous optimization over "
            f"{[i.key for i in tunable]}. Ctrl-C to stop.",
            flush=True,
        )
        idx = 0
        while True:
            inst = tunable[idx % len(tunable)]
            idx += 1
            trial = self.trial_one(inst)
            if trial:
                print(
                    f"[optimizer] {inst.key}: OOS Sharpe {trial.oos_sharpe:.3f} "
                    f"({trial.oos_trades} trades) — {trial.params}",
                    flush=True,
                )
            time.sleep(sleep)
