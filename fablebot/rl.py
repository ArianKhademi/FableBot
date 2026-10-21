"""RL agent — the last and most overfit-prone layer, kept on a short leash.

A tabular Q-learning policy learns *positioning* (flat / long / short) from a
small discretised market state. Its action space is deliberately constrained to
direction and timing: it never sizes a trade or moves a stop — the risk engine
still owns the sizing budget, ATR stops, gap handling and trailing. The agent
proposes a direction; the broker decides everything about risk.

Because RL overfits more readily than anything else here, it earns nothing for
free: the policy is learned on the train split only and must clear the same
research gate (folds, reseed stability, optional held-out and DSR checks).
See :func:`validate_rl` and docs/validation.md for skipped-check behavior.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategies.base import Action, Signal, Strategy

RL_WARMUP = 35  # bars before the state features are trustworthy
_ACTION_DIR = {0: 0, 1: 1, 2: -1}  # action -> desired direction (flat/long/short)
_DIR_OF = {None: 0, "long": 1, "short": -1}
_POS_INDEX = {None: 0, "long": 1, "short": 2}


def rl_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """The three backward-looking signals the state is discretised from."""
    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["trend"] = ind.ema(close, 10) - ind.ema(close, 30)
    out["rsi"] = ind.rsi(close, 14)
    out["zscore"] = ind.zscore(close, 20)
    return out


def discretise(trend: float, rsi: float, zscore: float, pos_dir: Optional[str]) -> tuple:
    """Map raw signals + current position to a small discrete state (81 states
    x 3 actions — learnable from a few hundred bars)."""
    t = 0 if not np.isfinite(trend) or abs(trend) < 1e-12 else (1 if trend > 0 else -1)
    r = 1 if not np.isfinite(rsi) else (0 if rsi < 40 else (2 if rsi > 60 else 1))
    z = 1 if not np.isfinite(zscore) else (0 if zscore < -1 else (2 if zscore > 1 else 1))
    return (t, r, z, _POS_INDEX[pos_dir])


class QLearningAgent:
    """ε-greedy tabular Q-learning over the discrete state with three actions:
    0 = flat, 1 = long, 2 = short."""

    def __init__(self, lr=0.15, gamma=0.95, epsilon=0.1, epochs=20, cost=0.0005, seed=0):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.cost = cost
        self.rng = np.random.default_rng(seed)
        self.Q: dict[tuple, np.ndarray] = {}

    def _q(self, state: tuple) -> np.ndarray:
        return self.Q.setdefault(state, np.zeros(3))

    def train(self, df: pd.DataFrame) -> bool:
        feats = rl_state_features(df)
        close = df["close"].astype(float)
        ret_next = close.pct_change().shift(-1)  # return realised over the bar after t
        data = feats.copy()
        data["ret_next"] = ret_next
        data = data.dropna()
        if len(data) < 60:
            return False
        trend = data["trend"].to_numpy(float)
        rsi = data["rsi"].to_numpy(float)
        z = data["zscore"].to_numpy(float)
        rets = data["ret_next"].to_numpy(float)
        n = len(data)

        for _ in range(self.epochs):
            pos: Optional[str] = None
            for t in range(n - 1):
                s = discretise(trend[t], rsi[t], z[t], pos)
                q = self._q(s)
                a = int(self.rng.integers(0, 3)) if self.rng.random() < self.epsilon else int(np.argmax(q))
                desired = _ACTION_DIR[a]
                reward = desired * rets[t] - self.cost * abs(desired - _DIR_OF[pos])
                pos = {0: None, 1: "long", -1: "short"}[desired]
                s2 = discretise(trend[t + 1], rsi[t + 1], z[t + 1], pos)
                q[a] += self.lr * (reward + self.gamma * float(self._q(s2).max()) - q[a])
        return True

    def act(self, state: tuple) -> int:
        return int(np.argmax(self._q(state)))


class RLStrategy(Strategy):
    """Turns a trained agent's greedy action into a direction signal. Risk stays
    with the broker — the agent only chooses flat/long/short."""

    name = "rl_qlearn"

    def __init__(self, agent: QLearningAgent, risk: Optional[dict] = None):
        super().__init__(risk or {"atr_period": 14, "atr_stop_mult": 2.0})
        self.agent = agent
        self._min_bars = max(RL_WARMUP, self.atr_period) + 2
        self._feat_cache = None

    def min_bars(self) -> int:
        return self._min_bars

    def prepare(self, df: pd.DataFrame) -> "RLStrategy":
        self._feat_cache = rl_state_features(df)
        return self

    def _row(self, window: pd.DataFrame):
        if self._feat_cache is not None:
            try:
                return self._feat_cache.loc[window.index[-1]]
            except KeyError:
                pass
        return rl_state_features(window).iloc[-1]

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        cur_atr = self.current_atr(df)
        if len(df) < self._min_bars or math.isnan(cur_atr):
            return Signal(Action.HOLD, "insufficient data", cur_atr)
        row = self._row(df)
        state = discretise(float(row["trend"]), float(row["rsi"]), float(row["zscore"]), position_direction)
        desired = _ACTION_DIR[self.agent.act(state)]

        if position_direction == "long":
            return Signal(Action.HOLD, "agent stays long", cur_atr) if desired == 1 \
                else Signal(Action.CLOSE, "agent left long", cur_atr)
        if position_direction == "short":
            return Signal(Action.HOLD, "agent stays short", cur_atr) if desired == -1 \
                else Signal(Action.CLOSE, "agent left short", cur_atr)
        if desired == 1:
            return Signal(Action.OPEN_LONG, "agent long", cur_atr)
        if desired == -1:
            return Signal(Action.OPEN_SHORT, "agent short", cur_atr)
        return Signal(Action.HOLD, "agent flat", cur_atr)


# ── walk-forward + hardened gate ─────────────────────────────────────────────


def rl_walk_forward(df, inst, oos_frac=0.25, risk_per_trade=0.01, agent_kwargs=None, seed=0, friction_bps=0.0):
    from .backtest import run_backtest
    from .optimizer import BARS_PER_YEAR, Trial, sharpe_ratio

    split = int(len(df) * (1 - oos_frac))
    train_df, oos_df = df.iloc[:split], df.iloc[split:]
    agent = QLearningAgent(seed=seed, **(agent_kwargs or {}))
    if not agent.train(train_df):
        return None
    strat = RLStrategy(agent)
    if len(oos_df) < strat.min_bars() * 2 or len(train_df) < strat.min_bars() * 2:
        return None
    bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
    is_res = run_backtest(train_df, RLStrategy(agent).prepare(train_df), risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
    oos_res = run_backtest(oos_df, RLStrategy(agent).prepare(oos_df), risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
    return Trial(
        instrument=inst.key, strategy="rl_qlearn", timeframe=inst.timeframe,
        params={"seed": seed, **(agent_kwargs or {})},
        is_sharpe=sharpe_ratio(is_res.equity_curve, bpy),
        oos_sharpe=sharpe_ratio(oos_res.equity_curve, bpy),
        is_trades=is_res.trades, oos_trades=oos_res.trades,
        is_bars=len(train_df), oos_bars=len(oos_df),
        is_return_pct=is_res.return_pct, oos_return_pct=oos_res.return_pct,
        oos_equity_curve=list(oos_res.equity_curve),
    )


def _train_and_backtest(df_train, df_eval, inst, risk, seed, agent_kwargs, friction_bps=0.0):
    from .backtest import run_backtest
    from .optimizer import BARS_PER_YEAR, sharpe_ratio

    agent = QLearningAgent(seed=seed, **(agent_kwargs or {}))
    if df_train is None or df_eval is None or not agent.train(df_train):
        return None
    strat = RLStrategy(agent)
    if len(df_eval) < strat.min_bars() * 2:
        return None
    strat.prepare(df_eval)
    res = run_backtest(df_eval, strat, risk_per_trade=risk, instrument=inst.key, friction_bps=friction_bps)
    return sharpe_ratio(res.equity_curve, BARS_PER_YEAR.get(inst.timeframe, 252)), res.trades


def validate_rl(search_df, holdout_df, inst, current_genome, rng,
                n_trials=0, sr_std_perbar=0.0, risk_per_trade=0.01, agent_kwargs=None, friction_bps=0.0):
    """Hardened gate for an RL policy vs the instrument's current strategy:
    folds (policy relearned per fold), reseed-stability perturbation, embargoed
    held-out confirmation, and the deflated-Sharpe correction."""
    from .backtest import run_backtest
    from .optimizer import (
        BARS_PER_YEAR, DSR_THRESHOLD, MIN_HOLDOUT_TRADES, deflated_sharpe_ratio,
        equity_returns, perturbation_passes, run_walk_forward, sharpe_ratio,
        validation_folds,
    )
    from .search import genome_instrument

    curr_inst = genome_instrument(inst.symbol, inst.timeframe, current_genome, key=inst.key)
    bpy = BARS_PER_YEAR.get(inst.timeframe, 252)

    folds_won = folds_valid = 0
    for fold in validation_folds(search_df):
        s = int(len(fold) * 0.75)
        rl = _train_and_backtest(fold.iloc[:s], fold.iloc[s:], inst, risk_per_trade, 0, agent_kwargs, friction_bps)
        cur = run_walk_forward(fold, curr_inst, {}, risk_per_trade, friction_bps)
        if rl is None or cur is None:
            continue
        folds_valid += 1
        folds_won += int(rl[0] > cur.oos_sharpe)

    main = rl_walk_forward(search_df, inst, risk_per_trade=risk_per_trade, agent_kwargs=agent_kwargs, seed=0, friction_bps=friction_bps)
    candidate_oos = main.oos_sharpe if main else float("-inf")
    base_full = run_walk_forward(search_df, curr_inst, {}, risk_per_trade, friction_bps)
    baseline_oos = base_full.oos_sharpe if base_full else float("-inf")

    reseeded = []
    for s in (1, 2, 3):
        r = rl_walk_forward(search_df, inst, risk_per_trade=risk_per_trade, agent_kwargs=agent_kwargs, seed=s, friction_bps=friction_bps)
        if r is not None:
            reseeded.append(r.oos_sharpe)
    median_reseed = float(np.median(reseeded)) if reseeded else float("-inf")

    split = int(len(search_df) * 0.75)
    cand_h = _train_and_backtest(search_df.iloc[:split], holdout_df, inst, risk_per_trade, 0, agent_kwargs, friction_bps) if holdout_df is not None else None
    curr_h = None
    if holdout_df is not None:
        from .strategies import build_strategy
        try:
            strat = build_strategy(curr_inst.strategy, {}, genome=curr_inst.genome)
            if len(holdout_df) >= strat.min_bars() * 2:
                res = run_backtest(holdout_df, strat, risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
                curr_h = (sharpe_ratio(res.equity_curve, bpy), res.trades)
        except Exception:
            curr_h = None

    # Preserve the legacy qualification rule, but distinguish a skipped check
    # from a computed probability. The 1.0 sentinel alone is not evidence.
    dsr = 1.0
    dsr_checked = main is not None and n_trials >= 10
    if dsr_checked:
        dsr = deflated_sharpe_ratio(equity_returns(main.oos_equity_curve), n_trials, sr_std_perbar)

    h_cand = cand_h[0] if cand_h else float("-inf")
    h_curr = curr_h[0] if curr_h else float("-inf")
    folds_ok = folds_valid >= 3 and folds_won >= math.ceil(0.75 * folds_valid)
    perturb_ok = perturbation_passes(median_reseed, candidate_oos, baseline_oos)
    holdout_ok = holdout_df is None or (
        cand_h is not None and curr_h is not None and cand_h[1] >= MIN_HOLDOUT_TRADES and h_cand > h_curr
    )
    dsr_ok = dsr >= DSR_THRESHOLD

    dsr_label = f"{dsr:.3f}" if dsr_checked else "not checked"
    holdout_label = f"held-out {h_cand:.3f} > {h_curr:.3f}" if holdout_df is not None else "held-out not checked"

    if not folds_ok:
        reason = f"regime test failed: beat current in {folds_won}/{folds_valid} folds"
    elif not perturb_ok:
        reason = f"reseed-stability failed: median {median_reseed:.3f} vs own {candidate_oos:.3f}/base {baseline_oos:.3f}"
    elif not dsr_ok:
        reason = f"deflated-Sharpe {dsr:.3f} < {DSR_THRESHOLD} after {n_trials} trials"
    elif not holdout_ok:
        reason = f"held-out failed: agent {h_cand:.3f} vs current {h_curr:.3f}"
    else:
        reason = f"won {folds_won}/{folds_valid} folds; reseed-stable; {holdout_label}; DSR {dsr_label}"

    limitations = []
    if not dsr_checked:
        limitations.append("DSR not checked (requires a valid trial and at least 10 effective trials)")
    if holdout_df is None:
        limitations.append("held-out confirmation not checked (no holdout supplied)")
    if limitations:
        reason += "; " + "; ".join(limitations)

    return {
        "deflated_sharpe_checked": dsr_checked,
        "holdout_checked": holdout_df is not None,
        "limitations": limitations,
        "qualified": folds_ok and perturb_ok and holdout_ok and dsr_ok,
        "folds_won": folds_won, "folds_valid": folds_valid,
        "candidate_oos_sharpe": candidate_oos, "baseline_oos_sharpe": baseline_oos,
        "median_reseed_sharpe": median_reseed, "deflated_sharpe": dsr,
        "holdout_candidate_sharpe": h_cand, "holdout_current_sharpe": h_curr,
        "reason": reason,
    }
