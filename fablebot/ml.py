"""ML signal models — a strategy *type* that learns its entries.

A gradient-boosted regression-tree ensemble (pure numpy, no heavy deps) is
trained to predict the next ``horizon`` bars' ATR-normalised return from
engineered features. ``MLStrategy`` thresholds that prediction into long/short
entries; sizing, the hard stop and trailing stay with the risk engine, exactly
as for every other strategy.

ML is the easiest thing in the engine to overfit, so it earns nothing for free:
the model is fit on the *train* split only and judged out-of-sample, then must
clear the Workstream-A gate (folds + perturbation + embargoed held-out +
deflated Sharpe) on data it never saw — see ``fablebot.ml.validate_ml``.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategies.base import Action, Signal, Strategy

# ── feature engineering ──────────────────────────────────────────────────────

FEATURE_WARMUP = 60  # bars before the engineered features are trustworthy


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """A stable, leakage-free feature matrix aligned to ``df.index``. Every
    column at bar i uses only information available at or before i."""
    close = df["close"].astype(float)
    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = close.pct_change(1)
    feats["ret_5"] = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["rsi"] = ind.rsi(close, 14) / 100.0
    line, signal, hist = ind.macd(close)
    feats["macd_hist"] = hist / close
    feats["zscore"] = ind.zscore(close, 20)
    adx_, plus_di, minus_di = ind.adx(df, 14)
    feats["adx"] = adx_ / 100.0
    feats["di_diff"] = (plus_di - minus_di) / 100.0
    atr_ = ind.atr(df, 14)
    feats["atr_pct"] = atr_ / close
    feats["ema_spread"] = (ind.ema(close, 10) - ind.ema(close, 30)) / close
    bbu, bbm, bbl = ind.bollinger(close, 20, 2.0)
    width = (bbu - bbl).replace(0.0, np.nan)
    feats["bb_pos"] = (close - bbm) / width
    if "volume" in df.columns:
        feats["volume_z"] = ind.volume_zscore(df, 20)
    return feats


FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_10", "rsi", "macd_hist", "zscore",
    "adx", "di_diff", "atr_pct", "ema_spread", "bb_pos", "volume_z",
]


def make_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Target: forward return over ``horizon`` bars, normalised by ATR% so the
    label is comparable across instruments and volatility regimes."""
    close = df["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    atr_pct = (ind.atr(df, 14) / close).replace(0.0, np.nan)
    return fwd / atr_pct


# ── gradient-boosted regression trees (numpy) ────────────────────────────────


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value=0.0):
        self.feature = None
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = value


def _build_tree(X, y, depth, max_depth, min_leaf, n_thresholds):
    node = _Node(value=float(y.mean()))
    if depth >= max_depth or len(y) < 2 * min_leaf:
        return node
    best = None  # (sse, feature, threshold)
    for f in range(X.shape[1]):
        col = X[:, f]
        lo, hi = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(lo) or lo == hi:
            continue
        for thr in np.unique(np.quantile(col, np.linspace(0.1, 0.9, n_thresholds))):
            mask = col <= thr
            nl, nr = int(mask.sum()), int((~mask).sum())
            if nl < min_leaf or nr < min_leaf:
                continue
            yl, yr = y[mask], y[~mask]
            sse = float(((yl - yl.mean()) ** 2).sum() + ((yr - yr.mean()) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, f, float(thr))
    if best is None:
        return node
    _, feature, threshold = best
    mask = X[:, feature] <= threshold
    node.feature, node.threshold = feature, threshold
    node.left = _build_tree(X[mask], y[mask], depth + 1, max_depth, min_leaf, n_thresholds)
    node.right = _build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_leaf, n_thresholds)
    return node


def _predict_tree(node: _Node, x: np.ndarray) -> float:
    while node.left is not None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.value


class GradientBoostedTrees:
    """Squared-error gradient boosting over shallow regression trees."""

    def __init__(self, n_estimators=60, learning_rate=0.1, max_depth=3,
                 min_leaf=10, n_thresholds=12, subsample=0.8, seed=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_thresholds = n_thresholds
        self.subsample = subsample
        self.rng = np.random.default_rng(seed)
        self.base = 0.0
        self.trees: list[_Node] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTrees":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.base = float(y.mean())
        pred = np.full(len(y), self.base)
        n_sub = max(self.min_leaf * 2, int(self.subsample * len(y)))
        for _ in range(self.n_estimators):
            residual = y - pred
            idx = self.rng.choice(len(y), size=min(n_sub, len(y)), replace=False)
            tree = _build_tree(X[idx], residual[idx], 0, self.max_depth, self.min_leaf, self.n_thresholds)
            update = np.array([_predict_tree(tree, row) for row in X])
            pred += self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict_one(self, x: np.ndarray) -> float:
        out = self.base
        for tree in self.trees:
            out += self.learning_rate * _predict_tree(tree, x)
        return float(out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(row) for row in np.asarray(X, float)])


# ── trained model bundle + strategy ──────────────────────────────────────────


class MLModel:
    """Feature scaler + GBT + the entry threshold derived from training-set
    prediction dispersion. Frozen after :meth:`train` — inference never refits."""

    def __init__(self, horizon=5, entry_q=0.6, seed=0, **gbt_kwargs):
        self.horizon = horizon
        self.entry_q = entry_q
        self.seed = seed
        self.gbt_kwargs = gbt_kwargs
        self.columns: list[str] = []
        self.mean = None
        self.std = None
        self.gbt: Optional[GradientBoostedTrees] = None
        self.entry_threshold = 0.0

    def _matrix(self, df: pd.DataFrame):
        feats = make_features(df)
        cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
        return feats[cols], cols

    def train(self, df: pd.DataFrame) -> bool:
        feats, cols = self._matrix(df)
        # Drop degenerate columns (all-NaN or zero-variance, e.g. constant
        # volume) so a single dead feature can't empty the training set.
        cols = [c for c in cols if feats[c].notna().any() and feats[c].std(skipna=True) > 0]
        if not cols:
            return False
        labels = make_labels(df, self.horizon)
        data = feats[cols].copy()
        data["_y"] = labels
        data = data.dropna()
        if len(data) < 80:
            return False
        X = data[cols].to_numpy(float)
        y = data["_y"].to_numpy(float)
        self.columns = cols
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Xs = (X - self.mean) / self.std
        self.gbt = GradientBoostedTrees(seed=self.seed, **self.gbt_kwargs).fit(Xs, y)
        preds = self.gbt.predict(Xs)
        # trade only on convictions beyond the entry_q quantile of |prediction|
        self.entry_threshold = float(np.quantile(np.abs(preds), self.entry_q))
        return True

    def predict_row(self, row: pd.Series) -> float:
        """Score one already-computed feature row (NaN-safe)."""
        if row is None or row.isna().any():
            return float("nan")
        x = (row.to_numpy(float) - self.mean) / self.std
        return self.gbt.predict_one(x)

    def score_last(self, window: pd.DataFrame) -> float:
        return self.predict_row(make_features(window)[self.columns].iloc[-1])


class MLStrategy(Strategy):
    """Turns a trained :class:`MLModel` into entry/exit signals."""

    name = "ml_gbt"

    def __init__(self, model: MLModel, risk: Optional[dict] = None):
        super().__init__(risk or {"atr_period": 14, "atr_stop_mult": 2.0})
        self.model = model
        self._min_bars = max(FEATURE_WARMUP, self.atr_period) + model.horizon + 2
        self._feat_cache = None

    def min_bars(self) -> int:
        return self._min_bars

    def prepare(self, df: pd.DataFrame) -> "MLStrategy":
        """Precompute features for every bar of ``df`` once. The features are
        backward-looking, so the cached value at timestamp t equals what an
        expanding-window recompute would yield — exact, but O(n) instead of
        O(n^2) across a backtest."""
        self._feat_cache = make_features(df)[self.model.columns]
        return self

    def _feature_row(self, window: pd.DataFrame):
        cache = self._feat_cache
        if cache is not None:
            try:
                return cache.loc[window.index[-1]]
            except KeyError:
                pass
        return make_features(window)[self.model.columns].iloc[-1]

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        cur_atr = self.current_atr(df)
        if len(df) < self._min_bars or math.isnan(cur_atr):
            return Signal(Action.HOLD, "insufficient data", cur_atr)
        score = self.model.predict_row(self._feature_row(df))
        if math.isnan(score):
            return Signal(Action.HOLD, "no prediction", cur_atr)
        thr = self.model.entry_threshold

        if position_direction == "long":
            if score < 0:
                return Signal(Action.CLOSE, f"model turned bearish ({score:+.3f})", cur_atr)
            return Signal(Action.HOLD, "model still bullish", cur_atr)
        if position_direction == "short":
            if score > 0:
                return Signal(Action.CLOSE, f"model turned bullish ({score:+.3f})", cur_atr)
            return Signal(Action.HOLD, "model still bearish", cur_atr)

        if score >= thr:
            return Signal(Action.OPEN_LONG, f"model long ({score:+.3f} >= {thr:.3f})", cur_atr)
        if score <= -thr:
            return Signal(Action.OPEN_SHORT, f"model short ({score:+.3f} <= -{thr:.3f})", cur_atr)
        return Signal(Action.HOLD, f"low conviction ({score:+.3f})", cur_atr)


# ── walk-forward + hardened-gate validation for ML ───────────────────────────


def ml_walk_forward(df, inst, oos_frac=0.25, risk_per_trade=0.01, model_kwargs=None, seed=0, friction_bps=0.0):
    """Fit an MLModel on the train split, then backtest it in/out-of-sample.
    Returns an optimizer.Trial (so it flows through the existing pipeline) or
    None when there isn't enough data to train."""
    from .backtest import run_backtest
    from .optimizer import BARS_PER_YEAR, Trial, sharpe_ratio

    split = int(len(df) * (1 - oos_frac))
    train_df, oos_df = df.iloc[:split], df.iloc[split:]
    model = MLModel(seed=seed, **(model_kwargs or {}))
    if not model.train(train_df):
        return None
    strat = MLStrategy(model, risk=getattr(inst, "params", None) or None)
    if len(oos_df) < strat.min_bars() * 2 or len(train_df) < strat.min_bars() * 2:
        return None
    bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
    is_res = run_backtest(train_df, MLStrategy(model).prepare(train_df), risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
    oos_res = run_backtest(oos_df, MLStrategy(model).prepare(oos_df), risk_per_trade=risk_per_trade, instrument=inst.key, friction_bps=friction_bps)
    return Trial(
        instrument=inst.key, strategy="ml_gbt", timeframe=inst.timeframe,
        params={"horizon": model.horizon, "seed": seed, **(model_kwargs or {})},
        is_sharpe=sharpe_ratio(is_res.equity_curve, bpy),
        oos_sharpe=sharpe_ratio(oos_res.equity_curve, bpy),
        is_trades=is_res.trades, oos_trades=oos_res.trades,
        is_bars=len(train_df), oos_bars=len(oos_df),
        is_return_pct=is_res.return_pct, oos_return_pct=oos_res.return_pct,
        oos_equity_curve=list(oos_res.equity_curve),
    )


def _fit_and_backtest(df_train, df_eval, inst, risk, seed, model_kwargs, friction_bps=0.0):
    """Train on df_train, score sharpe on the whole df_eval slice (no re-split)."""
    from .backtest import run_backtest
    from .optimizer import BARS_PER_YEAR, sharpe_ratio

    model = MLModel(seed=seed, **(model_kwargs or {}))
    if df_train is None or df_eval is None or not model.train(df_train):
        return None
    strat = MLStrategy(model)
    if len(df_eval) < strat.min_bars() * 2:
        return None
    strat.prepare(df_eval)
    res = run_backtest(df_eval, strat, risk_per_trade=risk, instrument=inst.key, friction_bps=friction_bps)
    return sharpe_ratio(res.equity_curve, BARS_PER_YEAR.get(inst.timeframe, 252)), res.trades


def validate_ml(search_df, holdout_df, inst, current_genome, rng,
                n_trials=0, sr_std_perbar=0.0, risk_per_trade=0.01, model_kwargs=None, friction_bps=0.0):
    """Hardened gate for an ML model vs the instrument's current strategy.

    Folds: model retrained per non-overlapping fold (train half) and scored on
    the fold's eval half, vs the current strategy. Perturbation: retrain with
    different seeds — a real edge survives reseeding. Held-out: train on the
    search slice, confirm on the embargoed holdout. Plus the deflated-Sharpe
    correction. ML clears the same bar as everything else, on data it never saw.
    """
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
        ml = _fit_and_backtest(fold.iloc[:s], fold.iloc[s:], inst, risk_per_trade, 0, model_kwargs, friction_bps)
        cur = run_walk_forward(fold, curr_inst, {}, risk_per_trade, friction_bps)
        if ml is None or cur is None:
            continue
        folds_valid += 1
        folds_won += int(ml[0] > cur.oos_sharpe)

    main = ml_walk_forward(search_df, inst, risk_per_trade=risk_per_trade, model_kwargs=model_kwargs, seed=0, friction_bps=friction_bps)
    candidate_oos = main.oos_sharpe if main else float("-inf")
    base_full = run_walk_forward(search_df, curr_inst, {}, risk_per_trade, friction_bps)
    baseline_oos = base_full.oos_sharpe if base_full else float("-inf")

    reseeded = []
    for s in (1, 2, 3):
        r = ml_walk_forward(search_df, inst, risk_per_trade=risk_per_trade, model_kwargs=model_kwargs, seed=s, friction_bps=friction_bps)
        if r is not None:
            reseeded.append(r.oos_sharpe)
    median_reseed = float(np.median(reseeded)) if reseeded else float("-inf")

    # held-out: fit on the whole search slice, evaluate on the embargoed tail
    split = int(len(search_df) * 0.75)
    cand_h = _fit_and_backtest(search_df.iloc[:split], holdout_df, inst, risk_per_trade, 0, model_kwargs, friction_bps) if holdout_df is not None else None
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

    dsr = 1.0
    if main is not None and n_trials >= 10:
        dsr = deflated_sharpe_ratio(equity_returns(main.oos_equity_curve), n_trials, sr_std_perbar)

    h_cand = cand_h[0] if cand_h else float("-inf")
    h_curr = curr_h[0] if curr_h else float("-inf")
    folds_ok = folds_valid >= 3 and folds_won >= math.ceil(0.75 * folds_valid)
    perturb_ok = perturbation_passes(median_reseed, candidate_oos, baseline_oos)
    holdout_ok = holdout_df is None or (
        cand_h is not None and curr_h is not None and cand_h[1] >= MIN_HOLDOUT_TRADES and h_cand > h_curr
    )
    dsr_ok = dsr >= DSR_THRESHOLD

    if not folds_ok:
        reason = f"regime test failed: beat current in {folds_won}/{folds_valid} folds"
    elif not perturb_ok:
        reason = f"reseed-stability failed: median {median_reseed:.3f} vs own {candidate_oos:.3f}/base {baseline_oos:.3f}"
    elif not dsr_ok:
        reason = f"deflated-Sharpe {dsr:.3f} < {DSR_THRESHOLD} after {n_trials} trials"
    elif not holdout_ok:
        reason = f"held-out failed: model {h_cand:.3f} vs current {h_curr:.3f}"
    else:
        reason = f"won {folds_won}/{folds_valid} folds; reseed-stable; held-out {h_cand:.3f} > {h_curr:.3f}; DSR {dsr:.3f}"

    return {
        "qualified": folds_ok and perturb_ok and holdout_ok and dsr_ok,
        "folds_won": folds_won, "folds_valid": folds_valid,
        "candidate_oos_sharpe": candidate_oos, "baseline_oos_sharpe": baseline_oos,
        "median_reseed_sharpe": median_reseed, "deflated_sharpe": dsr,
        "holdout_candidate_sharpe": h_cand, "holdout_current_sharpe": h_curr,
        "reason": reason,
    }
