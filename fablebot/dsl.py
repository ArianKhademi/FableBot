"""Composable strategy DSL — a search space of strategies, not three fixed ones.

A *genome* is a serialisable spec (a plain dict, so it lives happily in
``backtest_history`` / ``proposals``) describing:

  - ``features``: named indicator series drawn from the palette below
  - ``entry_long`` / ``entry_short`` / ``exit_long`` / ``exit_short``: boolean
    condition trees composed from those features, constants and price
  - ``risk``: ATR period + stop/trail multiples handed to the *unchanged* risk
    engine

``GenomeStrategy`` instantiates a genome into an ordinary
:class:`~fablebot.strategies.base.Strategy`. It only decides *what signals fire*;
sizing, the hard 1% stop, gap protection and trailing stay entirely in the risk
engine and broker, exactly as for the three hand-written strategies. Any genome
can run on any symbol — the five fixed assignments in ``config.yaml`` are just
the seed population the optimizer starts from.

The genome operators (:func:`sample_genome`, :func:`mutate_genome`,
:func:`crossover_genome`) define the space; the guided/evolutionary search
(``fablebot.search``) explores it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategies.base import Action, Signal, Strategy

# ── feature palette ──────────────────────────────────────────────────────────
# Each entry: kind -> (builder(df, params) -> Series, lookback(params) -> int).
# ``lookback`` is the warm-up the feature needs; it drives min_bars().


def _macd_component(idx: int):
    return lambda df, p: ind.macd(
        df["close"], int(p.get("fast", 12)), int(p.get("slow", 26)), int(p.get("signal", 9))
    )[idx]


def _adx_component(idx: int):
    return lambda df, p: ind.adx(df, int(p.get("period", 14)))[idx]


def _bb_component(idx: int):
    return lambda df, p: ind.bollinger(df["close"], int(p.get("period", 20)), float(p.get("num_std", 2.0)))[idx]


def _kc_component(idx: int):
    return lambda df, p: ind.keltner(df, int(p.get("period", 20)), float(p.get("atr_mult", 2.0)))[idx]


FEATURES: dict[str, tuple[Callable, Callable]] = {
    "close": (lambda df, p: df["close"].astype(float), lambda p: 1),
    "high": (lambda df, p: df["high"].astype(float), lambda p: 1),
    "low": (lambda df, p: df["low"].astype(float), lambda p: 1),
    "open": (lambda df, p: df["open"].astype(float), lambda p: 1),
    "ema": (lambda df, p: ind.ema(df["close"], int(p["period"])), lambda p: int(p["period"])),
    "sma": (lambda df, p: ind.sma(df["close"], int(p["period"])), lambda p: int(p["period"])),
    "rsi": (lambda df, p: ind.rsi(df["close"], int(p.get("period", 14))), lambda p: int(p.get("period", 14)) + 1),
    "zscore": (lambda df, p: ind.zscore(df["close"], int(p["lookback"])), lambda p: int(p["lookback"])),
    "atr": (lambda df, p: ind.atr(df, int(p.get("period", 14))), lambda p: int(p.get("period", 14)) + 1),
    "macd_line": (_macd_component(0), lambda p: int(p.get("slow", 26)) + int(p.get("signal", 9))),
    "macd_signal": (_macd_component(1), lambda p: int(p.get("slow", 26)) + int(p.get("signal", 9))),
    "macd_hist": (_macd_component(2), lambda p: int(p.get("slow", 26)) + int(p.get("signal", 9))),
    "adx": (_adx_component(0), lambda p: int(p.get("period", 14)) * 2),
    "plus_di": (_adx_component(1), lambda p: int(p.get("period", 14)) * 2),
    "minus_di": (_adx_component(2), lambda p: int(p.get("period", 14)) * 2),
    "bb_upper": (_bb_component(0), lambda p: int(p.get("period", 20))),
    "bb_mid": (_bb_component(1), lambda p: int(p.get("period", 20))),
    "bb_lower": (_bb_component(2), lambda p: int(p.get("period", 20))),
    "kc_upper": (_kc_component(0), lambda p: int(p.get("period", 20)) + 1),
    "kc_mid": (_kc_component(1), lambda p: int(p.get("period", 20))),
    "kc_lower": (_kc_component(2), lambda p: int(p.get("period", 20)) + 1),
    "donchian_hi": (lambda df, p: ind.donchian(df, int(p["period"]))[0], lambda p: int(p["period"]) + 1),
    "donchian_lo": (lambda df, p: ind.donchian(df, int(p["period"]))[1], lambda p: int(p["period"]) + 1),
    "volume_z": (lambda df, p: ind.volume_zscore(df, int(p.get("lookback", 20))), lambda p: int(p.get("lookback", 20))),
}

CMP_OPS = {
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
}


# ── feature computation ──────────────────────────────────────────────────────


def build_feature(df: pd.DataFrame, spec: dict) -> pd.Series:
    kind = spec["kind"]
    if kind not in FEATURES:
        raise ValueError(f"unknown feature kind {kind!r}; choose from {sorted(FEATURES)}")
    builder, _ = FEATURES[kind]
    return builder(df, spec.get("params", {}))


def feature_lookback(spec: dict) -> int:
    _, lb = FEATURES[spec["kind"]]
    return int(lb(spec.get("params", {})))


def compute_features(df: pd.DataFrame, specs: list[dict]) -> dict[str, pd.Series]:
    return {spec["id"]: build_feature(df, spec) for spec in specs}


# ── condition evaluation ─────────────────────────────────────────────────────


def _operand_value(operand: dict, features: dict[str, pd.Series], back: int) -> float:
    """Value of an operand ``back`` bars from the end (0 = current bar)."""
    if "const" in operand:
        return float(operand["const"])
    series = features.get(operand["feature"])
    if series is None:
        return float("nan")
    pos = -1 - back - int(operand.get("lag", 0))
    if len(series) < -pos:
        return float("nan")
    return float(series.iloc[pos])


def eval_condition(node: Optional[dict], features: dict[str, pd.Series]) -> bool:
    """Evaluate a condition tree on the latest bar. A NaN operand (warm-up,
    missing feature) makes its comparison False, never an exception."""
    if not node:
        return False
    t = node["type"]
    if t == "and":
        return all(eval_condition(n, features) for n in node["nodes"])
    if t == "or":
        return any(eval_condition(n, features) for n in node["nodes"])
    if t == "not":
        return not eval_condition(node["node"], features)
    if t == "cmp":
        a = _operand_value(node["left"], features, 0)
        b = _operand_value(node["right"], features, 0)
        if math.isnan(a) or math.isnan(b):
            return False
        return bool(CMP_OPS[node["op"]](a, b))
    if t == "cross":
        a_now = _operand_value(node["left"], features, 0)
        b_now = _operand_value(node["right"], features, 0)
        a_prev = _operand_value(node["left"], features, 1)
        b_prev = _operand_value(node["right"], features, 1)
        if any(math.isnan(x) for x in (a_now, b_now, a_prev, b_prev)):
            return False
        if node["dir"] == "above":
            return a_prev <= b_prev and a_now > b_now
        return a_prev >= b_prev and a_now < b_now
    raise ValueError(f"unknown condition type {t!r}")


# ── genome <-> strategy ──────────────────────────────────────────────────────


def validate_genome(genome: dict) -> None:
    if not genome.get("features"):
        raise ValueError("genome has no features")
    ids = [f["id"] for f in genome["features"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate feature ids in genome")
    for spec in genome["features"]:
        if spec["kind"] not in FEATURES:
            raise ValueError(f"unknown feature kind {spec['kind']!r}")
    if not (genome.get("entry_long") or genome.get("entry_short")):
        raise ValueError("genome has no entry condition")


def genome_id(genome: dict) -> str:
    blob = json.dumps(genome, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def genome_min_bars(genome: dict) -> int:
    lookbacks = [feature_lookback(s) for s in genome["features"]]
    risk_period = int(genome.get("risk", {}).get("atr_period", 14))
    return max(lookbacks + [risk_period]) + 2


class GenomeStrategy(Strategy):
    """A strategy assembled from a genome. Emits signals; the risk engine still
    owns sizing and stops."""

    def __init__(self, genome: dict):
        validate_genome(genome)
        self.genome = copy.deepcopy(genome)
        super().__init__(self.genome.get("risk", {}))
        self.name = self.genome.get("name", "genome")
        self._specs = self.genome["features"]
        self._min_bars = genome_min_bars(self.genome)

    def min_bars(self) -> int:
        return self._min_bars

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        cur_atr = self.current_atr(df)
        if len(df) < self._min_bars or math.isnan(cur_atr):
            return Signal(Action.HOLD, "insufficient data", cur_atr)
        feats = compute_features(df, self._specs)

        if position_direction == "long":
            if eval_condition(self.genome.get("exit_long"), feats):
                return Signal(Action.CLOSE, "exit_long", cur_atr)
            return Signal(Action.HOLD, "holding long", cur_atr)
        if position_direction == "short":
            if eval_condition(self.genome.get("exit_short"), feats):
                return Signal(Action.CLOSE, "exit_short", cur_atr)
            return Signal(Action.HOLD, "holding short", cur_atr)

        if eval_condition(self.genome.get("entry_long"), feats):
            return Signal(Action.OPEN_LONG, "entry_long", cur_atr)
        if eval_condition(self.genome.get("entry_short"), feats):
            return Signal(Action.OPEN_SHORT, "entry_short", cur_atr)
        return Signal(Action.HOLD, "no signal", cur_atr)

    def watching(self, df: pd.DataFrame) -> str:
        return f"genome {self.genome.get('name', '?')} with {len(self._specs)} features"


# ── parameter overrides (used by the guided search to tune a genome) ─────────


def genome_param_paths(genome: dict) -> dict[str, float]:
    """Flat, dotted view of every tunable scalar in a genome:
    ``feat.<id>.<param>`` and ``risk.<param>``. Lets the search nudge a genome's
    numbers without knowing its structure."""
    out: dict[str, float] = {}
    for spec in genome["features"]:
        for k, v in spec.get("params", {}).items():
            out[f"feat.{spec['id']}.{k}"] = v
    for k, v in genome.get("risk", {}).items():
        out[f"risk.{k}"] = v
    return out


def apply_genome_params(genome: dict, overrides: dict[str, float]) -> dict:
    """Return a copy of ``genome`` with dotted-path scalars replaced."""
    g = copy.deepcopy(genome)
    by_id = {s["id"]: s for s in g["features"]}
    for path, val in overrides.items():
        parts = path.split(".")
        if parts[0] == "feat":
            _, fid, key = parts
            by_id[fid].setdefault("params", {})[key] = val
        elif parts[0] == "risk":
            g.setdefault("risk", {})[parts[1]] = val
    return g


# ── seed genomes: the three hand-written strategies, as genomes ──────────────


def seed_mean_reversion(lookback=20, entry_z=2.0, exit_z=0.5, atr_period=14, atr_stop_mult=2.0) -> dict:
    return {
        "name": "mean_reversion",
        "features": [{"id": "z", "kind": "zscore", "params": {"lookback": lookback}}],
        "entry_long": {"type": "cmp", "op": "lte", "left": {"feature": "z"}, "right": {"const": -entry_z}},
        "entry_short": {"type": "cmp", "op": "gte", "left": {"feature": "z"}, "right": {"const": entry_z}},
        "exit_long": {"type": "cmp", "op": "gte", "left": {"feature": "z"}, "right": {"const": -exit_z}},
        "exit_short": {"type": "cmp", "op": "lte", "left": {"feature": "z"}, "right": {"const": exit_z}},
        "risk": {"atr_period": atr_period, "atr_stop_mult": atr_stop_mult},
    }


def seed_trend_following(fast_ema=20, slow_ema=50, atr_period=14, atr_stop_mult=3.0, trail_atr_mult=3.0) -> dict:
    fast = {"feature": "fast"}
    slow = {"feature": "slow"}
    return {
        "name": "trend_following",
        "features": [
            {"id": "fast", "kind": "ema", "params": {"period": fast_ema}},
            {"id": "slow", "kind": "ema", "params": {"period": slow_ema}},
        ],
        "entry_long": {"type": "cmp", "op": "gt", "left": fast, "right": slow},
        "entry_short": {"type": "cmp", "op": "lt", "left": fast, "right": slow},
        "exit_long": {"type": "cmp", "op": "lt", "left": fast, "right": slow},
        "exit_short": {"type": "cmp", "op": "gt", "left": fast, "right": slow},
        "risk": {"atr_period": atr_period, "atr_stop_mult": atr_stop_mult, "trail_atr_mult": trail_atr_mult},
    }


def seed_momentum_breakout(entry_channel=20, exit_channel=10, atr_period=14, atr_stop_mult=2.5, trail_atr_mult=2.5) -> dict:
    return {
        "name": "momentum_breakout",
        "features": [
            {"id": "close", "kind": "close"},
            {"id": "entry_hi", "kind": "donchian_hi", "params": {"period": entry_channel}},
            {"id": "entry_lo", "kind": "donchian_lo", "params": {"period": entry_channel}},
            {"id": "exit_hi", "kind": "donchian_hi", "params": {"period": exit_channel}},
            {"id": "exit_lo", "kind": "donchian_lo", "params": {"period": exit_channel}},
        ],
        "entry_long": {"type": "cmp", "op": "gt", "left": {"feature": "close"}, "right": {"feature": "entry_hi"}},
        "entry_short": {"type": "cmp", "op": "lt", "left": {"feature": "close"}, "right": {"feature": "entry_lo"}},
        "exit_long": {"type": "cmp", "op": "lt", "left": {"feature": "close"}, "right": {"feature": "exit_lo"}},
        "exit_short": {"type": "cmp", "op": "gt", "left": {"feature": "close"}, "right": {"feature": "exit_hi"}},
        "risk": {"atr_period": atr_period, "atr_stop_mult": atr_stop_mult, "trail_atr_mult": trail_atr_mult},
    }


def seed_genomes() -> list[dict]:
    return [seed_mean_reversion(), seed_trend_following(), seed_momentum_breakout()]


# ── genome operators: sample / mutate / crossover (the search space) ─────────

# Sane clamps for nudging scalars, keyed by the trailing param name. Integer
# params are rounded; everything else is clamped to a reasonable band so a
# mutation can never produce a degenerate strategy.
_PARAM_BOUNDS = {
    "lookback": (8, 80, True), "period": (5, 100, True),
    "fast": (4, 40, True), "slow": (20, 120, True), "signal": (4, 20, True),
    "atr_period": (5, 40, True),
    "atr_stop_mult": (1.0, 5.0, False), "trail_atr_mult": (1.0, 6.0, False),
    "num_std": (1.0, 3.5, False), "atr_mult": (1.0, 4.0, False),
    "entry_z": (1.0, 3.5, False), "exit_z": (0.1, 1.5, False),
}


def _clamp(name: str, val: float) -> float:
    lo, hi, is_int = _PARAM_BOUNDS.get(name, (None, None, False))
    if lo is not None:
        val = min(max(val, lo), hi)
    return int(round(val)) if is_int else round(float(val), 3)


def _cmp(op, left, right):
    return {"type": "cmp", "op": op, "left": left, "right": right}


def _arch_mean_reversion(rng):
    lb = int(rng.integers(10, 50)); ez = round(float(rng.uniform(1.5, 3.0)), 2); xz = round(float(rng.uniform(0.2, 1.0)), 2)
    return seed_mean_reversion(lb, ez, xz, 14, round(float(rng.uniform(1.5, 3.5)), 2))


def _arch_trend(rng):
    fast = int(rng.integers(5, 25)); slow = int(rng.integers(fast + 10, 100))
    trail = round(float(rng.uniform(2.0, 4.0)), 2)
    return seed_trend_following(fast, slow, 14, round(float(rng.uniform(2.0, 4.0)), 2), trail)


def _arch_breakout(rng):
    entry = int(rng.integers(15, 55)); exit_ = int(rng.integers(5, max(6, entry - 2)))
    return seed_momentum_breakout(entry, exit_, 14, round(float(rng.uniform(1.5, 4.0)), 2), round(float(rng.uniform(1.5, 4.0)), 2))


def _arch_rsi(rng):
    period = int(rng.integers(7, 21)); lo = int(rng.integers(20, 35)); hi = int(rng.integers(65, 80))
    r = {"feature": "rsi"}
    return {
        "name": "rsi_reversion",
        "features": [{"id": "rsi", "kind": "rsi", "params": {"period": period}}],
        "entry_long": _cmp("lt", r, {"const": lo}),
        "entry_short": _cmp("gt", r, {"const": hi}),
        "exit_long": _cmp("gte", r, {"const": 50}),
        "exit_short": _cmp("lte", r, {"const": 50}),
        "risk": {"atr_period": 14, "atr_stop_mult": round(float(rng.uniform(1.5, 3.5)), 2)},
    }


def _arch_macd(rng):
    fast = int(rng.integers(6, 16)); slow = int(rng.integers(fast + 8, 40)); signal = int(rng.integers(5, 12))
    line = {"feature": "line"}; sig = {"feature": "sig"}
    params = {"fast": fast, "slow": slow, "signal": signal}
    return {
        "name": "macd_cross",
        "features": [
            {"id": "line", "kind": "macd_line", "params": dict(params)},
            {"id": "sig", "kind": "macd_signal", "params": dict(params)},
        ],
        "entry_long": {"type": "cross", "dir": "above", "left": line, "right": sig},
        "entry_short": {"type": "cross", "dir": "below", "left": line, "right": sig},
        "exit_long": {"type": "cross", "dir": "below", "left": line, "right": sig},
        "exit_short": {"type": "cross", "dir": "above", "left": line, "right": sig},
        "risk": {"atr_period": 14, "atr_stop_mult": round(float(rng.uniform(2.0, 4.0)), 2),
                 "trail_atr_mult": round(float(rng.uniform(2.0, 4.0)), 2)},
    }


def _arch_bollinger(rng):
    period = int(rng.integers(12, 30)); num_std = round(float(rng.uniform(1.5, 3.0)), 2)
    p = {"period": period, "num_std": num_std}
    return {
        "name": "bollinger_revert",
        "features": [
            {"id": "close", "kind": "close"},
            {"id": "bbu", "kind": "bb_upper", "params": dict(p)},
            {"id": "bbm", "kind": "bb_mid", "params": dict(p)},
            {"id": "bbl", "kind": "bb_lower", "params": dict(p)},
        ],
        "entry_long": _cmp("lt", {"feature": "close"}, {"feature": "bbl"}),
        "entry_short": _cmp("gt", {"feature": "close"}, {"feature": "bbu"}),
        "exit_long": _cmp("gte", {"feature": "close"}, {"feature": "bbm"}),
        "exit_short": _cmp("lte", {"feature": "close"}, {"feature": "bbm"}),
        "risk": {"atr_period": 14, "atr_stop_mult": round(float(rng.uniform(1.5, 3.0)), 2)},
    }


def _arch_adx_trend(rng):
    period = int(rng.integers(10, 20)); thresh = int(rng.integers(18, 30))
    ema_p = int(rng.integers(20, 60))
    return {
        "name": "adx_trend",
        "features": [
            {"id": "close", "kind": "close"},
            {"id": "trend", "kind": "ema", "params": {"period": ema_p}},
            {"id": "adx", "kind": "adx", "params": {"period": period}},
            {"id": "pdi", "kind": "plus_di", "params": {"period": period}},
            {"id": "mdi", "kind": "minus_di", "params": {"period": period}},
        ],
        "entry_long": {"type": "and", "nodes": [
            _cmp("gt", {"feature": "adx"}, {"const": thresh}),
            _cmp("gt", {"feature": "pdi"}, {"feature": "mdi"}),
            _cmp("gt", {"feature": "close"}, {"feature": "trend"}),
        ]},
        "entry_short": {"type": "and", "nodes": [
            _cmp("gt", {"feature": "adx"}, {"const": thresh}),
            _cmp("gt", {"feature": "mdi"}, {"feature": "pdi"}),
            _cmp("lt", {"feature": "close"}, {"feature": "trend"}),
        ]},
        "exit_long": _cmp("lt", {"feature": "pdi"}, {"feature": "mdi"}),
        "exit_short": _cmp("lt", {"feature": "mdi"}, {"feature": "pdi"}),
        "risk": {"atr_period": 14, "atr_stop_mult": round(float(rng.uniform(2.0, 4.0)), 2),
                 "trail_atr_mult": round(float(rng.uniform(2.0, 4.0)), 2)},
    }


ARCHETYPES = [
    _arch_mean_reversion, _arch_trend, _arch_breakout,
    _arch_rsi, _arch_macd, _arch_bollinger, _arch_adx_trend,
]


def sample_genome(rng: np.random.Generator) -> dict:
    """A random, structurally-valid genome drawn from the archetype palette."""
    genome = ARCHETYPES[int(rng.integers(0, len(ARCHETYPES)))](rng)
    validate_genome(genome)
    return genome


def mutate_genome(genome: dict, rng: np.random.Generator, frac: float = 0.15, resample_prob: float = 0.1) -> dict:
    """Nudge a genome. Mostly perturbs its scalars (and toggles/retunes the
    trailing stop); occasionally jumps to a fresh archetype for diversity."""
    if rng.random() < resample_prob:
        return sample_genome(rng)
    g = copy.deepcopy(genome)
    paths = genome_param_paths(g)
    if paths:
        # nudge a random subset of scalars
        for path in rng.choice(list(paths.keys()), size=max(1, len(paths) // 3), replace=False):
            name = path.split(".")[-1]
            new = float(paths[path]) * (1.0 + float(rng.uniform(-frac, frac)))
            g = apply_genome_params(g, {path: _clamp(name, new)})
    # keep slow > fast where both exist
    _repair(g)
    # toggle the trailing stop
    risk = g.setdefault("risk", {})
    if "trail_atr_mult" in risk and rng.random() < 0.2:
        del risk["trail_atr_mult"]
    elif "trail_atr_mult" not in risk and rng.random() < 0.2:
        risk["trail_atr_mult"] = _clamp("trail_atr_mult", float(rng.uniform(2.0, 4.0)))
    validate_genome(g)
    return g


def crossover_genome(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """Recombine two genomes: take one parent's structure, inherit the other's
    risk block, and mix any scalars they share by name."""
    child = copy.deepcopy(a if rng.random() < 0.5 else b)
    donor = b if child is not a else a
    child["risk"] = copy.deepcopy(donor["risk"]) if rng.random() < 0.5 else child["risk"]
    a_paths, b_paths = genome_param_paths(a), genome_param_paths(b)
    shared = set(a_paths) & set(b_paths) & set(genome_param_paths(child))
    overrides = {p: (a_paths[p] if rng.random() < 0.5 else b_paths[p]) for p in shared}
    if overrides:
        child = apply_genome_params(child, overrides)
    _repair(child)
    validate_genome(child)
    return child


def _repair(genome: dict) -> None:
    """Fix invariants a mutation might break (fast<slow, exit_channel<entry)."""
    by_id = {s["id"]: s for s in genome["features"]}
    if "fast" in by_id and "slow" in by_id:
        f = int(by_id["fast"]["params"]["period"]); s = int(by_id["slow"]["params"]["period"])
        if f >= s:
            by_id["fast"]["params"]["period"] = max(4, s - 5)
    if "entry_hi" in by_id and "exit_hi" in by_id:
        en = int(by_id["entry_hi"]["params"]["period"]); ex = int(by_id["exit_hi"]["params"]["period"])
        if ex >= en:
            new_ex = max(3, en - 2)
            for fid in ("exit_hi", "exit_lo"):
                if fid in by_id:
                    by_id[fid]["params"]["period"] = new_ex
