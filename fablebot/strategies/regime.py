"""Regime-switching strategy: mean-reversion in range-bound markets, momentum in
trending ones, decided each bar by ADX (trend strength).

The classic failure of mean-reversion is fading a strong, sustained trend (price
keeps breaking past the mean); the classic failure of momentum is whipsawing in
chop. ADX measures exactly that — trend strength — so this mean-reverts when ADX
is low (range) and rides momentum when ADX is high (trend). Both sub-strategies
are the existing, tested ones; this only routes between them, so it inherits
their entries, exits, and the same immutable risk engine.

Note (v1): the active regime is recomputed every bar, so a position opened in one
regime is then managed by whichever regime is current — fine for an experiment,
but it means a mid-trade regime flip hands the exit to the other strategy's logic.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..indicators import adx
from .base import Action, Signal, Strategy
from .mean_reversion import MeanReversion
from .momentum_breakout import MomentumBreakout


class RegimeSwitch(Strategy):
    name = "regime"

    def __init__(self, params: dict):
        super().__init__(params)
        self.adx_period = int(self.params.get("adx_period", 14))
        self.adx_threshold = float(self.params.get("adx_threshold", 25.0))
        # Optional three-zone mode: when set, momentum only fires at ADX >= this
        # (a genuinely strong trend), mean-reversion only below adx_threshold,
        # and the band in between HOLDS — live data showed the marginal
        # ADX-25..40 "trends" whipsawed momentum (9/9 losers) while the only
        # winners fired at ADX 50+. Unset => classic two-zone switch.
        amt = self.params.get("adx_momentum_threshold")
        self.adx_momentum_threshold = float(amt) if amt is not None else None
        # Both sub-strategies read the same params dict; each picks the keys it
        # needs (lookback/entry_z for MR, entry_channel for momentum) and ignores
        # the rest, so one config block configures both.
        self.range_strategy = MeanReversion(params)
        self.trend_strategy = MomentumBreakout(params)

    def min_bars(self) -> int:
        return max(self.range_strategy.min_bars(), self.trend_strategy.min_bars(), self.adx_period + 1)

    def _adx(self, df: pd.DataFrame) -> float:
        series, _, _ = adx(df, self.adx_period)
        return float(series.iloc[-1])

    def _mode(self, df: pd.DataFrame) -> tuple[str, float]:
        """('range' | 'mid' | 'trend', adx). 'mid' only exists in three-zone mode."""
        a = self._adx(df)
        if math.isnan(a):
            return "range", a
        hi = self.adx_momentum_threshold if self.adx_momentum_threshold is not None else self.adx_threshold
        if a >= hi:
            return "trend", a
        if a < self.adx_threshold:
            return "range", a
        return "mid", a

    def _active(self, df: pd.DataFrame):
        """(active sub-strategy, adx value, trending?). In the mid band the range
        strategy is the manager (it can exit a held position at the mean) but
        evaluate() blocks new entries there."""
        mode, a = self._mode(df)
        return (self.trend_strategy if mode == "trend" else self.range_strategy), a, mode == "trend"

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        mode, a = self._mode(df)
        adx_str = f"{a:.0f}" if not math.isnan(a) else "?"
        if mode == "mid":
            # No fresh risk in the in-between band; a held position is still
            # managed (exits/holds) by the range strategy so it isn't stranded.
            sig = self.range_strategy.evaluate(df, position_direction)
            if sig.opens:
                return Signal(Action.HOLD, f"[mid adx={adx_str}] between regimes — no new entries", sig.atr)
            return Signal(sig.action, f"[mid adx={adx_str}] {sig.reason}", sig.atr)
        strat = self.trend_strategy if mode == "trend" else self.range_strategy
        sig = strat.evaluate(df, position_direction)
        return Signal(sig.action, f"[{mode} adx={adx_str}] {sig.reason}", sig.atr)

    def context(self, df: pd.DataFrame) -> dict:
        strat, a, trending = self._active(df)
        ctx = strat.context(df)
        ctx["adx"] = round(a, 2) if not math.isnan(a) else None
        ctx["regime"] = "trend" if trending else "range"
        return ctx

    def watching(self, df: pd.DataFrame) -> str:
        strat, a, trending = self._active(df)
        regime = "trend → momentum" if trending else "range → mean-reversion"
        adx_str = f"{a:.0f}" if not math.isnan(a) else "warming up"
        return f"regime: {regime} (adx={adx_str})"
