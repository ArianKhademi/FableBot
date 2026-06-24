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
from .base import Signal, Strategy
from .mean_reversion import MeanReversion
from .momentum_breakout import MomentumBreakout


class RegimeSwitch(Strategy):
    name = "regime"

    def __init__(self, params: dict):
        super().__init__(params)
        self.adx_period = int(self.params.get("adx_period", 14))
        self.adx_threshold = float(self.params.get("adx_threshold", 25.0))
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

    def _active(self, df: pd.DataFrame):
        """(active sub-strategy, adx value, trending?)."""
        a = self._adx(df)
        trending = (not math.isnan(a)) and a >= self.adx_threshold
        return (self.trend_strategy if trending else self.range_strategy), a, trending

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        strat, a, trending = self._active(df)
        sig = strat.evaluate(df, position_direction)
        regime = "trend" if trending else "range"
        adx_str = f"{a:.0f}" if not math.isnan(a) else "?"
        return Signal(sig.action, f"[{regime} adx={adx_str}] {sig.reason}", sig.atr)

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
