"""Mean reversion for index products (S&P 500, Nasdaq) on 15-minute bars.

Indices make small overextensions every few hours. Measure the close as a
z-score against its rolling mean: stretched beyond ``entry_z`` standard
deviations, fade the move; exit once price reverts inside ``exit_z``.
"""

from __future__ import annotations

from typing import Optional

import math

import pandas as pd

from ..indicators import zscore
from .base import Action, Signal, Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(self, params: dict):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 20))
        self.entry_z = float(self.params.get("entry_z", 2.0))
        self.exit_z = float(self.params.get("exit_z", 0.5))

    def min_bars(self) -> int:
        return max(self.lookback, self.atr_period) + 1

    def _z(self, df: pd.DataFrame) -> float:
        return float(zscore(df["close"], self.lookback).iloc[-1])

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        z = self._z(df)
        cur_atr = self.current_atr(df)
        if math.isnan(z):
            return Signal(Action.HOLD, "insufficient data", cur_atr)

        if position_direction == "long":
            if z >= -self.exit_z:
                return Signal(Action.CLOSE, f"reverted to mean (z={z:+.2f})", cur_atr)
            return Signal(Action.HOLD, f"awaiting reversion (z={z:+.2f})", cur_atr)
        if position_direction == "short":
            if z <= self.exit_z:
                return Signal(Action.CLOSE, f"reverted to mean (z={z:+.2f})", cur_atr)
            return Signal(Action.HOLD, f"awaiting reversion (z={z:+.2f})", cur_atr)

        if z <= -self.entry_z:
            return Signal(Action.OPEN_LONG, f"oversold stretch (z={z:+.2f})", cur_atr)
        if z >= self.entry_z:
            return Signal(Action.OPEN_SHORT, f"overbought stretch (z={z:+.2f})", cur_atr)
        return Signal(Action.HOLD, f"inside band (z={z:+.2f})", cur_atr)

    def watching(self, df: pd.DataFrame) -> str:
        z = self._z(df)
        if math.isnan(z):
            return "warming up"
        return (
            f"z-score {z:+.2f} vs ±{self.entry_z:.1f} entry band — "
            f"fading overextensions on the {self.lookback}-bar mean"
        )

    def context(self, df: pd.DataFrame) -> dict:
        ctx = super().context(df)
        z = self._z(df)
        ctx.update(zscore=None if math.isnan(z) else z, entry_z=self.entry_z, exit_z=self.exit_z)
        return ctx
