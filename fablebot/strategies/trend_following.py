"""Slow trend following for commodities (gold, oil) on 4-hour bars.

Commodities move in cleaner waves; the 4-hour timeframe filters out intraday
whipsaws. Classic dual-EMA system: fast above slow → long, fast below slow →
short, position flipped on the opposite cross.
"""

from __future__ import annotations

from typing import Optional

import math

import pandas as pd

from ..indicators import ema
from .base import Action, Signal, Strategy


class TrendFollowing(Strategy):
    name = "trend_following"

    def __init__(self, params: dict):
        super().__init__(params)
        self.fast = int(self.params.get("fast_ema", 20))
        self.slow = int(self.params.get("slow_ema", 50))

    def min_bars(self) -> int:
        return max(self.slow, self.atr_period) + 2

    def _emas(self, df: pd.DataFrame) -> tuple[float, float]:
        close = df["close"]
        return float(ema(close, self.fast).iloc[-1]), float(ema(close, self.slow).iloc[-1])

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        fast, slow = self._emas(df)
        cur_atr = self.current_atr(df)
        if math.isnan(fast) or math.isnan(slow):
            return Signal(Action.HOLD, "insufficient data", cur_atr)

        trend = "long" if fast > slow else "short"
        desc = f"EMA{self.fast} {fast:,.2f} vs EMA{self.slow} {slow:,.2f}"

        if position_direction is None:
            if trend == "long":
                return Signal(Action.OPEN_LONG, f"uptrend ({desc})", cur_atr)
            return Signal(Action.OPEN_SHORT, f"downtrend ({desc})", cur_atr)

        if position_direction != trend:
            return Signal(Action.CLOSE, f"trend flipped ({desc})", cur_atr)
        return Signal(Action.HOLD, f"trend intact ({desc})", cur_atr)

    def watching(self, df: pd.DataFrame) -> str:
        fast, slow = self._emas(df)
        if math.isnan(fast) or math.isnan(slow):
            return "warming up"
        trend = "up" if fast > slow else "down"
        return f"{trend}trend — EMA{self.fast} {fast:,.2f} vs EMA{self.slow} {slow:,.2f}"
