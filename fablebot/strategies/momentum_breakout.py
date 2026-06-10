"""Momentum breakouts for Bitcoin on 1-hour bars.

Crypto trends harder than indices, so ride the move instead of fading it:
a close beyond the prior ``entry_channel``-bar extreme opens a position, and
the opposite side of a shorter ``exit_channel`` keeps you in the trend until
it actually bends.
"""

from __future__ import annotations

from typing import Optional

import math

import pandas as pd

from ..indicators import donchian
from .base import Action, Signal, Strategy


class MomentumBreakout(Strategy):
    name = "momentum_breakout"

    def __init__(self, params: dict):
        super().__init__(params)
        self.entry_channel = int(self.params.get("entry_channel", 20))
        self.exit_channel = int(self.params.get("exit_channel", 10))

    def min_bars(self) -> int:
        return max(self.entry_channel, self.atr_period) + 2

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        close = float(df["close"].iloc[-1])
        cur_atr = self.current_atr(df)
        entry_hi, entry_lo = donchian(df, self.entry_channel)
        exit_hi, exit_lo = donchian(df, self.exit_channel)
        hi, lo = float(entry_hi.iloc[-1]), float(entry_lo.iloc[-1])

        if math.isnan(hi) or math.isnan(lo):
            return Signal(Action.HOLD, "insufficient data", cur_atr)

        if position_direction == "long":
            trail = float(exit_lo.iloc[-1])
            if close < trail:
                return Signal(
                    Action.CLOSE,
                    f"closed below {self.exit_channel}-bar low {trail:,.2f}",
                    cur_atr,
                )
            return Signal(Action.HOLD, "riding the breakout", cur_atr)
        if position_direction == "short":
            trail = float(exit_hi.iloc[-1])
            if close > trail:
                return Signal(
                    Action.CLOSE,
                    f"closed above {self.exit_channel}-bar high {trail:,.2f}",
                    cur_atr,
                )
            return Signal(Action.HOLD, "riding the breakdown", cur_atr)

        if close > hi:
            return Signal(
                Action.OPEN_LONG,
                f"breakout above {self.entry_channel}-bar high {hi:,.2f}",
                cur_atr,
            )
        if close < lo:
            return Signal(
                Action.OPEN_SHORT,
                f"breakdown below {self.entry_channel}-bar low {lo:,.2f}",
                cur_atr,
            )
        return Signal(Action.HOLD, "inside channel", cur_atr)

    def watching(self, df: pd.DataFrame) -> str:
        entry_hi, entry_lo = donchian(df, self.entry_channel)
        hi, lo = float(entry_hi.iloc[-1]), float(entry_lo.iloc[-1])
        if math.isnan(hi):
            return "warming up"
        return f"breakout levels: above {hi:,.2f} or below {lo:,.2f} ({self.entry_channel}-bar channel)"
