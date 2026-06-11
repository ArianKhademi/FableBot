"""Strategy interface.

A strategy looks at an OHLCV DataFrame and the current position (if any) and
emits a :class:`Signal`. It knows nothing about sizing, stops, correlation,
or execution — that all belongs to the risk engine and broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from ..indicators import atr


class Action(Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class Signal:
    action: Action
    reason: str
    atr: float = float("nan")  # current ATR, used for sizing and stops

    @property
    def opens(self) -> bool:
        return self.action in (Action.OPEN_LONG, Action.OPEN_SHORT)

    @property
    def direction(self) -> Optional[str]:
        if self.action is Action.OPEN_LONG:
            return "long"
        if self.action is Action.OPEN_SHORT:
            return "short"
        return None


class Strategy:
    name = "base"

    def __init__(self, params: dict):
        self.params = dict(params)
        self.atr_period = int(self.params.get("atr_period", 14))
        self.atr_stop_mult = float(self.params.get("atr_stop_mult", 2.0))

    def min_bars(self) -> int:
        """Bars of history required before signals are meaningful."""
        return self.atr_period + 1

    def current_atr(self, df: pd.DataFrame) -> float:
        return float(atr(df, self.atr_period).iloc[-1])

    def evaluate(self, df: pd.DataFrame, position_direction: Optional[str]) -> Signal:
        """``position_direction`` is 'long', 'short', or None when flat."""
        raise NotImplementedError

    def watching(self, df: pd.DataFrame) -> str:
        """One-line description of what the strategy is waiting for,
        used in the morning briefing."""
        return ""

    def context(self, df: pd.DataFrame) -> dict:
        """Indicator state at decision time, written to the decision journal
        so the learning loop can reconstruct what the strategy saw."""
        close = float(df["close"].iloc[-1])
        cur_atr = self.current_atr(df)
        return {
            "close": close,
            "atr": cur_atr,
            "atr_pct": (cur_atr / close * 100.0) if close else 0.0,
        }

    def _hold(self, df: pd.DataFrame, reason: str = "no signal") -> Signal:
        return Signal(Action.HOLD, reason, self.current_atr(df))
