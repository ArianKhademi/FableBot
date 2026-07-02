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
        # Optional adaptive trailing stop: the trail tightens from trail_atr_mult
        # (loose, at entry, far from the mean) to trail_near_mult (tight, as price
        # nears the mean). Unset => the trail is constant (or absent if no
        # trail_atr_mult). See current_trail_mult.
        near = self.params.get("trail_near_mult")
        self.trail_near_mult = float(near) if near is not None else None
        # Optional trend filter: don't fade AGAINST a longer-term trend. When set,
        # skip shorts while price is above its trend_filter_ma-bar average, and skip
        # longs while below it — so we only mean-revert with (or neutral to) the
        # primary trend, never into it. Kills the counter-trend fades that bleed in
        # a sustained move. Backtests best with a LONG MA relative to lookback.
        tfm = self.params.get("trend_filter_ma")
        self.trend_filter_ma = int(tfm) if tfm else None

    def min_bars(self) -> int:
        return max(self.lookback, self.atr_period) + 1

    def _z(self, df: pd.DataFrame) -> float:
        return float(zscore(df["close"], self.lookback).iloc[-1])

    def current_trail_mult(self, df: pd.DataFrame, position_direction: Optional[str]) -> Optional[float]:
        """Narrowing trail: loose far from the mean so the reversion has room,
        tightening toward ``trail_near_mult`` as |z| falls toward ``exit_z`` (near
        the target, where giving back the gain to a bounce-away hurts most).
        Linearly interpolates on |z| between entry_z (loose) and exit_z (tight);
        without trail_near_mult it's the constant trail."""
        if not self.trail_atr_mult or self.trail_near_mult is None:
            return self.trail_atr_mult
        z = abs(self._z(df))
        if math.isnan(z):
            return self.trail_atr_mult
        span = self.entry_z - self.exit_z
        progress = 1.0 if span <= 0 else (self.entry_z - z) / span
        progress = max(0.0, min(1.0, progress))   # 0 at entry_z (far), 1 at exit_z (near)
        return self.trail_atr_mult + progress * (self.trail_near_mult - self.trail_atr_mult)

    def _fights_trend(self, df: pd.DataFrame, opening: Action) -> bool:
        """True if opening this fade would fight the longer-term trend: a short
        while price is above the trend MA, or a long while below it."""
        if not self.trend_filter_ma or len(df) <= self.trend_filter_ma:
            return False
        sma = float(df["close"].rolling(self.trend_filter_ma).mean().iloc[-1])
        px = float(df["close"].iloc[-1])
        if math.isnan(sma):
            return False
        if opening is Action.OPEN_SHORT and px > sma:
            return True
        if opening is Action.OPEN_LONG and px < sma:
            return True
        return False

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
            if self._fights_trend(df, Action.OPEN_LONG):
                return Signal(Action.HOLD, f"oversold but below trend — not fading (z={z:+.2f})", cur_atr)
            return Signal(Action.OPEN_LONG, f"oversold stretch (z={z:+.2f})", cur_atr)
        if z >= self.entry_z:
            if self._fights_trend(df, Action.OPEN_SHORT):
                return Signal(Action.HOLD, f"overbought but above trend — not fading (z={z:+.2f})", cur_atr)
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
