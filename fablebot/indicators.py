"""Technical indicators. All functions take pandas inputs and return Series.

DataFrames are expected to carry lowercase ``open/high/low/close`` columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed."""
    return true_range(df).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def zscore(series: pd.Series, lookback: int) -> pd.Series:
    """How many standard deviations the series sits from its rolling mean."""
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def donchian(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """Prior-bar Donchian channel: highest high / lowest low of the previous
    ``period`` bars, excluding the current bar so a breakout can be detected
    against it."""
    upper = df["high"].rolling(period).max().shift(1)
    lower = df["low"].rolling(period).min().shift(1)
    return upper, lower
