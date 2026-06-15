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


# ── extended palette (used by the strategy DSL and ML features) ──────────────


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, 0–100."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # all-gains windows have zero average loss -> RSI 100
    out = out.where(avg_loss != 0.0, 100.0)
    return out.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return line, sig, line - sig


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index with +DI / -DI (Wilder)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean() / atr_.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx_, plus_di, minus_di


def bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands: (upper, middle, lower)."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def keltner(
    df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner channels around an EMA, width scaled by ATR."""
    mid = ema(df["close"], period)
    width = atr_mult * atr(df, period)
    return mid + width, mid, mid - width


def volume_zscore(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """How unusual the current bar's volume is vs its rolling history."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return zscore(df["volume"].astype(float), lookback)
