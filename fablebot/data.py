"""Market data via Yahoo Finance.

Yahoo serves 15m and 1h bars directly; 4h is resampled from 1h. Columns are
normalized to lowercase ``open/high/low/close/volume``.
"""

from __future__ import annotations

import pandas as pd

# How much history to request per timeframe — enough for the slowest
# indicator (50-bar EMA on 4h) with margin, within Yahoo's intraday limits.
_FETCH_WINDOW = {"15m": "5d", "1h": "30d", "4h": "60d"}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols].dropna(subset=["close"])


def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df_1h.columns:
        agg["volume"] = "sum"
    return df_1h.resample("4h").agg(agg).dropna(subset=["close"])


def fetch_ohlcv(symbol: str, timeframe: str, days: int | None = None) -> pd.DataFrame:
    """Fetch recent OHLCV bars for ``symbol`` at ``timeframe``
    ('15m', '1h', or '4h').

    When ``days`` is set, request that many calendar days of history instead of
    the default fetch window for the timeframe.
    """
    import yfinance as yf

    if timeframe not in _FETCH_WINDOW:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    if days is not None and days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    interval = "1h" if timeframe == "4h" else timeframe
    period = f"{days}d" if days is not None else _FETCH_WINDOW[timeframe]
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol} ({timeframe})")
    df = _normalize(df)
    if timeframe == "4h":
        df = resample_4h(df)
    return df
