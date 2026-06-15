"""Market data behind a swappable source interface.

The engine only ever calls an :class:`OHLCVSource`; which one it gets is a
config choice, so adding Polygon/Tiingo later (Workstream H, paid) is a new
adapter, not a rewrite. Today's free, no-API sources cover a lot:

  - **yfinance** — decades of *daily* history, plus 15m/1h intraday (Yahoo caps
    intraday: ~7-8d of 1m, ~60d of 15m), 4h resampled from 1h.
  - **Stooq** — free deep *daily* CSV, no key.
  - **Binance** — free deep *intraday* crypto (1m back years) via public klines.

A :class:`CachedSource` wraps any source with an on-disk CSV cache so repeated
backtests don't re-hit the network. Columns are always normalised to lowercase
open/high/low/close/volume with a DatetimeIndex.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd

# yfinance intraday windows (and a deep daily default). 30s bars are not
# offered; 1m ~ last 7-8d, 15m ~ last 60d — deep intraday equity history needs
# a paid source. Daily is effectively unlimited.
_FETCH_WINDOW = {"15m": "5d", "1h": "30d", "4h": "60d", "1d": "10y"}

BARS_PER_YEAR = {"15m": 252 * 26, "1h": 252 * 6.5, "4h": 252 * 1.625, "1d": 252}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols].dropna(subset=["close"])


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=["close"])


def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:  # kept for back-compat
    return resample_ohlcv(df_1h, "4h")


# ── source interface + adapters ──────────────────────────────────────────────


class OHLCVSource(Protocol):
    def fetch(self, symbol: str, timeframe: str, days: Optional[int] = None) -> pd.DataFrame:
        ...


class YFinanceSource:
    """Yahoo Finance. 4h is resampled from 1h; everything else is native."""

    name = "yfinance"

    def fetch(self, symbol: str, timeframe: str, days: Optional[int] = None) -> pd.DataFrame:
        import yfinance as yf

        if timeframe not in _FETCH_WINDOW:
            raise ValueError(f"Unsupported timeframe {timeframe!r}")
        if days is not None and days <= 0:
            raise ValueError(f"days must be positive, got {days}")
        interval = "1h" if timeframe == "4h" else timeframe
        period = f"{days}d" if days is not None else _FETCH_WINDOW[timeframe]
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {symbol} ({timeframe})")
        df = _normalize(df)
        if timeframe == "4h":
            df = resample_ohlcv(df, "4h")
        return df


def _parse_stooq_csv(text: str) -> pd.DataFrame:
    import io

    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise RuntimeError("unexpected Stooq payload")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return _normalize(df)


class StooqSource:
    """Free deep daily history via Stooq CSV (daily only)."""

    name = "stooq"

    def fetch(self, symbol: str, timeframe: str, days: Optional[int] = None) -> pd.DataFrame:
        import requests

        if timeframe != "1d":
            raise ValueError("StooqSource only serves daily ('1d') data")
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = _parse_stooq_csv(resp.text)
        if days is not None:
            df = df.iloc[-days:]
        if df.empty:
            raise RuntimeError(f"No Stooq data for {symbol}")
        return df


_BINANCE_INTERVAL = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}


def _parse_binance_klines(rows: list) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "trades", "tbbav", "tbqav", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms")
    df = df.set_index("ts")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return _normalize(df)


class BinanceSource:
    """Free deep intraday crypto via Binance public klines (no key)."""

    name = "binance"

    def fetch(self, symbol: str, timeframe: str, days: Optional[int] = None) -> pd.DataFrame:
        import requests

        if timeframe not in _BINANCE_INTERVAL:
            raise ValueError(f"Unsupported Binance timeframe {timeframe!r}")
        params = {"symbol": symbol.upper(), "interval": _BINANCE_INTERVAL[timeframe], "limit": 1000}
        resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
        resp.raise_for_status()
        df = _parse_binance_klines(resp.json())
        if df.empty:
            raise RuntimeError(f"No Binance data for {symbol}")
        return df


class CachedSource:
    """Wrap any source with an on-disk CSV cache keyed by (symbol, timeframe).
    A cache hit within ``ttl_minutes`` skips the network entirely."""

    def __init__(self, source: OHLCVSource, cache_dir: Path | str, ttl_minutes: float = 1440):
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = float(ttl_minutes) * 60.0

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in symbol)
        return self.cache_dir / f"{safe}__{timeframe}.csv"

    def fetch(self, symbol: str, timeframe: str, days: Optional[int] = None) -> pd.DataFrame:
        path = self._path(symbol, timeframe)
        if path.exists() and (time.time() - path.stat().st_mtime) < self.ttl_seconds:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if days is not None:
                df = df.iloc[-days:]
            return df
        df = self.source.fetch(symbol, timeframe, days)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        return df


_SOURCES = {"yfinance": YFinanceSource, "stooq": StooqSource, "binance": BinanceSource}


def get_source(
    name: str = "yfinance",
    cache_dir: Optional[Path | str] = None,
    ttl_minutes: float = 1440,
) -> OHLCVSource:
    """Build a source by name, optionally wrapped in an on-disk cache."""
    if name not in _SOURCES:
        raise ValueError(f"Unknown data source {name!r}; choose from {sorted(_SOURCES)}")
    source: OHLCVSource = _SOURCES[name]()
    if cache_dir is not None:
        source = CachedSource(source, cache_dir, ttl_minutes)
    return source


def fetch_ohlcv(symbol: str, timeframe: str, days: int | None = None) -> pd.DataFrame:
    """Backward-compatible convenience: fetch via the default yfinance source.

    Fetch recent OHLCV bars for ``symbol`` at ``timeframe`` ('15m', '1h', '4h',
    '1d'); ``days`` overrides the default window.
    """
    return YFinanceSource().fetch(symbol, timeframe, days)
