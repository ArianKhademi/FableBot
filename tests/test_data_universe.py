"""Data source interface (caching, normalisation, payload parsing) and the
config-driven universe + screener. No network: adapters are exercised through
their pure parsers and a stub source."""

import time

import numpy as np
import pandas as pd
import pytest

from fablebot import data, universe
from fablebot.config import load_config, universe_instruments
from tests.conftest import make_ohlcv


# ── normalisation / resampling ───────────────────────────────────────────────

def test_normalize_lowercases_and_drops_nan_close():
    raw = pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3], "Low": [0, 1], "Close": [1.5, np.nan], "Volume": [10, 20]},
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )
    out = data._normalize(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 1  # the NaN-close row is dropped


def test_resample_aggregates_ohlcv():
    df = make_ohlcv(np.linspace(100, 110, 8), freq="1h")
    four_h = data.resample_ohlcv(df, "4h")
    assert len(four_h) == 2
    assert four_h["high"].iloc[0] == df["high"].iloc[:4].max()
    assert four_h["low"].iloc[0] == df["low"].iloc[:4].min()


# ── payload parsers (adapters without the network) ───────────────────────────

def test_parse_binance_klines():
    rows = [
        [1700000000000, "100.0", "110.0", "95.0", "105.0", "12.0", 0, "0", 3, "0", "0", "0"],
        [1700003600000, "105.0", "115.0", "100.0", "108.0", "8.0", 0, "0", 2, "0", "0", "0"],
    ]
    df = data._parse_binance_klines(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [105.0, 108.0]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_parse_stooq_csv():
    text = "Date,Open,High,Low,Close,Volume\n2026-01-02,10,11,9,10.5,1000\n2026-01-03,10.5,12,10,11.5,1200\n"
    df = data._parse_stooq_csv(text)
    assert df["close"].tolist() == [10.5, 11.5]
    assert df.index[0] == pd.Timestamp("2026-01-02")


# ── caching wrapper ──────────────────────────────────────────────────────────

class _StubSource:
    def __init__(self, df):
        self.df = df
        self.calls = 0

    def fetch(self, symbol, timeframe, days=None):
        self.calls += 1
        df = self.df
        return df.iloc[-days:] if days else df


def test_cached_source_avoids_refetch(tmp_path):
    df = make_ohlcv(np.linspace(100, 120, 50), freq="D")
    stub = _StubSource(df)
    cached = data.CachedSource(stub, tmp_path, ttl_minutes=60)
    a = cached.fetch("XYZ", "1d")
    b = cached.fetch("XYZ", "1d")  # served from disk
    assert stub.calls == 1
    assert len(a) == len(b) == len(df)
    assert (tmp_path / "XYZ__1d.csv").exists()


def test_cached_source_refetches_when_stale(tmp_path):
    df = make_ohlcv(np.linspace(100, 120, 50), freq="D")
    stub = _StubSource(df)
    cached = data.CachedSource(stub, tmp_path, ttl_minutes=0)  # always stale
    cached.fetch("XYZ", "1d")
    cached.fetch("XYZ", "1d")
    assert stub.calls == 2


def test_get_source_unknown_raises():
    with pytest.raises(ValueError):
        data.get_source("nope")


# ── universe config + screener ───────────────────────────────────────────────

def test_config_loads_universe_and_data(tmp_path):
    config = load_config("config.yaml")
    assert config.data.source in data._SOURCES
    insts = universe_instruments(config)
    assert len(insts) >= 10
    # universe defaults applied; any genome can later run on any of these
    assert all(i.timeframe == config.universe.default_timeframe for i in insts)
    assert {i.symbol for i in insts} >= {"SPY", "QQQ", "BTC-USD"}


def test_average_dollar_volume_and_screen():
    df = make_ohlcv(np.linspace(100, 100, 80), freq="D")
    df = df.copy()
    df["volume"] = 1000.0
    assert universe.average_dollar_volume(df, 60) == pytest.approx(100 * 1000.0)

    from fablebot.config import InstrumentConfig

    class MultiStub:
        def __init__(self):
            self.vols = {"A": 5000.0, "B": 50000.0, "C": 500.0}

        def fetch(self, symbol, timeframe, days=None):
            d = make_ohlcv(np.linspace(100, 100, 300), freq="D").copy()
            d["volume"] = self.vols[symbol] / 100.0  # close=100 -> dollar vol = vols[symbol]
            return d

    insts = [InstrumentConfig(key=s, symbol=s, name=s, strategy="trend_following", timeframe="1d") for s in "ABC"]
    ranked = universe.screen_universe(insts, MultiStub(), top_n=2)
    assert [i.symbol for i in ranked] == ["B", "A"]  # most liquid first, capped at 2
