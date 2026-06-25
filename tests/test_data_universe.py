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


def test_cached_source_hit_returns_full_window_not_row_sliced(tmp_path):
    # intraday: `days` is a fetch *window* (calendar days), never a row count.
    # A real source returns the full windowed series; a cache hit must return
    # all of it, not its last `days` rows.
    df = make_ohlcv(np.linspace(100, 120, 4000), freq="1h")

    class FullSource:  # models YFinanceSource: windows by calendar, not rows
        def __init__(self): self.calls = 0
        def fetch(self, symbol, timeframe, days=None):
            self.calls += 1
            return df

    src = FullSource()
    cached = data.CachedSource(src, tmp_path, ttl_minutes=60)
    cached.fetch("BTCUSDT", "1h", days=700)        # miss: caches full 4000 bars
    hit = cached.fetch("BTCUSDT", "1h", days=700)  # hit: must not truncate to 700 rows
    assert len(hit) == 4000
    assert src.calls == 1


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


# ── local CSV history source (WRDS / CRSP exports) ───────────────────────────

def test_read_ohlcv_csv_standard_and_crsp(tmp_path):
    std = tmp_path / "SPY.csv"
    std.write_text("Date,Open,High,Low,Close,Volume\n2026-01-02,10,11,9,10.5,1000\n2026-01-03,10.5,12,10,11.5,1200\n")
    df = data._read_ohlcv_csv(std)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [10.5, 11.5]
    assert isinstance(df.index, pd.DatetimeIndex)
    # CRSP layout: caldt + OPENPRC/ASKHI/BIDLO/PRC (PRC negative => bid/ask midpoint) + VOL
    crsp = tmp_path / "GSPC.csv"
    crsp.write_text("caldt,OPENPRC,ASKHI,BIDLO,PRC,VOL\n2026-01-02,10,11,9,-10.5,1000\n")
    c = data._read_ohlcv_csv(crsp)
    assert c["close"].tolist() == [10.5]   # magnitude of -10.5
    assert c["high"].tolist() == [11.0]


def test_read_ohlcv_csv_close_only_fills_ohlc(tmp_path):
    p = tmp_path / "X.csv"
    p.write_text("date,close\n2026-01-02,100\n2026-01-03,101\n")
    df = data._read_ohlcv_csv(p)
    assert df["open"].tolist() == df["close"].tolist() == [100.0, 101.0]
    assert (df["volume"] == 0.0).all()


def test_csv_source_serves_daily_then_falls_back(tmp_path):
    (tmp_path / "SPY.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n2026-01-02,10,11,9,10.5,1000\n2026-01-03,10.5,12,10,11.5,1200\n")
    fallback = _StubSource(make_ohlcv(np.linspace(100, 110, 30), freq="D"))
    src = data.CSVSource(tmp_path, fallback=fallback)
    assert src.fetch("SPY", "1d")["close"].tolist() == [10.5, 11.5]   # from CSV
    assert fallback.calls == 0
    assert len(src.fetch("AAPL", "1d")) == 30 and fallback.calls == 1   # no CSV -> fallback


def test_csv_source_only_intercepts_daily(tmp_path):
    (tmp_path / "SPY.csv").write_text("Date,Close\n2026-01-02,10.5\n")
    fallback = _StubSource(make_ohlcv(np.linspace(100, 110, 30), freq="h"))
    data.CSVSource(tmp_path, fallback=fallback, timeframes=("1d",)).fetch("SPY", "1h")
    assert fallback.calls == 1   # intraday -> fallback even though SPY.csv exists


def test_csv_source_matches_caret_symbol(tmp_path):
    (tmp_path / "GSPC.csv").write_text("Date,Close\n2026-01-02,4000\n")
    fallback = _StubSource(make_ohlcv(np.linspace(100, 110, 5), freq="D"))
    src = data.CSVSource(tmp_path, fallback=fallback)
    assert src.fetch("^GSPC", "1d")["close"].tolist() == [4000.0] and fallback.calls == 0


def test_source_from_config_uses_csv_for_history_not_live(tmp_path):
    config = load_config("config.yaml")
    config.data.csv_dir = str(tmp_path)
    history = data.source_from_config(config, cache_dir=str(tmp_path / "cache"))
    assert isinstance(history, data.CSVSource)          # search/history feed wraps CSV
    live = data.source_from_config(config, cache_dir=None)
    assert not isinstance(live, data.CSVSource)         # live feed stays real-time


# ── per-instrument source routing (crypto -> Binance) ────────────────────────

def test_binance_symbol_translation():
    assert data._to_binance_symbol("BTC-USD") == "BTCUSDT"
    assert data._to_binance_symbol("ETH/USD") == "ETHUSDT"
    assert data._to_binance_symbol("BTCUSDT") == "BTCUSDT"


def test_routing_source_dispatches_by_symbol():
    eq = _StubSource(make_ohlcv(np.linspace(100, 110, 30), freq="1d"))
    cr = _StubSource(make_ohlcv(np.linspace(200, 210, 30), freq="1d"))
    router = data.RoutingSource(default=eq, routes={"BTC-USD": cr})
    router.fetch("AAPL", "1d")
    router.fetch("BTC-USD", "1d")
    assert eq.calls == 1 and cr.calls == 1  # each went to its own backend


def test_source_from_config_routes_crypto_to_binance():
    config = load_config("config.yaml")
    # uncached (live feed): BTC routes to a raw BinanceSource, equities to default
    live = data.source_from_config(config, cache_dir=None)
    assert isinstance(live, data.RoutingSource)
    assert isinstance(live.routes["BTC-USD"], data.BinanceSource)
    assert isinstance(live.default, data.YFinanceSource)
    # cached (history feed): routed backends are wrapped in the on-disk cache.
    # When config.data.csv_dir is set, the whole cached feed is further wrapped in
    # a CSVSource (history-only); the routing is then its fallback. Unwrap it so the
    # assertion holds whether or not a local CSV history is configured.
    cached = data.source_from_config(config, cache_dir="state/data_cache")
    routing = cached.fallback if isinstance(cached, data.CSVSource) else cached
    assert isinstance(routing.routes["BTC-USD"], data.CachedSource)
    assert isinstance(routing.routes["BTC-USD"].source, data.BinanceSource)


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
