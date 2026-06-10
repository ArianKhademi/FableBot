import numpy as np

from fablebot.indicators import atr, donchian, ema, zscore
from tests.conftest import make_ohlcv


def test_atr_constant_range():
    # Every bar spans exactly 1.0 (spread 0.5 both sides, flat closes)
    df = make_ohlcv([100.0] * 50, spread=0.5)
    value = atr(df, period=14).iloc[-1]
    assert abs(value - 1.0) < 1e-9


def test_atr_positive_and_scales_with_volatility():
    quiet = make_ohlcv(100 + np.sin(np.arange(60)) * 0.5)
    wild = make_ohlcv(100 + np.sin(np.arange(60)) * 10)
    assert atr(quiet, 14).iloc[-1] > 0
    assert atr(wild, 14).iloc[-1] > atr(quiet, 14).iloc[-1]


def test_zscore_flags_stretch():
    closes = [100.0] * 30 + [110.0]  # sudden jump from a flat series
    df = make_ohlcv(closes)
    z = zscore(df["close"], 20).iloc[-1]
    assert z > 2.0


def test_zscore_flat_series_is_nan():
    df = make_ohlcv([100.0] * 30)
    assert np.isnan(zscore(df["close"], 20).iloc[-1])


def test_donchian_excludes_current_bar():
    closes = list(np.linspace(100, 110, 30))
    df = make_ohlcv(closes, spread=0.0)
    upper, _ = donchian(df, 20)
    # Channel top is the prior bar's high, not the current (highest) close
    assert upper.iloc[-1] < df["high"].iloc[-1]
    assert upper.iloc[-1] == df["high"].iloc[-2]


def test_ema_tracks_trend_direction():
    df = make_ohlcv(np.linspace(100, 200, 100))
    fast = ema(df["close"], 10).iloc[-1]
    slow = ema(df["close"], 50).iloc[-1]
    assert fast > slow
