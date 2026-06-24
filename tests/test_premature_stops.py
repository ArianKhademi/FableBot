"""Premature-stop recovery: did a hard stop cut a trade that would have come
back to break-even? Pure (high, low)-path logic, no data feed."""

from fablebot.analytics import stop_recovery


def test_long_comes_back_to_break_even():
    # stopped long at 100; dips, reclaims 100 on bar 3, runs to 104
    post = [(98, 96), (99, 97), (101, 99), (104, 102)]
    r = stop_recovery("long", 100.0, post, horizon=10)
    assert r["recovered"] is True
    assert r["bars_to_recover"] == 3
    assert r["best_favorable_per_unit"] == 4.0   # 104 - 100


def test_long_never_comes_back():
    post = [(98, 96), (97, 95), (96, 94)]
    r = stop_recovery("long", 100.0, post, horizon=10)
    assert r["recovered"] is False
    assert r["bars_to_recover"] is None
    assert r["best_favorable_per_unit"] == 0.0   # never traded back above entry


def test_short_comes_back():
    # stopped short at 100; price spikes then falls back to 100 (bar 2), low of 97
    post = [(103, 101), (102, 100), (101, 97)]
    r = stop_recovery("short", 100.0, post, horizon=10)
    assert r["recovered"] is True
    assert r["bars_to_recover"] == 2
    assert r["best_favorable_per_unit"] == 3.0   # 100 - 97


def test_horizon_is_respected():
    post = [(98, 96)] * 5 + [(101, 99)]          # only recovers on bar 6
    assert stop_recovery("long", 100.0, post, horizon=5)["recovered"] is False
    assert stop_recovery("long", 100.0, post, horizon=6)["recovered"] is True
