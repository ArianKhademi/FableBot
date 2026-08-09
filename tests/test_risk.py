import pytest

from fablebot.risk import CorrelationFilter, size_position


def test_atr_sizing_keeps_dollar_risk_constant():
    # Quiet gold day vs volatile bitcoin day: same dollars at risk
    gold = size_position(100_000, 0.01, 2000.0, atr=8.0, atr_stop_mult=3.0, direction="long")
    btc = size_position(100_000, 0.01, 60_000.0, atr=1500.0, atr_stop_mult=2.5, direction="long")
    assert gold.risk_amount == pytest.approx(1000.0)
    assert btc.risk_amount == pytest.approx(1000.0)
    assert gold.units * gold.stop_distance == pytest.approx(1000.0)
    assert btc.units * btc.stop_distance == pytest.approx(1000.0)
    # Quieter instrument gets the bigger size
    assert gold.units > btc.units


def test_stop_price_sides():
    long = size_position(100_000, 0.01, 100.0, atr=2.0, atr_stop_mult=2.0, direction="long")
    short = size_position(100_000, 0.01, 100.0, atr=2.0, atr_stop_mult=2.0, direction="short")
    assert long.stop_price == pytest.approx(96.0)
    assert short.stop_price == pytest.approx(104.0)


def test_size_rejects_bad_inputs():
    with pytest.raises(ValueError):
        size_position(100_000, 0.01, 100.0, atr=0.0, atr_stop_mult=2.0, direction="long")
    with pytest.raises(ValueError):
        size_position(100_000, 0.01, 100.0, atr=1.0, atr_stop_mult=2.0, direction="sideways")


@pytest.fixture
def risk_on_filter():
    return CorrelationFilter({"risk_on": ["SPX", "NDX", "BTC"]}, max_same_direction=2)


def test_blocks_third_risk_on_long(risk_on_filter):
    open_positions = {"SPX": "long", "NDX": "long"}
    allowed, reason = risk_on_filter.allows("BTC", "long", open_positions)
    assert not allowed
    assert "correlation" in reason


def test_allows_second_risk_on_long(risk_on_filter):
    allowed, _ = risk_on_filter.allows("NDX", "long", {"SPX": "long"})
    assert allowed


def test_opposite_direction_not_blocked(risk_on_filter):
    open_positions = {"SPX": "long", "NDX": "long"}
    allowed, _ = risk_on_filter.allows("BTC", "short", open_positions)
    assert allowed


def test_ungrouped_instrument_never_blocked(risk_on_filter):
    open_positions = {"SPX": "long", "NDX": "long", "BTC": "long"}
    allowed, _ = risk_on_filter.allows("GOLD", "long", open_positions)
    assert allowed


def test_size_position_rejects_nan_atr_and_depleted_equity():
    import math
    import pytest as _pytest
    from fablebot.risk import size_position
    with _pytest.raises(ValueError):
        size_position(equity=100000, risk_per_trade=0.01, entry_price=100,
                      atr=math.nan, atr_stop_mult=2.0, direction="long")
    with _pytest.raises(ValueError):
        size_position(equity=0.0, risk_per_trade=0.01, entry_price=100,
                      atr=1.0, atr_stop_mult=2.0, direction="long")
