import pytest

from fablebot.risk import CorrelationFilter, size_position


@pytest.mark.parametrize("risk", [0, -0.01, 0.010001, float("nan"), float("inf"), True, "0.01"])
def test_risk_ceiling_and_invalid_budgets_rejected(risk):
    from fablebot.config import AccountConfig

    with pytest.raises(ValueError, match="risk_per_trade"):
        AccountConfig(risk_per_trade=risk)
    with pytest.raises(ValueError, match="risk_per_trade"):
        size_position(100_000, risk, 100, 2, 2, "long")


@pytest.mark.parametrize("field,value", [
    ("equity", float("nan")), ("equity", float("inf")),
    ("atr", float("inf")), ("atr_stop_mult", 0), ("atr_stop_mult", -1),
    ("atr_stop_mult", float("nan")), ("entry_price", float("nan")),
    ("contract_multiplier", 0), ("contract_multiplier", -5),
    ("max_contracts", -1), ("max_contracts", 1.5),
])
def test_invalid_sizing_cannot_create_nonfinite_or_inverted_positions(field, value):
    kwargs = dict(equity=100_000, risk_per_trade=0.01, entry_price=100,
                  atr=2, atr_stop_mult=2, direction="long")
    kwargs[field] = value
    with pytest.raises(ValueError):
        size_position(**kwargs)


def test_custom_yaml_cannot_bypass_risk_ceiling(tmp_path):
    from fablebot.config import load_config

    path = tmp_path / "custom.yaml"
    path.write_text("account:\n  risk_per_trade: 0.02\n")
    with pytest.raises(ValueError, match="risk_per_trade"):
        load_config(path)


def test_all_bundled_configs_still_load():
    from pathlib import Path
    from fablebot.config import load_config

    for path in Path(__file__).resolve().parents[1].glob("config*.yaml"):
        assert 0 < load_config(path).account.risk_per_trade <= 0.01


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
