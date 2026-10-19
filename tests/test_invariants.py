"""The risk constitution (RISK_RULES.md) as executable tests.

The learning loop may rewrite strategies, but every proposal must keep this
file passing. If a change breaks one of these, it is not a tuning decision
— it is a constitutional change that needs an explicit human yes.
"""

from pathlib import Path

import pytest
import yaml

from fablebot.portfolio import PaperBroker, Position
from fablebot.risk import size_position

ROOT = Path(__file__).resolve().parent.parent


def test_constitution_document_exists():
    assert (ROOT / "RISK_RULES.md").exists()


def test_configured_risk_budget_never_exceeds_one_percent():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert config["account"]["risk_per_trade"] <= 0.01


@pytest.mark.parametrize(
    "equity,price,atr,mult",
    [
        (100_000, 7_300, 28.0, 2.0),  # index, quiet
        (100_000, 62_000, 520.0, 2.5),  # bitcoin, lively
        (50_000, 4_150, 46.0, 3.0),  # gold, smaller account
        (250_000, 89.7, 1.9, 3.0),  # oil, bigger account
    ],
)
def test_sizing_always_risks_exactly_the_budget(equity, price, atr, mult):
    for direction in ("long", "short"):
        sized = size_position(equity, 0.01, price, atr, mult, direction)
        loss_at_stop = sized.units * abs(price - sized.stop_price)
        assert loss_at_stop == pytest.approx(equity * 0.01)


def test_every_position_carries_a_stop_on_the_losing_side():
    long = size_position(100_000, 0.01, 100.0, 2.0, 2.0, "long")
    short = size_position(100_000, 0.01, 100.0, 2.0, 2.0, "short")
    assert long.stop_price < 100.0
    assert short.stop_price > 100.0


def test_gap_protection_cannot_be_outrun():
    """Even if price never prints the stop level, a loss reaching the risk
    budget is force-closed. The realized loss can still exceed that budget."""
    broker = PaperBroker(100_000.0)
    broker.open_position(
        Position(
            instrument="BTC",
            direction="long",
            units=0.5,
            entry_price=60_000.0,
            stop_price=1.0,  # absurdly far stop: only gap protection remains
            risk_amount=1000.0,
            strategy="any",
            reason="test",
        )
    )
    trade = broker.enforce_stops("BTC", high=58_100.0, low=57_900.0, close=58_000.0)
    assert trade is not None
    assert trade.pnl <= -1000.0


def test_no_pyramiding_one_position_per_instrument():
    broker = PaperBroker(100_000.0)
    pos = Position(
        instrument="BTC",
        direction="long",
        units=0.5,
        entry_price=60_000.0,
        stop_price=58_000.0,
        risk_amount=1000.0,
        strategy="any",
        reason="test",
    )
    broker.open_position(pos)
    with pytest.raises(ValueError):
        broker.open_position(pos)
