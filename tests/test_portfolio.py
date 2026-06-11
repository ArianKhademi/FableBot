import pytest

from fablebot.portfolio import PaperBroker, Position


def make_position(**overrides):
    base = dict(
        instrument="BTC",
        direction="long",
        units=0.5,
        entry_price=60_000.0,
        stop_price=58_000.0,
        risk_amount=1000.0,
        strategy="momentum_breakout",
        reason="test",
    )
    base.update(overrides)
    return Position(**base)


def test_open_close_round_trip():
    broker = PaperBroker(100_000.0)
    broker.open_position(make_position())
    trade = broker.close_position("BTC", 62_000.0, "take profit")
    assert trade.pnl == pytest.approx(1000.0)
    assert broker.equity == pytest.approx(101_000.0)
    assert not broker.positions


def test_short_pnl_sign():
    broker = PaperBroker(100_000.0)
    broker.open_position(make_position(direction="short", stop_price=62_000.0))
    trade = broker.close_position("BTC", 58_000.0, "cover")
    assert trade.pnl == pytest.approx(1000.0)


def test_hard_stop_triggers_on_bar_low():
    broker = PaperBroker(100_000.0)
    broker.open_position(make_position())
    trade = broker.enforce_stops("BTC", high=60_500.0, low=57_900.0, close=58_500.0)
    assert trade is not None
    assert trade.exit_price == pytest.approx(58_000.0)  # filled at the stop
    assert trade.pnl == pytest.approx(-1000.0)  # exactly the 1% risk budget
    assert "hard stop" in trade.close_reason


def test_gap_through_stop_still_cut():
    broker = PaperBroker(100_000.0)
    pos = make_position()
    broker.open_position(pos)
    # A close below entry deep enough that the loss exceeds the budget, with
    # an intrabar low that never printed the stop level... still gets cut.
    pos.stop_price = 0.0  # disable the price stop to isolate the budget check
    trade = broker.enforce_stops("BTC", high=58_100.0, low=57_500.0, close=57_500.0)
    assert trade is not None
    assert "gap" in trade.close_reason
    assert trade.pnl <= -1000.0


def test_no_stop_when_safe():
    broker = PaperBroker(100_000.0)
    broker.open_position(make_position())
    assert broker.enforce_stops("BTC", high=61_000.0, low=59_000.0, close=60_500.0) is None
    assert "BTC" in broker.positions


def test_excursions_tracked_and_copied_to_trade():
    broker = PaperBroker(100_000.0)
    broker.open_position(make_position())  # long 0.5 @ 60,000
    broker.update_excursions("BTC", high=63_000.0, low=59_000.0)
    trade = broker.close_position("BTC", 60_500.0, "exit")
    assert trade.max_favorable_pnl == pytest.approx(1500.0)
    assert trade.max_adverse_pnl == pytest.approx(-500.0)
    assert trade.risk_amount == pytest.approx(1000.0)


def test_state_persistence_round_trip(tmp_path):
    state = tmp_path / "portfolio.json"
    broker = PaperBroker(100_000.0, state_path=state)
    broker.open_position(make_position())
    broker.close_position("BTC", 62_000.0, "take profit")
    broker.open_position(make_position(instrument="GOLD", entry_price=2000.0, stop_price=1976.0))

    reloaded = PaperBroker(0.0, state_path=state)
    assert reloaded.starting_equity == pytest.approx(100_000.0)
    assert reloaded.realized_pnl == pytest.approx(1000.0)
    assert set(reloaded.positions) == {"GOLD"}
    assert len(reloaded.trades) == 1
