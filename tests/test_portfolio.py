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


# -- honest gap fills + trail labeling + tolerant load ------------------------

def test_gap_through_stop_fills_at_the_open_not_the_stop():
    """When price gaps past the stop, the fill is the bar's open (the first
    price that actually traded), so the booked loss exceeds the budget honestly
    instead of pretending the stop level traded."""
    from fablebot.portfolio import PaperBroker, Position
    broker = PaperBroker(100_000.0)
    broker.open_position(Position(
        instrument="X", direction="long", units=100.0, entry_price=100.0,
        stop_price=96.0, risk_amount=400.0, strategy="t", reason="t"))
    trade = broker.enforce_stops("X", high=91.0, low=89.0, close=90.0, open_=90.0)
    assert trade is not None
    assert trade.exit_price == 90.0                       # the open, not 96
    assert trade.pnl == -1000.0                           # honest gap loss
    assert "gapped, filled 90.00" in trade.close_reason


def test_normal_stop_still_fills_at_stop_price():
    from fablebot.portfolio import PaperBroker, Position
    broker = PaperBroker(100_000.0)
    broker.open_position(Position(
        instrument="X", direction="long", units=100.0, entry_price=100.0,
        stop_price=96.0, risk_amount=400.0, strategy="t", reason="t"))
    trade = broker.enforce_stops("X", high=99.0, low=95.5, close=97.0, open_=99.0)
    assert trade.exit_price == 96.0 and trade.pnl == -400.0
    assert trade.close_reason.startswith("hard stop hit")


def test_trailed_stop_exit_is_labeled_trail_stop():
    from fablebot.portfolio import PaperBroker, Position
    broker = PaperBroker(100_000.0)
    broker.open_position(Position(
        instrument="X", direction="long", units=100.0, entry_price=100.0,
        stop_price=96.0, risk_amount=400.0, strategy="t", reason="t"))
    assert broker.positions["X"].tighten_to(101.0)        # trail above entry
    trade = broker.enforce_stops("X", high=102.0, low=100.5, close=101.5, open_=102.0)
    assert trade.pnl > 0 and trade.close_reason.startswith("trail stop hit")


def test_stop_fill_pays_friction_when_configured():
    from fablebot.portfolio import PaperBroker, Position
    broker = PaperBroker(100_000.0, friction_bps=10.0)    # 10 bps for visibility
    broker.open_position(Position(
        instrument="X", direction="long", units=100.0, entry_price=100.0,
        stop_price=96.0, risk_amount=400.0, strategy="t", reason="t"))
    trade = broker.enforce_stops("X", high=97.0, low=95.0, close=96.5, open_=97.0)
    assert trade.exit_price == pytest.approx(96.0 * (1 - 0.001))   # adverse slip


def test_load_tolerates_unknown_fields(tmp_path):
    """A field from newer code (or a removed field) in portfolio.json must not
    brick the book on rollback."""
    import json as _json
    from fablebot.portfolio import PaperBroker
    state = {
        "starting_equity": 100000.0, "realized_pnl": 50.0,
        "positions": {"X": {"instrument": "X", "direction": "long", "units": 1.0,
                            "entry_price": 100.0, "stop_price": 96.0, "risk_amount": 4.0,
                            "strategy": "t", "reason": "t", "opened_at": "2026-01-01T00:00:00+00:00",
                            "max_favorable_pnl": 0.0, "max_adverse_pnl": 0.0,
                            "from_the_future": 42}},
        "trades": [{"instrument": "X", "direction": "long", "units": 1.0, "entry_price": 90.0,
                    "exit_price": 95.0, "pnl": 5.0, "strategy": "t", "open_reason": "o",
                    "close_reason": "c", "opened_at": "2026-01-01T00:00:00+00:00",
                    "closed_at": "2026-01-01T01:00:00+00:00", "mystery_field": True}],
    }
    p = tmp_path / "portfolio.json"
    p.write_text(_json.dumps(state))
    broker = PaperBroker(1.0, state_path=p)
    assert broker.positions["X"].units == 1.0 and broker.trades[0].pnl == 5.0
