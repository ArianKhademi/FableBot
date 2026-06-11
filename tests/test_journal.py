from datetime import date

from fablebot.journal import Journal
from fablebot.portfolio import Trade


def make_trade(**overrides):
    base = dict(
        instrument="BTC",
        direction="long",
        units=0.5,
        entry_price=60_000.0,
        exit_price=61_000.0,
        pnl=500.0,
        strategy="momentum_breakout",
        open_reason="breakout",
        close_reason="channel exit",
        opened_at="2026-06-10T11:00:00+00:00",
        risk_amount=1000.0,
    )
    base.update(overrides)
    return Trade(**base)


def test_in_memory_round_trip():
    journal = Journal()
    journal.record_decision(
        instrument="SPX",
        action="open_long",
        reason="oversold stretch",
        price=7300.0,
        atr=28.0,
        context={"zscore": -2.39},
        position_direction=None,
        equity=100_000.0,
    )
    journal.record_trade(make_trade())
    entries = journal.read()
    assert [e["event"] for e in entries] == ["decision", "trade_closed"]
    assert entries[0]["context"]["zscore"] == -2.39
    assert entries[1]["trade"]["pnl"] == 500.0
    assert entries[1]["shadow"] is False


def test_file_backed_journal_persists(tmp_path):
    path = tmp_path / "journal.jsonl"
    Journal(path).record("decision", instrument="BTC", action="hold")
    Journal(path).record_trade(make_trade(), shadow=True)

    entries = Journal(path).read()
    assert len(entries) == 2
    assert entries[1]["shadow"] is True


def test_read_since_filters_by_date(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.record("decision", instrument="BTC", action="hold")
    assert journal.read(since=date.today()) != []
    assert journal.read(since=date(2099, 1, 1)) == []
