"""Daily performance digest — pure renderer + weekly P&L windowing.

No broker, network, or filesystem: feed dicts, assert on the Markdown / numbers.
"""

from datetime import datetime

from fablebot import dashboard, reporting


def _book(name, starting, realized, trades, marks=None):
    snap = dashboard.build_snapshot(
        {"starting_equity": starting, "realized_pnl": realized, "positions": {}, "trades": trades},
        marks or {},
    )
    return {
        "name": name,
        "account": snap["account"],
        "stats": snap["stats"],
        "by_instrument": snap["by_instrument"],
        "by_strategy": snap["by_strategy"],
        "weekly": reporting.weekly_pnl(trades, datetime(2026, 6, 25, 12, 0)),
    }


def test_weekly_pnl_splits_recent_from_prior_and_finds_top():
    now = datetime(2026, 6, 25, 12, 0)
    trades = [
        {"pnl": 100.0, "closed_at": "2026-06-24T10:00:00", "instrument": "SPY"},  # last 7d
        {"pnl": -30.0, "closed_at": "2026-06-23T10:00:00", "instrument": "QQQ"},  # last 7d
        {"pnl": 40.0, "closed_at": "2026-06-20T10:00:00", "instrument": "SPY"},   # last 7d
        {"pnl": 40.0, "closed_at": "2026-06-15T10:00:00", "instrument": "SPY"},   # prior 7d
        {"pnl": 999.0, "closed_at": "2026-06-01T10:00:00", "instrument": "SPY"},  # older — ignored
    ]
    w = reporting.weekly_pnl(trades, now)
    assert round(w["last"], 2) == 110.0      # 100 - 30 + 40
    assert w["trades"] == 3
    assert w["top"] == ("SPY", 140.0)        # 100 + 40
    assert round(w["prev"], 2) == 40.0       # only the 06-15 trade


def test_weekly_pnl_empty():
    w = reporting.weekly_pnl([], datetime(2026, 6, 25, 12, 0))
    assert w == {"last": 0.0, "prev": 0.0, "trades": 0, "top": None}


def test_render_report_ranks_and_lists_books():
    winner = _book("Aggressive", 100000.0, 5000.0,
                   [{"pnl": 100.0, "closed_at": "2026-06-24T10:00:00", "instrument": "SPY", "strategy": "mean_reversion"}])
    loser = _book("Conservative", 100000.0, -2000.0,
                  [{"pnl": -50.0, "closed_at": "2026-06-24T10:00:00", "instrument": "QQQ", "strategy": "mean_reversion"}])
    md = reporting.render_report([loser, winner], "2026-06-25 07:30")
    assert md.startswith("# FableBot")
    assert "## Leaderboard" in md and "## By book" in md and "## Observations" in md
    assert "Aggressive" in md and "Conservative" in md
    # Sorted by return: the winner is listed before the laggard.
    assert md.index("Aggressive") < md.index("Conservative")
    assert "Leader:" in md and "Aggressive" in md


def test_render_report_handles_no_books():
    md = reporting.render_report([], "now")
    assert "No books found" in md
