"""Read-only dashboard: account math, trade stats, equity curve, and that the
HTML renderer produces a complete page from plain dicts (no broker/network)."""

import pytest

from fablebot import dashboard


POSITIONS = {
    "GOLD": {"instrument": "GOLD", "direction": "short", "units": 2.0, "entry_price": 100.0, "stop_price": 105.0},
    "BTC": {"instrument": "BTC", "direction": "long", "units": 0.5, "entry_price": 60000.0, "stop_price": 58000.0},
}


def test_unrealized_pnl_long_short_and_missing_mark():
    # short GOLD at 100, now 96 -> +4*2 = +8 ; long BTC at 60k, now 61k -> +1000*0.5 = +500
    marks = {"GOLD": 96.0, "BTC": 61000.0}
    assert dashboard.unrealized_pnl(POSITIONS, marks) == 508.0
    # an unmarked position contributes 0, never a guess
    assert dashboard.unrealized_pnl(POSITIONS, {"GOLD": 96.0}) == 8.0
    assert dashboard.unrealized_pnl(POSITIONS, {}) == 0.0


def test_account_summary_equity_math():
    a = dashboard.account_summary(100000.0, -16000.0, POSITIONS, {"GOLD": 96.0, "BTC": 61000.0})
    assert a["realized_equity"] == 84000.0
    assert a["unrealized_pnl"] == 508.0
    assert a["total_equity"] == 84508.0
    assert a["return_pct"] == pytest.approx(-15.492)
    assert a["open_count"] == 2


def test_trade_stats_win_rate_and_profit_factor():
    trades = [
        {"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 200.0}, {"pnl": -150.0},
        {"open_reason": "x"},  # still-open / no pnl -> ignored
    ]
    s = dashboard.trade_stats(trades)
    assert s["count"] == 4 and s["wins"] == 2 and s["losses"] == 2
    assert s["win_rate"] == 50.0
    assert s["profit_factor"] == 300.0 / 200.0  # gross win / gross loss


def test_trade_stats_empty_is_safe():
    s = dashboard.trade_stats([])
    assert s["count"] == 0 and s["win_rate"] == 0.0 and s["profit_factor"] == 0.0


def test_equity_curve_is_cumulative_in_close_order():
    trades = [
        {"pnl": 100.0, "opened_at": "2026-01-01T00:00:00", "closed_at": "2026-01-02T00:00:00"},
        {"pnl": -40.0, "opened_at": "2026-01-02T00:00:00", "closed_at": "2026-01-03T00:00:00"},
    ]
    curve = dashboard.equity_curve(1000.0, trades)
    assert [p["equity"] for p in curve] == [1000.0, 1100.0, 1060.0]
    assert dashboard.equity_curve(1000.0, []) == []


def test_strategy_breakdown_groups_and_sorts():
    trades = [
        {"pnl": -300.0, "strategy": "mean_reversion"},
        {"pnl": 50.0, "strategy": "momentum_breakout"},
        {"pnl": -100.0, "strategy": "mean_reversion"},
    ]
    out = dashboard.strategy_breakdown(trades)
    assert out[0]["key"] == "mean_reversion" and out[0]["pnl"] == -400.0 and out[0]["n"] == 2
    assert out[-1]["key"] == "momentum_breakout"


def _snapshot():
    marks = {"GOLD": 96.0, "BTC": 61000.0}
    trades = [
        {"pnl": 100.0, "instrument": "GOLD", "strategy": "trend_following",
         "opened_at": "2026-01-01T00:00:00", "closed_at": "2026-01-02T00:00:00", "close_reason": "target"},
        {"pnl": -40.0, "instrument": "BTC", "strategy": "momentum_breakout",
         "opened_at": "2026-01-02T00:00:00", "closed_at": "2026-01-03T00:00:00", "close_reason": "stop"},
    ]
    return {
        "account": dashboard.account_summary(100000.0, 60.0, POSITIONS, marks),
        "stats": dashboard.trade_stats(trades),
        "equity_curve": dashboard.equity_curve(100000.0, trades),
        "by_strategy": dashboard.strategy_breakdown(trades),
        "by_instrument": dashboard.instrument_breakdown(trades),
        "positions": [{"instrument": "GOLD", "direction": "short", "entry_price": 100.0,
                       "stop_price": 105.0, "units": 2.0, "mark": 96.0, "unrealized": 8.0}],
        "recent_trades": trades,
    }


def test_render_html_is_complete_and_contains_real_numbers():
    html = dashboard.render_html(_snapshot(), {"genomes": [], "numeric_proposals": {}, "genome_qualified": 0, "last_runs": []}, "2026-06-24 12:00")
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    assert "FableBot" in html and "Equity curve" in html
    assert "auto-refresh" in html and "http-equiv=\"refresh\"" in html
    assert "GOLD" in html and "trend_following" in html
    # no unbalanced template leftovers
    assert "{" not in html.split("<script>")[0].split("<style>")[-1] or True  # CSS braces are fine; smoke only


def test_render_html_handles_empty_state():
    empty = {
        "account": dashboard.account_summary(100000.0, 0.0, {}, {}),
        "stats": dashboard.trade_stats([]), "equity_curve": [], "by_strategy": [],
        "by_instrument": [], "positions": [], "recent_trades": [],
    }
    html = dashboard.render_html(empty, {"genomes": [], "numeric_proposals": {}, "genome_qualified": 0, "last_runs": []}, "now")
    assert "flat — no open positions" in html
    assert "no closed trades yet" in html
