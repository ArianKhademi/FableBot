"""Markdown performance digest across all paper books — pure renderers.

``scripts/bot_report.py`` assembles per-book data (account, trade stats,
per-instrument/strategy breakdowns, trailing-7-day P&L) and calls
``render_report()`` to get a Markdown string it writes to disk (e.g. the
Desktop) each day. Kept pure — data in, string out — so it's unit-testable
without a broker, a network, or a filesystem, exactly like ``fablebot/dashboard``.

The "observations" are deliberately mechanical: data attribution (who leads, what
drove the week), never forecasts. Nothing here interprets beyond the numbers.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def _money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _pnl(x: float) -> str:
    """Signed money, with an explicit + for gains (P&L cells read at a glance)."""
    if x > 0:
        return f"+${x:,.0f}"
    if x < 0:
        return f"-${abs(x):,.0f}"
    return "$0"


def _parse(ts) -> datetime | None:
    # Trade timestamps are ISO; truncate any tz/offset so comparisons stay naive
    # (Python 3.9's fromisoformat won't take a trailing 'Z').
    try:
        return datetime.fromisoformat(str(ts)[:19])
    except Exception:
        return None


def weekly_pnl(trades: list, now: datetime, days: int = 7) -> dict:
    """Realized P&L from trades closed in the trailing ``days`` vs the prior
    ``days``, the recent trade count, and the top contributing instrument.
    ``now`` is a naive local datetime; trades carry ``closed_at`` (ISO), ``pnl``
    and ``instrument``."""
    cut_recent = now - timedelta(days=days)
    cut_prior = now - timedelta(days=2 * days)
    last = prev = 0.0
    n = 0
    by_sym: dict[str, float] = {}
    for t in trades:
        pnl = t.get("pnl")
        if not isinstance(pnl, (int, float)):
            continue
        dt = _parse(t.get("closed_at"))
        if dt is None:
            continue
        if dt >= cut_recent:
            last += pnl
            n += 1
            sym = t.get("instrument", "?")
            by_sym[sym] = by_sym.get(sym, 0.0) + pnl
        elif dt >= cut_prior:
            prev += pnl
    top = max(by_sym.items(), key=lambda kv: kv[1]) if by_sym else None
    return {"last": last, "prev": prev, "trades": n, "top": top}


def render_report(books: list, generated_at: str) -> str:
    """One Markdown digest across every paper book. ``books`` is a list of dicts:
    ``name``, ``account`` (account_summary), ``stats`` (trade_stats),
    ``by_instrument`` / ``by_strategy`` (breakdowns), ``weekly`` (weekly_pnl)."""
    head = f"# FableBot — Daily Performance Report\n\n_Generated {generated_at} · paper trading · read-only snapshot_\n"
    if not books:
        return head + "\nNo books found yet — start a bot first.\n"

    ranked = sorted(books, key=lambda b: b["account"]["return_pct"], reverse=True)
    out = [head]

    # ── Leaderboard ──────────────────────────────────────────────────────────
    out.append("## Leaderboard\n")
    out.append("| Book | Equity | Return | Realized | Unrealized | Win% | Trades | Last 7d |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for b in ranked:
        a, s, w = b["account"], b["stats"], b["weekly"]
        out.append(
            f"| {b['name']} | {_money(a['total_equity'])} | {a['return_pct']:+.1f}% | "
            f"{_pnl(a['realized_pnl'])} | {_pnl(a['unrealized_pnl'])} | "
            f"{s['win_rate']:.0f}% | {s['count']} | {_pnl(w['last'])} |"
        )
    out.append("")

    # ── Per-book detail ──────────────────────────────────────────────────────
    out.append("## By book\n")
    for b in ranked:
        a, s, w = b["account"], b["stats"], b["weekly"]
        out.append(f"### {b['name']}\n")
        out.append(
            f"- **Equity** {_money(a['total_equity'])} ({a['return_pct']:+.1f}%) · "
            f"realized {_pnl(a['realized_pnl'])} · unrealized {_pnl(a['unrealized_pnl'])} · {a['open_count']} open"
        )
        if s["count"]:
            pf = s["profit_factor"]
            pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
            out.append(f"- **Record** {s['count']} closed · {s['win_rate']:.0f}% win · profit factor {pf_str}")
        else:
            out.append("- **Record** no closed trades yet")
        wk = f"- **Last 7d** {_pnl(w['last'])} over {w['trades']} trade(s) · prior 7d {_pnl(w['prev'])}"
        if w.get("top"):
            sym, p = w["top"]
            wk += f" — led by {sym} ({_pnl(p)})"
        out.append(wk)
        bi = b.get("by_instrument") or []
        if bi:
            best, worst = bi[-1], bi[0]  # instrument_breakdown sorts ascending by pnl
            line = f"- **Instruments** best {best['key']} {_pnl(best['pnl'])} ({best['n']} tr, {best['win_rate']:.0f}% win)"
            if worst["key"] != best["key"]:
                line += f" · worst {worst['key']} {_pnl(worst['pnl'])} ({worst['n']} tr)"
            out.append(line)
        bs = b.get("by_strategy") or []
        if len(bs) > 1:
            parts = ", ".join(f"{d['key']} {_pnl(d['pnl'])}" for d in sorted(bs, key=lambda d: d["pnl"], reverse=True))
            out.append(f"- **By strategy** {parts}")
        out.append("")

    # ── Observations (mechanical) ────────────────────────────────────────────
    out.append("## Observations\n")
    leader, laggard = ranked[0], ranked[-1]
    out.append(
        f"- **Leader:** {leader['name']} at {leader['account']['return_pct']:+.1f}% "
        f"({_pnl(leader['account']['total_equity'] - leader['account']['starting_equity'])} total)."
    )
    if len(ranked) > 1:
        out.append(f"- **Laggard:** {laggard['name']} at {laggard['account']['return_pct']:+.1f}%.")
    mover = max(books, key=lambda b: abs(b["weekly"]["last"]))
    if mover["weekly"]["last"]:
        drv = ""
        if mover["weekly"].get("top"):
            drv = f", led by {mover['weekly']['top'][0]} ({_pnl(mover['weekly']['top'][1])})"
        out.append(f"- **Biggest 7-day move:** {mover['name']} {_pnl(mover['weekly']['last'])}{drv}.")
    agg: dict[str, dict] = {}
    for b in books:
        for d in (b.get("by_instrument") or []):
            e = agg.setdefault(d["key"], {"pnl": 0.0, "n": 0})
            e["pnl"] += d["pnl"]
            e["n"] += d["n"]
    if agg:
        best_sym = max(agg.items(), key=lambda kv: kv[1]["pnl"])
        worst_sym = min(agg.items(), key=lambda kv: kv[1]["pnl"])
        out.append(
            f"- **Across all books:** best instrument {best_sym[0]} ({_pnl(best_sym[1]['pnl'])}, "
            f"{best_sym[1]['n']} tr); worst {worst_sym[0]} ({_pnl(worst_sym[1]['pnl'])}, {worst_sym[1]['n']} tr)."
        )
    out.append("")
    out.append("---\n")
    out.append(
        "_Paper trading. Equity is per-book book value; “Last 7d” is realized P&L from trades closed in the "
        "trailing 7 days. Observations are mechanical data attribution, not forecasts. "
        "Generated by `scripts/bot_report.py`._\n"
    )
    return "\n".join(out)
