"""Performance analytics over the decision journal.

This is the deterministic half of the learning loop: it turns the raw
journal into the numbers a reviewer (human or Claude) needs to decide what
to change — expectancy per strategy, stop quality, re-entries after
stop-outs, and the counterfactual P&L of trades the correlation filter
blocked.

It never changes anything. It only measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from .journal import Journal

# A losing trade is "near-miss" if it was up at least this fraction of its
# risk budget before dying — evidence the entry was fine but the exit wasn't.
NEAR_MISS_MFE_FRACTION = 0.5

# A re-entry is a trade opened on the same instrument within this window
# after a stop-out — the "revenge trade" / whipsaw pattern to watch.
REENTRY_WINDOW_HOURS = 24

# Below this many closed trades, per-bucket stats are flagged as noise.
MIN_SAMPLE = 20


@dataclass
class TradeStats:
    count: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0

    @property
    def losses(self) -> int:
        return self.count - self.wins

    @property
    def win_rate(self) -> float:
        return self.wins / self.count if self.count else 0.0

    @property
    def expectancy(self) -> float:
        return self.total_pnl / self.count if self.count else 0.0

    @property
    def profit_factor(self) -> Optional[float]:
        return self.gross_win / self.gross_loss if self.gross_loss else None

    def add(self, pnl: float) -> None:
        self.count += 1
        if pnl > 0:
            self.wins += 1
            self.gross_win += pnl
        else:
            self.gross_loss += -pnl
        self.total_pnl += pnl


@dataclass
class Digest:
    period_start: Optional[date]
    period_end: Optional[date]
    decisions: int
    blocked: int
    overall: TradeStats
    by_instrument: dict[str, TradeStats]
    by_strategy: dict[str, TradeStats]
    stop_outs: int
    near_miss_stop_outs: list[dict] = field(default_factory=list)
    reentries: list[dict] = field(default_factory=list)
    shadow: TradeStats = field(default_factory=TradeStats)


def _closed_trades(entries: list[dict], shadow: bool) -> list[dict]:
    return [
        e["trade"]
        for e in entries
        if e["event"] == "trade_closed" and e.get("shadow", False) is shadow
    ]


def _is_stop(trade: dict) -> bool:
    return "hard stop" in trade.get("close_reason", "")


def stop_recovery(direction: str, entry_price: float, post_bars, horizon: int) -> dict:
    """Would a stopped-out trade have come back? Given its direction + entry and
    the price bars AFTER it was stopped (an iterable of (high, low) tuples), look
    ``horizon`` bars ahead and report:

      - ``recovered``: did price return to the entry (break-even) within horizon —
        i.e. the stop locked in a loss the trade would have erased?
      - ``bars_to_recover``: how many bars that took.
      - ``best_favorable_per_unit``: the largest in-your-favor move after the stop
        (per unit) — how much the premature exit left on the table.

    Break-even (not a profit target) is the bar deliberately: clearing it already
    proves the loss was avoidable, and it's strategy-agnostic, so this works for
    mean-reversion, momentum, and trend alike."""
    recovered, bars_to_recover, best = False, None, 0.0
    for i, (high, low) in enumerate(list(post_bars)[:horizon]):
        if direction == "long":
            best = max(best, high - entry_price)
            if not recovered and high >= entry_price:
                recovered, bars_to_recover = True, i + 1
        else:  # short
            best = max(best, entry_price - low)
            if not recovered and low <= entry_price:
                recovered, bars_to_recover = True, i + 1
    return {"recovered": recovered, "bars_to_recover": bars_to_recover,
            "best_favorable_per_unit": best}


def analyze(entries: list[dict]) -> Digest:
    decisions = [e for e in entries if e["event"] == "decision"]
    trades = _closed_trades(entries, shadow=False)
    shadow_trades = _closed_trades(entries, shadow=True)

    overall = TradeStats()
    by_instrument: dict[str, TradeStats] = {}
    by_strategy: dict[str, TradeStats] = {}
    for t in trades:
        overall.add(t["pnl"])
        by_instrument.setdefault(t["instrument"], TradeStats()).add(t["pnl"])
        by_strategy.setdefault(t["strategy"], TradeStats()).add(t["pnl"])

    shadow_stats = TradeStats()
    for t in shadow_trades:
        shadow_stats.add(t["pnl"])

    stops = [t for t in trades if _is_stop(t)]
    near_misses = [
        t
        for t in stops
        if t.get("risk_amount", 0) > 0
        and t.get("max_favorable_pnl", 0.0) >= NEAR_MISS_MFE_FRACTION * t["risk_amount"]
    ]

    reentries = _find_reentries(trades)

    dates = sorted(datetime.fromisoformat(e["ts"]).date() for e in entries) if entries else []
    return Digest(
        period_start=dates[0] if dates else None,
        period_end=dates[-1] if dates else None,
        decisions=len(decisions),
        blocked=sum(1 for d in decisions if d.get("action") == "blocked"),
        overall=overall,
        by_instrument=by_instrument,
        by_strategy=by_strategy,
        stop_outs=len(stops),
        near_miss_stop_outs=near_misses,
        reentries=reentries,
        shadow=shadow_stats,
    )


def _find_reentries(trades: list[dict]) -> list[dict]:
    """Trades opened on an instrument within REENTRY_WINDOW_HOURS of a
    stop-out on that same instrument — with their outcomes, so the reviewer
    can judge whether re-entering after a stop pays or compounds the loss."""
    window = timedelta(hours=REENTRY_WINDOW_HOURS)
    stops_by_instrument: dict[str, list[datetime]] = {}
    for t in trades:
        if _is_stop(t):
            stops_by_instrument.setdefault(t["instrument"], []).append(
                datetime.fromisoformat(t["closed_at"])
            )
    found = []
    for t in trades:
        opened = datetime.fromisoformat(t["opened_at"])
        for stop_time in stops_by_instrument.get(t["instrument"], []):
            if stop_time < opened <= stop_time + window:
                found.append(
                    {
                        "instrument": t["instrument"],
                        "hours_after_stop": (opened - stop_time).total_seconds() / 3600.0,
                        "pnl": t["pnl"],
                        "open_reason": t.get("open_reason", ""),
                    }
                )
                break
    return found


# -- rendering ----------------------------------------------------------------


def _stats_line(stats: TradeStats) -> str:
    pf = f"{stats.profit_factor:.2f}" if stats.profit_factor is not None else "inf"
    noise = "" if stats.count >= MIN_SAMPLE else f" — SMALL SAMPLE (<{MIN_SAMPLE}), treat as noise"
    return (
        f"{stats.count} trades, {stats.win_rate:.0%} win rate, "
        f"P&L {stats.total_pnl:+,.2f}, expectancy {stats.expectancy:+,.2f}/trade, "
        f"profit factor {pf}{noise}"
    )


def render_digest(digest: Digest) -> str:
    lines = ["# FableBot performance digest", ""]
    if digest.period_start:
        lines.append(f"Period: {digest.period_start} to {digest.period_end}")
    lines += [
        f"Decisions journaled: {digest.decisions} ({digest.blocked} blocked by correlation filter)",
        "",
        "## Overall",
        _stats_line(digest.overall),
        "",
        "## By strategy",
    ]
    for name, stats in sorted(digest.by_strategy.items()):
        lines.append(f"- {name}: {_stats_line(stats)}")
    lines += ["", "## By instrument"]
    for name, stats in sorted(digest.by_instrument.items()):
        lines.append(f"- {name}: {_stats_line(stats)}")

    lines += ["", "## Stop-out quality"]
    lines.append(f"Stop-outs: {digest.stop_outs} of {digest.overall.count} trades")
    if digest.near_miss_stop_outs:
        lines.append(
            f"Near-miss stop-outs (were up >={NEAR_MISS_MFE_FRACTION:.0%} of risk "
            f"before dying): {len(digest.near_miss_stop_outs)}"
        )
        for t in digest.near_miss_stop_outs:
            lines.append(
                f"- {t['instrument']} {t['direction']}: peaked {t['max_favorable_pnl']:+,.2f} "
                f"then stopped for {t['pnl']:+,.2f}"
            )
    else:
        lines.append("No near-miss stop-outs — stopped trades never got meaningfully green.")

    lines += ["", f"## Re-entries within {REENTRY_WINDOW_HOURS}h of a stop-out"]
    if digest.reentries:
        re_pnl = sum(r["pnl"] for r in digest.reentries)
        lines.append(f"{len(digest.reentries)} re-entries, net P&L {re_pnl:+,.2f}")
        for r in digest.reentries:
            lines.append(
                f"- {r['instrument']}: {r['hours_after_stop']:.1f}h after stop, "
                f"P&L {r['pnl']:+,.2f} ({r['open_reason']})"
            )
    else:
        lines.append("None.")

    lines += ["", "## Correlation filter counterfactual (shadow book)"]
    if digest.shadow.count:
        lines.append(
            f"Blocked trades, had they run: {_stats_line(digest.shadow)}. "
            f"Positive means the filter cost money this period; negative means it saved it. "
            f"Judge on large samples only — the filter exists to cap correlated tail risk, "
            f"which rarely shows up in small windows."
        )
    else:
        lines.append("No blocked trades resolved this period.")
    return "\n".join(lines)


def digest_from_journal(journal: Journal, days: Optional[int] = None) -> Digest:
    since = date.today() - timedelta(days=days) if days else None
    return analyze(journal.read(since=since))
