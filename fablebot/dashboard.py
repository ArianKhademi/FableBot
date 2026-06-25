"""Read-only performance dashboard: turn the bot's state into one HTML page.

Pure data helpers (account math, trade stats, equity curve, breakdowns) plus a
self-contained HTML renderer. Nothing here mutates trading state — it only reads
``portfolio.json`` + engine state files and returns a string. ``scripts/dashboard.py``
wires the data in and writes the page; the renderer takes plain dicts so it stays
testable without a broker, a network, or a filesystem.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_SIGN = {"long": 1.0, "short": -1.0}


# ── account math ─────────────────────────────────────────────────────────────

def unrealized_pnl(positions: dict, marks: dict) -> float:
    """Mark-to-market P&L of the open book. ``marks`` maps instrument -> current
    price; a position with no mark contributes 0 (we don't guess)."""
    total = 0.0
    for key, p in positions.items():
        mark = marks.get(key)
        if mark is None:
            continue
        total += (float(mark) - float(p["entry_price"])) * float(p["units"]) * _SIGN.get(p.get("direction"), 0.0)
    return total


def account_summary(starting_equity: float, realized_pnl: float, positions: dict, marks: dict) -> dict:
    """The headline account state. ``realized_equity`` is closed P&L only;
    ``total_equity`` adds the open book's mark-to-market."""
    unreal = unrealized_pnl(positions, marks)
    realized_equity = float(starting_equity) + float(realized_pnl)
    total_equity = realized_equity + unreal
    return {
        "starting_equity": float(starting_equity),
        "realized_pnl": float(realized_pnl),
        "realized_equity": realized_equity,
        "unrealized_pnl": unreal,
        "total_equity": total_equity,
        "return_pct": (total_equity / starting_equity - 1.0) * 100.0 if starting_equity else 0.0,
        "open_count": len(positions),
    }


def _closed(trades: list) -> list:
    return [t for t in trades if isinstance(t.get("pnl"), (int, float))]


def trade_stats(trades: list) -> dict:
    closed = _closed(trades)
    n = len(closed)
    wins = [t for t in closed if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in closed if t["pnl"] < 0)
    losers = n - len(wins)
    return {
        "count": n,
        "wins": len(wins),
        "losses": losers,
        "win_rate": (len(wins) / n * 100.0) if n else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss else (float("inf") if gross_win else 0.0),
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (gross_loss / losers) if losers else 0.0,
    }


def equity_curve(starting_equity: float, trades: list) -> list:
    """Realized-equity path: starting equity, then running total after each
    closed trade in close-time order. (Unrealized swings aren't in here — this is
    the booked curve; the live total is shown separately in the KPIs.)"""
    closed = sorted((t for t in _closed(trades) if t.get("closed_at")), key=lambda t: t["closed_at"])
    if not closed:
        return []
    eq = float(starting_equity)
    pts = [{"t": closed[0].get("opened_at", closed[0]["closed_at"])[:10], "equity": round(eq, 2)}]
    for t in closed:
        eq += float(t["pnl"])
        pts.append({"t": t["closed_at"][:10], "equity": round(eq, 2)})
    return pts


def _group_breakdown(trades: list, field: str) -> list:
    by: dict[str, dict] = {}
    for t in _closed(trades):
        k = t.get(field, "?")
        d = by.setdefault(k, {"key": k, "pnl": 0.0, "n": 0, "wins": 0})
        d["pnl"] += t["pnl"]
        d["n"] += 1
        d["wins"] += 1 if t["pnl"] > 0 else 0
    for d in by.values():
        d["win_rate"] = (d["wins"] / d["n"] * 100.0) if d["n"] else 0.0
        d["pnl"] = round(d["pnl"], 2)
    return sorted(by.values(), key=lambda d: d["pnl"])


def strategy_breakdown(trades: list) -> list:
    return _group_breakdown(trades, "strategy")


def instrument_breakdown(trades: list) -> list:
    return _group_breakdown(trades, "instrument")


def build_snapshot(portfolio: dict, marks: dict) -> dict:
    """Render-ready snapshot from a portfolio dict + current marks (price per
    open instrument). Marks come from the live feed (the CLI fetches them) or the
    engine's ``last_prices`` (in-loop, no fetch) — either way, a position with no
    mark shows no unrealized rather than a guess."""
    positions = portfolio.get("positions", {})
    trades = portfolio.get("trades", [])
    pos_list = []
    for key, p in positions.items():
        mark = marks.get(key)
        unreal = unrealized_pnl({key: p}, marks) if mark is not None else 0.0
        pos_list.append({
            "instrument": p.get("instrument", key), "direction": p.get("direction", "?"),
            "entry_price": float(p.get("entry_price", 0)), "stop_price": float(p.get("stop_price", 0)),
            "units": float(p.get("units", 0)), "mark": mark, "unrealized": unreal,
        })
    recent = sorted((t for t in trades if t.get("closed_at")), key=lambda t: t["closed_at"], reverse=True)[:12]
    return {
        "account": account_summary(
            portfolio.get("starting_equity", 0.0), portfolio.get("realized_pnl", 0.0), positions, marks),
        "stats": trade_stats(trades),
        "equity_curve": equity_curve(portfolio.get("starting_equity", 0.0), trades),
        "by_strategy": strategy_breakdown(trades),
        "by_instrument": instrument_breakdown(trades),
        "positions": pos_list,
        "recent_trades": recent,
    }


# ── engine-side state (the unified view) ─────────────────────────────────────

_PARKED_MIN = 45.0  # population file untouched this long => not worked by the current run


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def read_engine_state(state_dir: Path, now: float | None = None) -> dict:
    """Genome-search progress, proposal-pipeline counts, and cron heartbeats —
    read-only, mirrors scripts/engine_status.py so the dashboard is one view."""
    state_dir = Path(state_dir)
    now = time.time() if now is None else now
    genomes = []
    for p in sorted(state_dir.glob("population_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        name = p.name[len("population_"):-len(".json")]
        hist = d.get("history", [])
        fits = [h.get("fitness") for h in hist if isinstance(h.get("fitness"), (int, float))]
        scored = set(d.get("scores", {}).keys())
        pop = d.get("population", [])
        # population id == hash; recompute cheaply via stored ids if present, else skip saturation
        sat = None
        try:
            from fablebot.dsl import genome_id
            sat = sum(1 for g in pop if genome_id(g) in scored) / max(len(pop), 1)
        except Exception:
            pass
        age_min = (now - p.stat().st_mtime) / 60.0
        if age_min > _PARKED_MIN:
            label = "parked"
        elif sat is not None and sat >= 0.9:
            label = "spinning"
        elif sat is not None and sat >= 0.5:
            label = "converging"
        else:
            label = "exploring"
        genomes.append({
            "name": name,
            "generation": d.get("generation", 0),
            "best": round(max(fits), 2) if fits else None,
            "label": label,
        })

    def _status_counts(events: list) -> dict:
        latest: dict[str, dict] = {}
        for e in events:
            pid = e.get("proposal_id")
            if pid is not None:
                latest.setdefault(pid, {}).update(e)
        counts: dict[str, int] = {}
        for p in latest.values():
            counts[p.get("status", "?")] = counts.get(p.get("status", "?"), 0) + 1
        return counts

    return {
        "genomes": genomes,
        "numeric_proposals": _status_counts(_load_jsonl(state_dir / "proposals.jsonl")),
        "genome_qualified": len(_load_jsonl(state_dir / "genome_proposals.jsonl")),
        "last_runs": _load_jsonl(state_dir / "run_log.jsonl")[-6:],
    }


# ── HTML rendering ───────────────────────────────────────────────────────────

def _money(v: float) -> str:
    return ("-$" if v < 0 else "$") + f"{abs(v):,.0f}"


def render_html(snapshot: dict, engine: dict, generated_at: str, refresh_secs: int = 150,
                title: str = "FableBot") -> str:
    """Assemble the self-contained dashboard page. ``snapshot`` carries the
    account summary, marked positions, trade stats, equity curve, and breakdowns;
    ``engine`` is read_engine_state's output. Returns a full HTML document — a
    file you open in a browser, so it's a complete doc (not a chat fragment)."""
    acct = snapshot["account"]
    cls = lambda v: "pos" if v > 0 else ("neg" if v < 0 else "")
    pf = snapshot["stats"]["profit_factor"]
    pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"

    kpis = [
        ("Account equity", _money(acct["total_equity"]), cls(acct["total_equity"] - acct["starting_equity"])),
        ("Total return", f"{acct['return_pct']:+.1f}%", cls(acct["return_pct"])),
        ("Realized P&L", _money(acct["realized_pnl"]), cls(acct["realized_pnl"])),
        ("Open (unrealized)", _money(acct["unrealized_pnl"]), cls(acct["unrealized_pnl"])),
        ("Win rate", f"{snapshot['stats']['win_rate']:.0f}% · {snapshot['stats']['wins']}/{snapshot['stats']['count']}", ""),
        ("Open positions", str(acct["open_count"]), ""),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="l">{lbl}</div><div class="v {c}">{val}</div></div>'
        for lbl, val, c in kpis
    )

    pos_rows = "".join(
        f'<tr><td>{p["instrument"]}</td><td><span class="tag">{p["direction"]}</span></td>'
        f'<td class="r">{p["entry_price"]:,.2f}</td>'
        f'<td class="r">{("%.2f" % p["mark"]) if p.get("mark") is not None else "—"}</td>'
        f'<td class="r">{p["stop_price"]:,.2f}</td>'
        f'<td class="r {cls(p.get("unrealized", 0))}">{_money(p.get("unrealized", 0)) if p.get("mark") is not None else "—"}</td></tr>'
        for p in snapshot["positions"]
    ) or '<tr><td colspan="6" class="muted">flat — no open positions</td></tr>'

    trade_rows = "".join(
        f'<tr><td>{t.get("closed_at","")[:10]}</td><td>{t.get("instrument","?")}</td>'
        f'<td>{t.get("strategy","?")}</td><td class="muted">{t.get("close_reason","")[:42]}</td>'
        f'<td class="r {cls(t.get("pnl",0))}">{_money(t.get("pnl",0))}</td></tr>'
        for t in snapshot["recent_trades"]
    ) or '<tr><td colspan="5" class="muted">no closed trades yet</td></tr>'

    strat_rows = "".join(
        f'<tr><td>{d["key"]}</td><td class="r">{d["n"]}</td><td class="r">{d["win_rate"]:.0f}%</td>'
        f'<td class="r {cls(d["pnl"])}">{_money(d["pnl"])}</td></tr>'
        for d in snapshot["by_strategy"]
    ) or '<tr><td colspan="4" class="muted">—</td></tr>'

    gen_rows = "".join(
        f'<tr><td>{g["name"]}</td><td class="r">{g["generation"]:,}</td>'
        f'<td class="r">{g["best"] if g["best"] is not None else "—"}</td>'
        f'<td><span class="tag t-{g["label"]}">{g["label"]}</span></td></tr>'
        for g in engine.get("genomes", [])
    ) or '<tr><td colspan="4" class="muted">no populations yet</td></tr>'

    np = engine.get("numeric_proposals", {})
    runs_html = "".join(
        f'<div class="run"><span class="muted">{r.get("ts","?")[:16].replace("T"," ")}</span> '
        f'<b>{r.get("job","?")}</b> {r.get("summary","")}</div>'
        for r in reversed(engine.get("last_runs", []))
    ) or '<div class="muted">no scheduled runs recorded yet</div>'

    data = json.dumps({
        "equity": snapshot["equity_curve"],
        "strat": [{"k": d["key"], "pnl": d["pnl"]} for d in snapshot["by_strategy"]],
    })

    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<meta http-equiv=\"refresh\" content=\"{int(refresh_secs)}\">"
        f"<title>{title} dashboard</title>"
        "<style>" + _CSS + "</style></head><body><div class=\"wrap\">"
        f"<header><h1>{title}</h1><span class=\"muted\">updated {generated_at} · auto-refresh {int(refresh_secs)}s · paper trading</span></header>"
        f"<section class=\"kpis\">{kpi_html}</section>"
        "<h2>Equity curve</h2><div class=\"chartbox\"><canvas id=\"eq\"></canvas></div>"
        "<div class=\"cols\">"
        "<div><h2>Open positions</h2><table><thead><tr><th>Inst</th><th>Side</th><th class=\"r\">Entry</th>"
        f"<th class=\"r\">Mark</th><th class=\"r\">Stop</th><th class=\"r\">Open P&L</th></tr></thead><tbody>{pos_rows}</tbody></table></div>"
        "<div><h2>By strategy</h2><table><thead><tr><th>Strategy</th><th class=\"r\">Trades</th>"
        f"<th class=\"r\">Win%</th><th class=\"r\">Realized</th></tr></thead><tbody>{strat_rows}</tbody></table></div>"
        "</div>"
        f"<h2>Recent closed trades <span class=\"muted\">· {snapshot['stats']['count']} total · PF {pf_str}</span></h2>"
        "<table><thead><tr><th>Closed</th><th>Inst</th><th>Strategy</th><th>Reason</th>"
        f"<th class=\"r\">P&L</th></tr></thead><tbody>{trade_rows}</tbody></table>"
        "<h2>Engine &amp; search</h2>"
        f"<div class=\"muted pill\">numeric: {np.get('qualified',0)} qualified · {np.get('candidate',0)} candidate · {np.get('rejected',0)} rejected · {np.get('promoted',0)} promoted &nbsp;|&nbsp; evolved genomes qualified: {engine.get('genome_qualified',0)}</div>"
        "<table><thead><tr><th>Genome pop</th><th class=\"r\">Gen</th><th class=\"r\">Best fitness</th><th>State</th></tr></thead>"
        f"<tbody>{gen_rows}</tbody></table>"
        f"<h2>Last scheduled runs</h2>{runs_html}"
        "<footer class=\"muted\">Read-only view. Nothing here is live until you (or the nightly reviewer) edit config.yaml.</footer>"
        "</div>"
        "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js\"></script>"
        f"<script>const D={data};</script><script>{_JS}</script>"
        "</body></html>"
    )


_CSS = """
:root{--bg:#fbfaf7;--card:#fff;--ink:#1c1c1a;--mut:#6b6a64;--line:#e7e5dd;--pos:#1d8a5e;--neg:#cf3d3d}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--card:#232320;--ink:#ecece6;--mut:#9a988f;--line:#36352f;--pos:#4cc28a;--neg:#e86a6a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px}
header{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:20px}
h1{font-size:22px;font-weight:600;margin:0}h2{font-size:16px;font-weight:600;margin:28px 0 10px}
.muted{color:var(--mut);font-size:13px;font-weight:400}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .l{color:var(--mut);font-size:13px;margin-bottom:6px}.kpi .v{font-size:23px;font-weight:600}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:640px){.cols{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--mut);font-weight:500;padding:7px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid var(--line)}.r{text-align:right}
.tag{font-size:11px;padding:2px 8px;border-radius:6px;background:var(--bg);border:1px solid var(--line);color:var(--mut)}
.t-parked{color:var(--mut)}.t-spinning{color:var(--neg)}.t-exploring{color:var(--pos)}
.chartbox{position:relative;height:240px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.pill{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin-bottom:10px}
.run{font-size:13px;padding:3px 0}footer{margin-top:28px;padding-top:14px;border-top:1px solid var(--line)}
"""

_JS = """
const dark=matchMedia('(prefers-color-scheme:dark)').matches;
const ink=dark?'#9a988f':'#6b6a64', grid=dark?'#36352f':'#e7e5dd';
new Chart(document.getElementById('eq'),{type:'line',
 data:{labels:D.equity.map(p=>p.t),datasets:[{data:D.equity.map(p=>p.equity),
  borderColor:'#378ADD',backgroundColor:'rgba(55,138,221,0.08)',fill:true,tension:0.2,pointRadius:0,borderWidth:2}]},
 options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
  scales:{x:{ticks:{color:ink,maxTicksLimit:8},grid:{display:false}},
   y:{ticks:{color:ink,callback:v=>'$'+(v/1000).toFixed(0)+'k'},grid:{color:grid}}}}});
"""
