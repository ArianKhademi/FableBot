"""Dynamic universe rotation — pick which instruments the live book trades.

Instead of trading a fixed five forever, a daily scan scores the whole universe
(plus the current book) by the best gated edge it can find on each symbol, and
proposes a rotation: add the strongest newcomers, drop stale holdings. It is
*evidence*, not an automatic live swap — the change is human-gated through the
reviewer, exactly like every other proposal.

Guardrails (in :func:`propose_rotation`):
  - **Never force-close an open position to rotate.** Instruments with a live
    position are sticky — kept regardless of score until they close naturally.
  - **Cap concurrency.** At most ``max_active`` instruments trade at once
    (open positions may carry the book slightly past the cap, by design).
  - **Hysteresis.** A challenger must beat a current holding by ``churn_margin``
    to replace it, so the book doesn't thrash on noise.
  - **Quality floor.** A newcomer must clear ``min_score`` to be added at all.
  - **Track-record tilt.** The score above is backtest *tradeability*; what the
    bot has *actually* earned matters too. :func:`performance_tilt` nudges each
    instrument's effective score up when its realized trades have been profitable
    and down when they've been losing — so a working book isn't dropped for a
    flashy backtest, and a persistently-losing one is demoted until it earns its
    way back.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def score_symbol(symbol: str, timeframe: str, df, friction: float = 0.0, min_trades: int = 20) -> float:
    """A symbol's tradeability score: the best out-of-sample Sharpe (with
    friction) achievable by a seed strategy on it, requiring enough trades in
    both halves. -inf if nothing clears the trade floor. This ranks symbols;
    the actual add still goes through the full hardened gate via the reviewer."""
    from . import dsl, search

    best = float("-inf")
    for genome in dsl.seed_genomes():
        trial = search.evaluate_genome(df, symbol, timeframe, genome, friction_bps=friction)
        if (
            trial is not None
            and trial.oos_trades >= min_trades
            and trial.is_trades >= min_trades
            and math.isfinite(trial.oos_sharpe)
        ):
            best = max(best, trial.oos_sharpe)
    return best


def _score_worker(args):
    """Picklable: score one symbol across worker processes."""
    symbol, timeframe, df, friction, min_trades = args
    return score_symbol(symbol, timeframe, df, friction, min_trades)


def read_open_positions(portfolio_path: Path | str) -> set[str]:
    """Instrument keys with a live open position — these are sticky in a
    rotation and never dropped."""
    path = Path(portfolio_path)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).get("positions", {}).keys())
    except Exception:
        return set()


def read_trades(portfolio_path: Path | str) -> list:
    """Closed trades recorded in the portfolio (for the realized-P&L tilt)."""
    path = Path(portfolio_path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("trades", []) or []
    except Exception:
        return []


def performance_tilt(
    trades: list,
    recent: int = 30,
    min_trades: int = 6,
    scale: float = 0.4,
    cap: float = 0.5,
) -> dict[str, float]:
    """Per-instrument score adjustment from the bot's *realized* track record.

    For each instrument with at least ``min_trades`` closed trades, take its
    expectancy in **R** (mean ``pnl / risk_amount``) over its most recent
    ``recent`` trades and map it to a bounded tilt ``clamp(expectancy_R * scale,
    -cap, +cap)``. Expectancy-in-R is risk-normalised, so instruments with
    different sizing and trade counts compare fairly, and it's on the same rough
    scale as the OOS-Sharpe scores it adjusts.

    Positive => the bot has been making money there (rotation should be reluctant
    to drop it); negative => it's been losing (rotation should demote it, even
    past the churn-margin hysteresis). Instruments the bot has never traded get
    no entry here, so newcomers are judged on tradeability score alone. Trades
    with no risk recorded (``risk_amount`` 0/absent) can't form an R and are
    skipped; if too few remain, the instrument gets no tilt (insufficient
    evidence — same spirit as the analytics' 20-trade noise floor).
    """
    by: dict[str, list] = {}
    for t in trades:
        if isinstance(t.get("pnl"), (int, float)) and t.get("closed_at"):
            by.setdefault(t.get("instrument", "?"), []).append(t)
    tilt: dict[str, float] = {}
    for key, ts in by.items():
        ts = sorted(ts, key=lambda t: t["closed_at"])[-recent:]
        rs = [t["pnl"] / t["risk_amount"] for t in ts if t.get("risk_amount")]
        if len(rs) < min_trades:
            continue
        exp_r = sum(rs) / len(rs)
        tilt[key] = max(-cap, min(cap, exp_r * scale))
    return tilt


def propose_rotation(
    current: list[str],
    scores: dict[str, float],
    open_keys,
    max_active: int,
    min_score: float = 0.0,
    churn_margin: float = 0.0,
    performance: Optional[dict[str, float]] = None,
) -> dict:
    """Decide the next live book. Pure. ``current`` is today's traded keys,
    ``scores`` maps every candidate key (current + universe) to its score,
    ``open_keys`` are instruments with a live position (sticky), and
    ``performance`` is the realized-P&L tilt per instrument (see
    :func:`performance_tilt`) — folded into the ranking so what's actually been
    profitable is harder to drop and what's been losing is easier."""
    current, open_keys = set(current), set(open_keys)
    performance = performance or {}

    # Effective rank = tradeability score, plus a churn_margin bonus for current
    # holdings (hysteresis against thrashing), plus the realized-P&L tilt. A
    # losing holding's negative tilt can overcome the churn margin and push it out;
    # a profitable one's positive tilt makes it stickier.
    def effective(k: str) -> float:
        adj = (churn_margin if k in current else 0.0) + performance.get(k, 0.0)
        return scores.get(k, float("-inf")) + adj

    eligible = [k for k in scores if scores[k] >= min_score or k in current or k in open_keys]
    ranked = sorted(eligible, key=effective, reverse=True)

    target: list[str] = []
    for k in ranked:
        if len(target) >= max_active and k not in open_keys:
            continue
        if k not in target:
            target.append(k)
    target_set = set(target) | open_keys  # open positions kept even past the cap

    # open holdings kept purely because they're sticky (would otherwise rank out)
    by_score_top = set(target[:max_active])
    return {
        "target": sorted(target_set),
        "add": sorted(target_set - current),
        "drop": sorted(current - target_set),
        "keep": sorted(current & target_set),
        "kept_sticky": sorted(open_keys - by_score_top),
        "performance": {k: round(performance[k], 3) for k in sorted(performance)},
    }
