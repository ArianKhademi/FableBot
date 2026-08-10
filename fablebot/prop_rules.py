"""Prop-firm account rules as a hard constraint layer (paper simulation).

Models the survival rules of a funded-futures evaluation (My Funded Futures
style) so a strategy can be tested against the rules that actually end
accounts, on paper, before any real order exists:

- **Trailing max drawdown** — the account blows when equity touches a floor
  that trails the equity high-watermark by ``max_drawdown``.
  ``drawdown_mode``:
    * ``"eod"``      — the watermark AND the breach check use end-of-day
      equity only; intraday dips don't count (evaluation + Rapid-EOD funded).
    * ``"realtime"`` — the watermark ratchets on every mark and a breach at
      any moment blows the account (Rapid sim-funded).
- **Buffer** — once the watermark clears ``starting + buffer`` the floor
  locks at the starting balance and stops trailing (approximation of MFF's
  buffer rule).
- **Profit target** — evaluation passes when equity reaches
  ``starting + profit_target`` (reported, trading continues).
- **Consistency** — no single day may contribute more than
  ``consistency_pct`` of total profit at the time the target is hit
  (reported; MFF applies it at payout/passing, not as an intraday stop).
- **Contract cap** — ``max_contracts`` (in MICRO contracts, 10:1 scaling
  already applied) across the whole account at once; enforced by the engine
  at entry.

Pure bookkeeping + JSON persistence. The engine feeds it equity marks and
day boundaries; it never touches positions itself. A blown account stays
blown across restarts — that's the point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PropState:
    starting_equity: float
    watermark: float
    floor: float
    blown: bool = False
    blown_reason: str = ""
    passed_on: str = ""              # date the profit target was first reached
    floor_locked: bool = False       # buffer reached -> floor stops trailing
    # session date -> realized-equity change that day (for the consistency rule)
    daily_pnl: dict = field(default_factory=dict)
    last_equity: float = 0.0


class PropAccount:
    def __init__(
        self,
        starting_equity: float,
        max_drawdown: float,
        drawdown_mode: str = "eod",
        profit_target: Optional[float] = None,
        consistency_pct: Optional[float] = None,
        buffer: Optional[float] = None,
        max_contracts: Optional[int] = None,
        state_path: Optional[Path | str] = None,
    ):
        mode = str(drawdown_mode).strip().lower()
        if mode not in ("eod", "realtime"):
            raise ValueError(f"drawdown_mode must be 'eod' or 'realtime', got {drawdown_mode!r}")
        self.max_drawdown = float(max_drawdown)
        self.drawdown_mode = mode
        self.profit_target = float(profit_target) if profit_target else None
        self.consistency_pct = float(consistency_pct) if consistency_pct else None
        self.buffer = float(buffer) if buffer else None
        self.max_contracts = int(max_contracts) if max_contracts else None
        self.state_path = Path(state_path) if state_path else None
        self.state = PropState(
            starting_equity=float(starting_equity),
            watermark=float(starting_equity),
            floor=float(starting_equity) - float(max_drawdown),
            last_equity=float(starting_equity),
        )
        if self.state_path and self.state_path.exists():
            self._load()

    # -- rule evaluation -----------------------------------------------------

    @property
    def blown(self) -> bool:
        return self.state.blown

    @property
    def passed(self) -> bool:
        return bool(self.state.passed_on)

    def _ratchet(self, equity: float) -> None:
        s = self.state
        if equity > s.watermark:
            s.watermark = equity
            if self.buffer and not s.floor_locked and s.watermark >= s.starting_equity + self.buffer:
                s.floor_locked = True
            s.floor = s.starting_equity if s.floor_locked else max(s.floor, s.watermark - self.max_drawdown)

    def _check_target(self, equity: float, day: str) -> None:
        s = self.state
        if self.profit_target and not s.passed_on and equity >= s.starting_equity + self.profit_target:
            s.passed_on = day

    def update(self, equity: float, day: str) -> Optional[str]:
        """Feed the current total (marked) equity. Returns a status line when
        something rule-relevant just happened, else None."""
        s = self.state
        if s.blown:
            return None
        s.last_equity = equity
        self._check_target(equity, day)
        if self.drawdown_mode == "realtime":
            self._ratchet(equity)
            if equity <= s.floor:
                s.blown = True
                s.blown_reason = (
                    f"realtime trailing drawdown: equity {equity:,.2f} touched floor {s.floor:,.2f}"
                )
                self._save()
                return f"PROP ACCOUNT BLOWN — {s.blown_reason}"
        self._save()
        if s.passed_on == day:
            return f"PROP EVAL TARGET REACHED ({equity - s.starting_equity:+,.2f}) — {self.consistency_note()}"
        return None

    def end_of_day(self, equity: float, day: str) -> Optional[str]:
        """Close a session: record the day's P&L; in EOD mode this is also the
        only place the drawdown ratchets and can blow the account."""
        s = self.state
        if s.blown:
            return None
        prev_total = s.starting_equity + sum(s.daily_pnl.values())
        s.daily_pnl[day] = equity - prev_total
        if self.drawdown_mode == "eod":
            if equity <= s.floor:
                s.blown = True
                s.blown_reason = f"EOD drawdown: close {equity:,.2f} at/below floor {s.floor:,.2f}"
                self._save()
                return f"PROP ACCOUNT BLOWN — {s.blown_reason}"
            self._ratchet(equity)
        self._save()
        return None

    def consistency_note(self) -> str:
        """Best-day share of total profit vs the consistency cap (informational
        — MFF applies it when passing/paying out, not as a live stop)."""
        s = self.state
        total = sum(p for p in s.daily_pnl.values())
        if not self.consistency_pct or total <= 0 or not s.daily_pnl:
            return "consistency: n/a"
        best = max(s.daily_pnl.values())
        share = best / total * 100.0
        ok = "OK" if share <= self.consistency_pct else "VIOLATED"
        return f"consistency {share:.0f}% of profit in best day (cap {self.consistency_pct:.0f}%) — {ok}"

    def status_line(self) -> str:
        s = self.state
        room = s.last_equity - s.floor
        tag = "BLOWN" if s.blown else ("PASSED" if s.passed_on else "live")
        return (f"prop[{tag}] eq {s.last_equity:,.0f} | floor {s.floor:,.0f} "
                f"({self.drawdown_mode}) | room {room:,.0f}")

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def _load(self) -> None:
        raw = json.loads(self.state_path.read_text())
        known = {f for f in PropState.__dataclass_fields__}
        self.state = PropState(**{k: v for k, v in raw.items() if k in known})
