"""The decision journal: an append-only JSONL record of everything the bot
decides and why.

Every cycle, every instrument gets an entry — opens, closes, stops, holds,
and *blocked* signals — together with the market context the strategy saw
(z-score, channel levels, EMAs, ATR). This is the raw material the learning
loop runs on: you can't learn from decisions you didn't write down.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    """JSONL-backed when given a path, in-memory otherwise (for tests)."""

    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path else None
        self._memory: list[dict] = []

    def record(self, event: str, **fields) -> dict:
        entry = {"ts": _utcnow(), "event": event, **fields}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
        else:
            self._memory.append(entry)
        return entry

    def record_decision(
        self,
        instrument: str,
        action: str,
        reason: str,
        price: float,
        atr: float,
        context: dict,
        position_direction: Optional[str],
        equity: float,
        shadow: bool = False,
    ) -> dict:
        return self.record(
            "decision",
            instrument=instrument,
            action=action,
            reason=reason,
            price=price,
            atr=atr,
            context=context,
            position_direction=position_direction,
            equity=equity,
            shadow=shadow,
        )

    def record_trade(self, trade, shadow: bool = False) -> dict:
        from dataclasses import asdict

        return self.record("trade_closed", shadow=shadow, trade=asdict(trade))

    def read(self, since: Optional[date] = None) -> list[dict]:
        if self.path:
            if not self.path.exists():
                return []
            entries = [
                json.loads(line)
                for line in self.path.read_text().splitlines()
                if line.strip()
            ]
        else:
            entries = list(self._memory)
        if since is not None:
            entries = [
                e for e in entries if datetime.fromisoformat(e["ts"]).date() >= since
            ]
        return entries
