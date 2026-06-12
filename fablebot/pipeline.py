"""The proposal pipeline: candidate -> validating -> qualified/rejected -> promoted.

A single walk-forward win can be luck. Proposals move through stages, with
each transition recorded in an append-only event log (state/proposals.jsonl)
so the full history of what was tried, validated, rejected, and promoted is
always reconstructable.

Stages:
  candidate   one walk-forward trial beat the baseline (weak evidence)
  qualified   survived multi-fold regime testing AND parameter perturbation
  rejected    failed validation — kept in the log so the optimizer and
              reviewer know this region of parameter space looked good once
              and didn't hold up
  promoted    the reviewer acted on it (config.yaml changed via PR)

Only *qualified* proposals reach the nightly reviewer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CANDIDATE = "candidate"
QUALIFIED = "qualified"
REJECTED = "rejected"
PROMOTED = "promoted"

STATUSES = (CANDIDATE, QUALIFIED, REJECTED, PROMOTED)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def proposal_id(instrument: str, params: dict) -> str:
    blob = json.dumps([instrument, params], sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


@dataclass
class Proposal:
    instrument: str
    strategy: str
    timeframe: str
    proposed_params: dict
    current_params: dict
    oos_sharpe: float
    baseline_oos_sharpe: float
    improvement: float
    oos_trades: int
    timestamp: str = field(default_factory=_utcnow)

    @property
    def id(self) -> str:
        return proposal_id(self.instrument, self.proposed_params)


class ProposalStore:
    """Append-only event log. The latest event per proposal id wins; earlier
    events are history, never rewritten."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict) -> None:
        record = {"ts": _utcnow(), **record}
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    # -- writes ---------------------------------------------------------------

    def add_candidate(self, proposal: Proposal) -> str:
        pid = proposal.id
        if pid in self.latest():  # don't re-add a proposal we've already seen
            return pid
        self._append({"proposal_id": pid, "status": CANDIDATE, **asdict(proposal)})
        return pid

    def set_status(self, pid: str, status: str, **details) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; choose from {STATUSES}")
        if pid not in self.latest():
            raise KeyError(f"unknown proposal id {pid!r}")
        self._append({"proposal_id": pid, "status": status, **details})

    # -- reads ----------------------------------------------------------------

    def latest(self) -> dict[str, dict]:
        """Current state per proposal: first event's data merged with every
        later event's updates, last status wins."""
        state: dict[str, dict] = {}
        for event in self._events():
            pid = event["proposal_id"]
            if pid in state:
                state[pid].update(event)
            else:
                state[pid] = dict(event)
        return state

    def by_status(self, status: str) -> list[dict]:
        return [p for p in self.latest().values() if p.get("status") == status]

    def history(self, pid: str) -> list[dict]:
        return [e for e in self._events() if e["proposal_id"] == pid]
