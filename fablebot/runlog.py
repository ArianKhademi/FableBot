"""A tiny append-only run log so scheduled jobs leave proof they ran.

Each periodic script records one timestamped line per run — even when it finds
nothing — so you can confirm cron actually fired and see what each job did,
without digging through separate log files. ``engine_status.py`` surfaces the
most recent entries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path("state/run_log.jsonl")


def record_run(name: str, summary: str, path: Path | str = DEFAULT_PATH) -> None:
    """Append a ``{ts, job, summary}`` line. Never raises — logging must not
    take down a job."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "job": name,
                "summary": summary,
            }) + "\n")
    except Exception:
        pass


def recent_runs(limit: int = 10, path: Path | str = DEFAULT_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows[-limit:]
