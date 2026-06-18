#!/usr/bin/env python3
"""Is the engine actually learning — and has it found/proposed anything?

A read-only snapshot that answers, in one place:
  - GENOME SEARCH: best fitness per instrument and its learning curve (rising
    then plateauing = healthy; flat from the start = stuck), and how many new
    genomes it evaluates per generation (near 0 = spinning, not learning).
  - NUMERIC SEARCH: trials logged and the best OOS Sharpe found (random search
    explores and plateaus — it doesn't "learn").
  - PROPOSAL STREAMS: what has actually cleared the gate in each pipeline
    (numeric params, evolved genomes, ML, RL) — i.e. what the reviewer can ship.

No network, no writes.

  python3 scripts/engine_status.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"


def _load_jsonl(path: Path):
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


def _latest_status(events):
    state = {}
    for e in events:
        pid = e.get("proposal_id")
        if pid is None:
            continue
        state.setdefault(pid, {}).update(e)
    return state


def main() -> None:
    print("=" * 72)
    print("FableBot engine status")
    print("=" * 72)

    # ── genome search ────────────────────────────────────────────────────────
    print("\nGENOME SEARCH  (the part that learns — fitness should rise then plateau)")
    pops = sorted(STATE.glob("population_*.json"))
    if not pops:
        print("  no populations yet — run scripts/run_engine.py")
    for p in pops:
        d = json.loads(p.read_text())
        name = p.name[len("population_"):-len(".json")]
        gen = d.get("generation", 0)
        hist = d.get("history", [])
        fits = [h["fitness"] for h in hist]
        if not fits:
            print(f"  {name:5} gen {gen:>5} — no evaluations yet")
            continue
        def best_at(frac):
            n = max(1, int(len(fits) * frac))
            return max(fits[:n])
        epg = len(hist) / max(gen, 1)
        trend = "rising" if best_at(1.0) > best_at(0.25) + 1e-6 else "flat/plateaued"
        print(f"  {name:5} gen {gen:>5} | best {max(fits):6.2f} "
              f"(@25%→now: {best_at(.25):.2f}→{max(fits):.2f}, {trend}) | "
              f"{epg:.1f} evals/gen {'⚠ spinning' if epg < 0.5 else ''}")

    # ── numeric search ───────────────────────────────────────────────────────
    print("\nNUMERIC SEARCH  (random param tuning — explores, plateaus)")
    hist = _load_jsonl(STATE / "backtest_history.jsonl")
    best, cnt, recent_best = defaultdict(lambda: float("-inf")), Counter(), defaultdict(lambda: float("-inf"))
    for t in hist:
        if t.get("is_baseline"):
            continue
        k, s = t.get("instrument"), t.get("oos_sharpe", float("-inf"))
        cnt[k] += 1
        if s > best[k]:
            best[k] = s
    for t in hist[-5000:]:
        if t.get("is_baseline"):
            continue
        k, s = t.get("instrument"), t.get("oos_sharpe", float("-inf"))
        if s > recent_best[k]:
            recent_best[k] = s
    for k in sorted(cnt):
        print(f"  {k:5} {cnt[k]:>7} trials | best OOS Sharpe ever {best[k]:6.2f} | last 5k {recent_best[k]:6.2f}")
    print(f"  total trials logged: {sum(cnt.values())}")

    # ── proposal streams (what has cleared the gate) ─────────────────────────
    print("\nPROPOSALS  (what the gate has passed — i.e. what the reviewer can ship)")
    num = _latest_status(_load_jsonl(STATE / "proposals.jsonl"))
    sc = Counter(p.get("status") for p in num.values())
    print(f"  numeric params : {sc.get('qualified',0)} qualified, {sc.get('candidate',0)} candidate, "
          f"{sc.get('rejected',0)} rejected, {sc.get('promoted',0)} promoted")
    for label, fname in [("evolved genomes", "genome_proposals.jsonl"),
                         ("ML models", "ml_proposals.jsonl"),
                         ("RL agents", "rl_proposals.jsonl")]:
        recs = _load_jsonl(STATE / fname)
        insts = sorted({r.get("instrument") for r in recs})
        print(f"  {label:15}: {len(recs)} qualified" + (f" — {insts}" if insts else ""))

    # ── scheduled-job heartbeat (did the crons actually fire?) ───────────────
    print("\nLAST SCHEDULED RUNS  (proof the cron jobs fired, even when they found nothing)")
    runs = _load_jsonl(STATE / "run_log.jsonl")[-8:]
    if not runs:
        print("  none recorded yet — train_ml/train_rl/rotate_universe append here when run")
    for r in runs:
        print(f"  {r.get('ts','?')}  {r.get('job','?'):16} {r.get('summary','')}")

    print("\nReminder: nothing here is live until you (or the nightly reviewer) edit config.yaml.")


if __name__ == "__main__":
    main()
