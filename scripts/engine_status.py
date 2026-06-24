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
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.dsl import genome_id

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"

# A population file untouched for longer than this isn't being evolved by the
# current run (run_engine rewrites every active population each cycle, seconds
# apart). Common cause: running `--universe`, which only evolves the scan list,
# so the traded populations sit idle — or run_engine is stopped entirely.
PARKED_MIN = 45.0


def classify_population(sat: float, age_min: float) -> str:
    """Describe a genome population from its saturation and how long since its
    file changed. The key distinction: a *parked* population (not updated by the
    current run) is not the same as one that's *spinning* (actively evolving but
    stuck re-scoring seen genomes). Telling someone running --universe to 'restart
    for the immigrant fix' because a parked traded pop looks saturated is wrong —
    the current run isn't touching it, and the fix is already in."""
    if age_min > PARKED_MIN:
        return f"parked {age_min / 60:.1f}h — not updated by the current run (e.g. running --universe)"
    if sat >= 0.9:
        return "⚠ spinning — still saturated with the immigrant fix active"
    if sat >= 0.5:
        return "nearing convergence"
    return "exploring"


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
        # Live convergence signal: how much of the CURRENT population is already
        # scored. ~100% means crossover/mutation only regenerate seen genomes —
        # the GA is spinning. (Retained history is capped + not chronological, so
        # an in-history "curve" would mislead; saturation reflects the present.)
        scored = set(d.get("scores", {}).keys())
        pop = d.get("population", [])
        sat = sum(1 for g in pop if genome_id(g) in scored) / max(len(pop), 1)
        age_min = (time.time() - p.stat().st_mtime) / 60.0
        state = classify_population(sat, age_min)
        print(f"  {name:5} gen {gen:>5} | best {max(fits):6.2f} | population {sat:.0%} already-explored — {state}")

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

    # ── live positions (some queued config changes wait on these) ────────────
    print("\nLIVE POSITIONS  (open paper trades — a queued change can be gated on one going flat)")
    pf = STATE / "portfolio.json"
    positions = {}
    if pf.exists():
        try:
            positions = json.loads(pf.read_text()).get("positions", {}) or {}
        except Exception:
            positions = {}
    if positions:
        for k in sorted(positions):
            p = positions[k]
            print(f"  {k:5} {p.get('direction','?'):5} {p.get('strategy','?'):16} "
                  f"since {str(p.get('opened_at',''))[:10]}")
    else:
        print("  flat — no open positions")
    if "NDX" in positions:
        print("  ⏳ NDX is held on 15m — the queued NDX→daily switch (promote_when_flat.sh) "
              "waits until it's flat")

    print("\nReminder: nothing here is live until you (or the nightly reviewer) edit config.yaml.")


if __name__ == "__main__":
    main()
