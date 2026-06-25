#!/usr/bin/env python3
"""Cross-instrument genome insights — what recurs in the best evolved strategies?

Each instrument's search keeps a history of every genome it scored. This mines the
top genomes from EVERY population and aggregates what the winners have in common:
which indicators show up, whether they trail their stop, what stop width they
settle on, and what kind of entry logic they use. It's the global view the
per-instrument search doesn't give you on its own — a read on whether some
structure is broadly better, not just lucky on one symbol.

Read-only.

  python3 scripts/genome_insights.py            # top 5 genomes per population
  python3 scripts/genome_insights.py --top 10
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"


def top_genomes(history: list, k: int) -> list:
    """The k highest-fitness (genome, fitness) pairs from a population's history."""
    scored = [
        (h["genome"], h["fitness"])
        for h in history
        if isinstance(h.get("fitness"), (int, float)) and math.isfinite(h["fitness"]) and h.get("genome")
    ]
    return sorted(scored, key=lambda gf: gf[1], reverse=True)[:k]


def feature_kinds(genome: dict) -> set:
    """The distinct indicator kinds a genome references (e.g. bb_upper, rsi, adx)."""
    return {f.get("kind") for f in genome.get("features", []) if f.get("kind")}


def has_trail(genome: dict) -> bool:
    return bool(genome.get("risk", {}).get("trail_atr_mult"))


def _entry_ops(genome: dict) -> set:
    ops = set()
    for side in ("entry_long", "entry_short"):
        node = genome.get(side)
        if isinstance(node, dict):
            ops.add(node.get("type", "?"))
    return ops


def summarize(by_instrument: dict[str, list], top_k: int) -> dict:
    """Aggregate the per-instrument top genomes into a cross-instrument view."""
    winners = []           # (instrument, genome, fitness)
    for inst, hist in by_instrument.items():
        for g, fit in top_genomes(hist, top_k):
            winners.append((inst, g, fit))
    n = len(winners)

    kind_genomes = Counter()       # winners that use this kind at all
    kind_instruments = {}          # distinct instruments where it shows up in a winner
    stop_mults, trailing = [], 0
    op_counts = Counter()
    for inst, g, _ in winners:
        for kind in feature_kinds(g):
            kind_genomes[kind] += 1
            kind_instruments.setdefault(kind, set()).add(inst)
        sm = g.get("risk", {}).get("atr_stop_mult")
        if isinstance(sm, (int, float)):
            stop_mults.append(sm)
        trailing += 1 if has_trail(g) else 0
        for op in _entry_ops(g):
            op_counts[op] += 1

    stop_mults.sort()
    champions = {
        inst: max(top_genomes(hist, top_k), key=lambda gf: gf[1])
        for inst, hist in by_instrument.items() if top_genomes(hist, 1)
    }
    return {
        "n_winners": n,
        "n_instruments": len(by_instrument),
        "indicators": [
            {"kind": k, "in_winners": c, "pct": (c / n * 100 if n else 0),
             "instruments": len(kind_instruments.get(k, set()))}
            for k, c in kind_genomes.most_common()
        ],
        "trailing_pct": (trailing / n * 100) if n else 0.0,
        "stop_median": (stop_mults[len(stop_mults) // 2] if stop_mults else None),
        "stop_range": ((min(stop_mults), max(stop_mults)) if stop_mults else None),
        "entry_ops": op_counts.most_common(),
        "champions": champions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-instrument genome insights")
    ap.add_argument("--top", type=int, default=5, help="top genomes per population to mine")
    ap.add_argument("--state-dir", default=STATE)
    args = ap.parse_args()

    by_instrument = {}
    for p in sorted(Path(args.state_dir).glob("population_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        name = p.name[len("population_"):-len(".json")]
        by_instrument[name] = d.get("history", [])
    if not by_instrument:
        print("No populations yet — run run_engine first.")
        return 0

    s = summarize(by_instrument, args.top)
    print("=" * 72)
    print("CROSS-INSTRUMENT GENOME INSIGHTS  (what recurs in the best evolved strategies)")
    print("=" * 72)
    print(f"\nMined the top {args.top} genomes from {s['n_instruments']} populations "
          f"= {s['n_winners']} winners.\n")
    print("Indicators most common in winners:")
    for ind in s["indicators"][:14]:
        print(f"  {ind['kind']:14} {ind['pct']:>3.0f}%  ({ind['in_winners']}/{s['n_winners']} winners, "
              f"{ind['instruments']}/{s['n_instruments']} instruments)")
    print("\nRisk shape of winners:")
    print(f"  trailing stop used in {s['trailing_pct']:.0f}% of winners")
    if s["stop_median"] is not None:
        lo, hi = s["stop_range"]
        print(f"  atr_stop_mult: median {s['stop_median']:.2f}  (range {lo:.1f}–{hi:.1f})")
    print(f"  entry logic: " + ", ".join(f"{op}×{c}" for op, c in s["entry_ops"]))
    print("\nPer-instrument champion (best genome found):")
    for inst in sorted(s["champions"], key=lambda i: -s["champions"][i][1]):
        g, fit = s["champions"][inst]
        feats = ", ".join(sorted(feature_kinds(g))) or "—"
        print(f"  {inst:8} fitness {fit:6.2f}  [{feats}]{'  +trail' if has_trail(g) else ''}")
    print("\nNote: high cross-instrument recurrence = a structure that generalises, not a one-symbol"
          "\nfluke. Fitness here is backtest OOS Sharpe — it still has to clear the gate to ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
