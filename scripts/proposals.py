#!/usr/bin/env python3
"""Summarise the optimizer's proposals and backtest history for the reviewer.

  python3 scripts/proposals.py                        # print to stdout
  python3 scripts/proposals.py --write analysis/latest_proposals.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def render(proposals_path: Path, history_path: Path) -> str:
    proposals = _load_jsonl(proposals_path)
    history = _load_jsonl(history_path)

    lines = ["# Optimizer summary", ""]

    # ── proposals ─────────────────────────────────────────────────────────────
    pending = [p for p in proposals if not p.get("promoted")]
    promoted = [p for p in proposals if p.get("promoted")]
    lines.append(f"## Pending proposals ({len(pending)})")
    if pending:
        for p in sorted(pending, key=lambda x: -x["improvement"]):
            lines.append(
                f"- **{p['instrument']}** ({p['strategy']}, {p['timeframe']}): "
                f"OOS Sharpe {p['oos_sharpe']:.3f} vs baseline {p['baseline_oos_sharpe']:.3f} "
                f"(+{p['improvement']:.3f}), {p['oos_trades']} OOS trades"
            )
            lines.append(f"  proposed: `{p['proposed_params']}`")
            lines.append(f"  current:  `{p['current_params']}`")
    else:
        lines.append("No pending proposals — optimizer hasn't beaten the baseline by ≥0.15 yet.")

    lines += ["", f"## Previously promoted ({len(promoted)})"]
    for p in promoted[-5:]:
        lines.append(
            f"- {p['instrument']}: OOS +{p['improvement']:.3f} Sharpe on {p['timestamp'][:10]}"
        )

    # ── history summary per instrument ───────────────────────────────────────
    lines += ["", "## Backtest history by instrument"]
    by_instrument: dict[str, list[dict]] = {}
    for t in history:
        by_instrument.setdefault(t["instrument"], []).append(t)

    for key in sorted(by_instrument):
        trials = by_instrument[key]
        n = len(trials)
        baseline = next((t for t in reversed(trials) if t.get("is_baseline")), None)
        non_baseline = [t for t in trials if not t.get("is_baseline")]
        if non_baseline:
            best = max(non_baseline, key=lambda t: t["oos_sharpe"])
            worst = min(non_baseline, key=lambda t: t["oos_sharpe"])
            avg_oos = sum(t["oos_sharpe"] for t in non_baseline) / len(non_baseline)
            lines.append(
                f"- **{key}**: {n} trials total "
                f"(baseline OOS Sharpe: {baseline['oos_sharpe']:.3f if baseline else 'n/a'}, "
                f"best found: {best['oos_sharpe']:.3f}, "
                f"avg: {avg_oos:.3f}, "
                f"worst: {worst['oos_sharpe']:.3f})"
            )
        else:
            lines.append(f"- **{key}**: {n} trials (baseline only so far)")

    total = len(history)
    lines += [
        "",
        f"Total trials logged: {total}. "
        "High in-sample / low OOS Sharpe = overfit; ignore those. "
        "Proposals reaching the reviewer already passed the OOS threshold filter.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise optimizer proposals")
    parser.add_argument("--proposals", default=ROOT / "state/proposals.jsonl")
    parser.add_argument("--history", default=ROOT / "state/backtest_history.jsonl")
    parser.add_argument("--write", help="also write output to this path")
    args = parser.parse_args()

    text = render(Path(args.proposals), Path(args.history))
    print(text)
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\n[written to {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
