#!/usr/bin/env python3
"""Pipeline view of the optimizer's proposals, plus status transitions.

  python3 scripts/proposals.py                          # print pipeline view
  python3 scripts/proposals.py --write analysis/latest_proposals.md
  python3 scripts/proposals.py --mark <id> promoted     # reviewer acted on it
  python3 scripts/proposals.py --mark <id> rejected --reason "..."
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.pipeline import CANDIDATE, PROMOTED, QUALIFIED, REJECTED, ProposalStore

ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _proposal_line(p: dict) -> list[str]:
    lines = [
        f"- `{p['proposal_id']}` **{p['instrument']}** ({p['strategy']}, {p['timeframe']}): "
        f"OOS Sharpe {p['oos_sharpe']:.3f} vs baseline {p['baseline_oos_sharpe']:.3f} "
        f"(+{p['improvement']:.3f}), {p['oos_trades']} OOS trades"
    ]
    lines.append(f"  proposed: `{p['proposed_params']}`")
    lines.append(f"  current:  `{p['current_params']}`")
    if "folds_won" in p:
        lines.append(
            f"  validation: won {p['folds_won']}/{p['folds_valid']} regime folds, "
            f"median perturbed Sharpe {p.get('median_perturbed_sharpe', float('nan')):.3f}"
        )
    if p.get("reason"):
        lines.append(f"  verdict: {p['reason']}")
    return lines


def render(store: ProposalStore, history_path: Path) -> str:
    lines = ["# Proposal pipeline", ""]
    lines.append("Stages: candidate (one walk-forward win) -> validation "
                 "(regime folds + perturbation) -> qualified -> promoted/rejected.")
    lines.append("**Only qualified proposals may be acted on.**")
    lines.append("")

    sections = [
        (QUALIFIED, "Qualified — ready for review"),
        (CANDIDATE, "Candidates — awaiting validation"),
        (PROMOTED, "Promoted — already shipped"),
        (REJECTED, "Rejected — failed validation (kept for the record)"),
    ]
    for status, title in sections:
        items = store.by_status(status)
        lines.append(f"## {title} ({len(items)})")
        if items:
            for p in sorted(items, key=lambda x: -x.get("improvement", 0)):
                if status == REJECTED:
                    # keep rejects terse — the reason is what matters
                    lines.append(
                        f"- `{p['proposal_id']}` {p['instrument']}: {p.get('reason', 'failed')}"
                    )
                else:
                    lines.extend(_proposal_line(p))
        else:
            lines.append("none")
        lines.append("")

    # ── backtest history summary ─────────────────────────────────────────────
    history = _load_jsonl(history_path)
    lines.append("## Backtest history by instrument")
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
            baseline_sharpe = f"{baseline['oos_sharpe']:.3f}" if baseline else "n/a"
            lines.append(
                f"- **{key}**: {n} trials total "
                f"(baseline OOS Sharpe: {baseline_sharpe}, "
                f"best found: {best['oos_sharpe']:.3f}, "
                f"avg: {avg_oos:.3f}, "
                f"worst: {worst['oos_sharpe']:.3f})"
            )
        else:
            lines.append(f"- **{key}**: {n} trials (baseline only so far)")

    lines += [
        "",
        f"Total trials logged: {len(history)}. "
        "High in-sample / low OOS Sharpe = overfit; the validation stage exists "
        "to catch what slips past the first filter.",
    ]

    # ── pipeline totals (bottom line: easy to read after a long history) ──────
    n_candidate = len(store.by_status(CANDIDATE))
    n_qualified = len(store.by_status(QUALIFIED))
    n_promoted = len(store.by_status(PROMOTED))
    n_rejected = len(store.by_status(REJECTED))
    n_validated = n_qualified + n_promoted + n_rejected  # completed the gate
    n_passed = n_qualified + n_promoted                  # survived validation
    pass_rate = (n_passed / n_validated * 100.0) if n_validated else 0.0
    lines += [
        "",
        "## Pipeline totals",
        f"- candidates awaiting validation: {n_candidate}",
        f"- **validated (passed)**: {n_passed}  "
        f"(qualified {n_qualified} + promoted {n_promoted})",
        f"- **rejected (failed)**: {n_rejected}",
        f"- validation pass rate: {pass_rate:.0f}% "
        f"({n_passed}/{n_validated} that finished validation)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Proposal pipeline view")
    parser.add_argument("--proposals", default=ROOT / "state/proposals.jsonl")
    parser.add_argument("--history", default=ROOT / "state/backtest_history.jsonl")
    parser.add_argument("--write", help="also write output to this path")
    parser.add_argument("--mark", nargs=2, metavar=("ID", "STATUS"),
                        help="transition a proposal (e.g. --mark ab12cd34ef promoted)")
    parser.add_argument("--reason", default="", help="reason recorded with --mark")
    args = parser.parse_args()

    store = ProposalStore(args.proposals)

    if args.mark:
        pid, status = args.mark
        store.set_status(pid, status, reason=args.reason or f"marked {status} via CLI")
        print(f"{pid} -> {status}")
        return

    text = render(store, Path(args.history))
    print(text)
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\n[written to {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
