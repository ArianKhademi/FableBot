#!/usr/bin/env python3
"""Render the performance digest from the decision journal.

  python3 scripts/analyze.py                 # last 30 days, to stdout
  python3 scripts/analyze.py --days 7
  python3 scripts/analyze.py --write analysis/latest_digest.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.analytics import digest_from_journal, render_digest
from fablebot.journal import Journal

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot journal analytics")
    parser.add_argument("--journal", default=ROOT / "state/journal.jsonl")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--write", help="also write the digest to this path")
    args = parser.parse_args()

    journal = Journal(args.journal)
    digest = digest_from_journal(journal, days=args.days)
    text = render_digest(digest)
    print(text)
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\n[written to {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
