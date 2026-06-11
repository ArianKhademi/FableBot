#!/usr/bin/env bash
# Nightly learning loop: digest the journal, have Claude Code review it, and
# open a draft PR if (and only if) it found something worth changing.
#
# Cron (after the 21:00 evening recap):
#   30 21 * * * /path/to/FableBot/scripts/nightly_review.sh >> /path/to/FableBot/logs/review.log 2>&1
#
# Requires: claude (Claude Code CLI, authenticated), gh (authenticated), git.

set -euo pipefail
cd "$(dirname "$0")/.."

DATE="$(date +%Y-%m-%d)"
BRANCH="bot-review/${DATE}"

echo "=== nightly review ${DATE} ==="

# 1. Digest the journal (analytics are deterministic; this always runs).
python3 scripts/analyze.py --days 30 --write analysis/latest_digest.md

# 2. Review on a fresh branch off main so proposals never touch the running code.
git fetch origin main
git checkout -B "${BRANCH}" origin/main

# 3. Claude reads the digest + journal + code under the reviewer constitution.
claude -p "$(cat prompts/reviewer.md)" --permission-mode acceptEdits

# 4. Only open a PR if the review actually committed a change.
if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
    git push -u origin "${BRANCH}"
    BODY="analysis/review_summary.md"
    if [ -f "${BODY}" ]; then
        gh pr create --draft --title "Bot review ${DATE}: proposed change" --body-file "${BODY}"
    else
        gh pr create --draft --fill
    fi
    echo "Review proposed a change: PR opened from ${BRANCH}."
else
    echo "Review complete: no change warranted."
fi

# 5. Return to the live branch so the bot keeps running yesterday's code
#    until a human merges the proposal.
git checkout -
