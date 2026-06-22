#!/usr/bin/env bash
# Nightly learning loop: digest the journal, have Claude Code review it, and
# open a draft PR if (and only if) it found something worth changing.
#
# Cron (after the 21:00 evening recap):
#   30 21 * * * /Users/ariankhademi/Documents/Business/FableBot/scripts/nightly_review.sh >> /Users/ariankhademi/Documents/Business/FableBot/logs/review.log 2>&1
#
# Requires: claude (Claude Code CLI, authenticated), gh (authenticated), git.
# Test it manually first: bash scripts/nightly_review.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Cron runs with a bare-bones environment; make sure Homebrew and user bins
# are on PATH so claude/gh/python3 resolve the same way they do in a shell.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/bin:$PATH"

# Cron has no login keychain, so gh's stored token is invisible (this is why the
# job failed every night with "gh is not authenticated"). Load a Personal Access
# Token from a gitignored file outside the repo; gh honours GH_TOKEN ahead of the
# keyring, and `gh auth setup-git` then serves it to git over HTTPS — no keychain.
SECRETS="${FABLEBOT_SECRETS:-$HOME/.config/fablebot/secrets.env}"
if [ -f "$SECRETS" ]; then
    set -a; . "$SECRETS"; set +a
fi
export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-$GH_TOKEN}"
gh auth setup-git 2>/dev/null || true

DATE="$(date +%Y-%m-%d)"
BRANCH="bot-review/${DATE}"

echo "=== nightly review ${DATE} ($(date '+%H:%M')) ==="

fail() { echo "REVIEW FAILED: $1" >&2; exit 1; }

# Preflight: surface the actual problem instead of a bare 'fatal:' line.
command -v python3 >/dev/null 2>&1 || fail "python3 not found on PATH"
command -v git     >/dev/null 2>&1 || fail "git not found on PATH"
command -v claude  >/dev/null 2>&1 || fail "claude CLI not found on PATH — is Claude Code installed for this user?"
command -v gh      >/dev/null 2>&1 || fail "gh CLI not found on PATH — brew install gh"
gh auth status >/dev/null 2>&1 || gh api user >/dev/null 2>&1 \
    || fail "gh cannot authenticate — set GH_TOKEN in $SECRETS (a GitHub PAT). See scripts/README or the repo docs."
# Only source changes should block the review; generated artefacts (logs the cron
# appends to, analysis/ digests the review itself regenerates) must not.
git diff --quiet -- ':!logs' ':!analysis' \
    || fail "working tree has uncommitted source changes — commit or stash them so the review can branch safely"

# 1. Digest the journal and summarise the optimizer's findings so the reviewer
#    has a single consolidated view rather than raw JSONL to parse.
python3 scripts/analyze.py --days 30 --write analysis/latest_digest.md
python3 scripts/proposals.py --write analysis/latest_proposals.md 2>/dev/null || true

# 2. Review on a fresh branch off main so proposals never touch the running code.
git fetch origin main \
    || fail "git fetch failed — cron has no interactive credentials; run: gh auth setup-git (or switch the remote to SSH)"
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
echo "=== review finished ==="
