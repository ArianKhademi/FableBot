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
# are on PATH so claude/gh resolve the same way they do in a shell.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/bin:$PATH"

# Pin Python to Apple's /usr/bin/python3 (what rotate_universe/train already use).
# Under macOS cron, TCC is per-binary: a Homebrew python in ~/Documents hits
# "Operation not permitted", while the signed system python is allowed — and it
# has the deps. Override with FABLEBOT_PYTHON if you ever need a different one.
PY="${FABLEBOT_PYTHON:-/usr/bin/python3}"

# Cron has no login keychain, so gh's stored token is invisible (this is why the
# job failed every night with "gh is not authenticated"). Load a Personal Access
# Token from a gitignored file outside the repo; gh honours GH_TOKEN ahead of the
# keyring, and `gh auth setup-git` then serves it to git over HTTPS — no keychain.
SECRETS="${FABLEBOT_SECRETS:-$HOME/.config/fablebot/secrets.env}"
if [ -f "$SECRETS" ]; then
    set -a
    . "$SECRETS" || { echo "REVIEW FAILED: could not source $SECRETS (check shell syntax)" >&2; exit 1; }
    set +a
fi
export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-$GH_TOKEN}"
gh auth setup-git 2>/dev/null || true

# Always restore the live branch on exit. If any step after the checkout below
# fails (claude, git push, gh pr create), the working tree must NOT be stranded
# on the review branch — the other cron jobs and run_engine share this tree and
# would then run the bot's proposed, unmerged code. The trap fires on success too.
START_REF="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"
trap 'git checkout --quiet "$START_REF" 2>/dev/null || true' EXIT

DATE="$(date +%Y-%m-%d)"
BRANCH="bot-review/${DATE}"

echo "=== nightly review ${DATE} ($(date '+%H:%M')) ==="

fail() { echo "REVIEW FAILED: $1" >&2; exit 1; }

# Preflight: surface the actual problem instead of a bare 'fatal:' line.
command -v "$PY" >/dev/null 2>&1 || fail "python3 not found at '$PY' (set FABLEBOT_PYTHON)"
command -v git     >/dev/null 2>&1 || fail "git not found on PATH"
command -v claude  >/dev/null 2>&1 || fail "claude CLI not found on PATH — is Claude Code installed for this user?"
command -v gh      >/dev/null 2>&1 || fail "gh CLI not found on PATH — brew install gh"
gh auth status >/dev/null 2>&1 || gh api user >/dev/null 2>&1 \
    || fail "gh cannot authenticate — set GH_TOKEN in $SECRETS (a GitHub PAT). See scripts/README or the repo docs."
# Only source changes should block the review; generated artefacts (logs the cron
# appends to, analysis/ digests the review itself regenerates) must not. Use
# porcelain so STAGED and UNTRACKED source changes are caught too, not just
# unstaged edits — otherwise they'd ride onto the review branch and into its PR.
if [ -n "$(git status --porcelain -- ':!logs' ':!analysis')" ]; then
    fail "working tree has uncommitted/untracked source changes — commit or stash them so the review can branch safely"
fi

# 1. Digest the journal and summarise the optimizer's findings so the reviewer
#    has a single consolidated view rather than raw JSONL to parse.
"$PY" scripts/analyze.py --days 30 --write analysis/latest_digest.md
"$PY" scripts/proposals.py --write analysis/latest_proposals.md 2>/dev/null || true

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

# 5. The EXIT trap restores the live branch (START_REF) so the bot keeps running
#    yesterday's code until a human merges the proposal — on success OR abort.
echo "=== review finished ==="
