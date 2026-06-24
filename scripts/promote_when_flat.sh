#!/usr/bin/env bash
# Queue helper: when an instrument goes FLAT, mark its queued "change-when-flat"
# PR ready-for-review and notify — so a config change that must not land mid-trade
# (e.g. NDX -> daily) is teed up the instant the open position closes, without a
# human watching for it. It NEVER merges; a human still does that. Read-only while
# the instrument is open, and idempotent (no-op once the PR is ready or merged).
#
# Cron (after the 21:00 evening recap, so it reflects the day's closes):
#   15 21 * * * /Users/ariankhademi/Documents/Business/FableBot/scripts/promote_when_flat.sh >> /Users/ariankhademi/Documents/Business/FableBot/logs/ndx_watch.log 2>&1
#
#   promote_when_flat.sh [INSTRUMENT] [BRANCH]
#   defaults: NDX  claude/ndx-daily-when-flat
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/bin:$PATH"

INSTRUMENT="${1:-NDX}"
BRANCH="${2:-claude/ndx-daily-when-flat}"
STAMP="[promote-when-flat $(date '+%Y-%m-%d %H:%M') ${INSTRUMENT}]"

# gh auth: same non-interactive PAT mechanism as the nightly review (cron has no
# login keychain). Without it the gh calls below fail safely (no PR found -> no-op).
# Source in an errexit-off subshell and re-import via export -p so a malformed
# secrets file (stray quote/paren) can't abort the whole watcher under set -e.
SECRETS="${FABLEBOT_SECRETS:-$HOME/.config/fablebot/secrets.env}"
if [ -f "$SECRETS" ]; then
    eval "$(set +e; set -a; . "$SECRETS" 2>/dev/null; export -p 2>/dev/null)" 2>/dev/null || true
fi
export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-$GH_TOKEN}"

pr_number() {  # the open PR number on $BRANCH, or empty (gh failure -> empty -> no-op)
    gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' 2>/dev/null || true
}
pr_is_draft() {  # "true" / "false" / "unknown"
    gh pr view "$1" --json isDraft --jq '.isDraft' 2>/dev/null || echo unknown
}

# Position check. `set -e` would abort on the non-zero exit codes we rely on, so
# capture the code explicitly. 0=flat, 1=open, 2=unknown.
status=0
python3 scripts/position_flat.py "$INSTRUMENT" >/dev/null 2>&1 || status=$?
if [ "$status" -eq 1 ]; then
    # OPEN. If a prior run already promoted the PR and the position has REOPENED,
    # walk it back to draft so the switch can't be merged mid-trade.
    num="$(pr_number)"
    if [ -n "$num" ] && [ "$(pr_is_draft "$num")" = "false" ]; then
        gh pr ready "$num" --undo \
            && echo "$STAMP reopened — PR #$num re-drafted to block a mid-trade merge." \
            || echo "$STAMP reopened — failed to re-draft PR #$num (will retry)." >&2
    fi
    echo "$STAMP still open — switch stays queued."
    exit 0
elif [ "$status" -ne 0 ]; then
    echo "$STAMP cannot determine position (portfolio.json missing/unreadable) — no action." >&2
    exit 0
fi

# FLAT. Find the queued PR by its head branch.
num="$(pr_number)"
if [ -z "$num" ]; then
    echo "$STAMP FLAT, but no open PR on $BRANCH (already merged, or not created) — nothing to do."
    exit 0
fi
if [ "$(pr_is_draft "$num")" != "true" ]; then
    echo "$STAMP FLAT; PR #$num already ready — nothing to do."
    exit 0
fi

# Promote — and only notify if it actually succeeded, so a transient gh/network
# failure can't fire a false "ready to merge" alert for a still-draft PR.
if gh pr ready "$num"; then
    echo "$STAMP FLAT — marked PR #$num ready for review."
else
    echo "$STAMP FLAT but 'gh pr ready' FAILED for PR #$num — not notifying (will retry)." >&2
    exit 1
fi
MSG="FableBot: $INSTRUMENT is now flat — PR #$num is ready to merge (switch $INSTRUMENT to daily)."
# Notify through the same webhook the daily reports use, if configured, then a
# best-effort macOS notification. Neither failing should fail the job.
if [ -n "${FABLEBOT_WEBHOOK_URL:-}" ]; then
    curl -fsS -X POST -H 'Content-Type: application/json' \
        -d "{\"text\": \"${MSG}\"}" "$FABLEBOT_WEBHOOK_URL" >/dev/null 2>&1 || true
fi
osascript -e "display notification \"${MSG}\" with title \"FableBot\"" >/dev/null 2>&1 || true
