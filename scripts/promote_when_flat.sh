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
SECRETS="${FABLEBOT_SECRETS:-$HOME/.config/fablebot/secrets.env}"
if [ -f "$SECRETS" ]; then set -a; . "$SECRETS" || true; set +a; fi
export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-$GH_TOKEN}"

# Position check. `set -e` would abort on the non-zero exit codes we rely on, so
# capture the code explicitly. 0=flat, 1=open, 2=unknown.
status=0
python3 scripts/position_flat.py "$INSTRUMENT" >/dev/null 2>&1 || status=$?
if [ "$status" -eq 1 ]; then
    echo "$STAMP still open — switch stays queued."
    exit 0
elif [ "$status" -ne 0 ]; then
    echo "$STAMP cannot determine position (portfolio.json missing/unreadable) — no action." >&2
    exit 0
fi

# FLAT. Find the queued PR by its head branch.
num="$(gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' 2>/dev/null || true)"
if [ -z "$num" ]; then
    echo "$STAMP FLAT, but no open PR on $BRANCH (already merged, or not created) — nothing to do."
    exit 0
fi
isdraft="$(gh pr view "$num" --json isDraft --jq '.isDraft' 2>/dev/null || echo unknown)"
if [ "$isdraft" != "true" ]; then
    echo "$STAMP FLAT; PR #$num already ready — nothing to do."
    exit 0
fi

gh pr ready "$num" && echo "$STAMP FLAT — marked PR #$num ready for review."
MSG="FableBot: $INSTRUMENT is now flat — PR #$num is ready to merge (switch $INSTRUMENT to daily)."
# Notify through the same webhook the daily reports use, if configured, then a
# best-effort macOS notification. Neither failing should fail the job.
if [ -n "${FABLEBOT_WEBHOOK_URL:-}" ]; then
    curl -fsS -X POST -H 'Content-Type: application/json' \
        -d "{\"text\": \"${MSG}\"}" "$FABLEBOT_WEBHOOK_URL" >/dev/null 2>&1 || true
fi
osascript -e "display notification \"${MSG}\" with title \"FableBot\"" >/dev/null 2>&1 || true
