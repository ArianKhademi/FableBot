#!/usr/bin/env bash
# Run the paper books in a tmux session — one window per bot, so they survive
# closing the terminal (and a laptop sleep/reconnect) instead of dying with it.
#
#   scripts/bots.sh start          # start every bot not already running
#   scripts/bots.sh status         # what's running, and is each one writing?
#   scripts/bots.sh attach         # watch them (detach again with: Ctrl-b then d)
#   scripts/bots.sh stop           # stop all bots (kills the session)
#   scripts/bots.sh stop trail     # stop one bot by name
#   scripts/bots.sh restart mr     # restart one bot by name
#   scripts/bots.sh logs futures   # tail one bot's log
#
# Each bot keeps its own tmux WINDOW (named) and its own log file, so nothing is
# ever "blindly killed" — `status` names every bot, whether its window is alive,
# and when it last wrote to its journal.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SESSION="${FABLEBOT_TMUX_SESSION:-fablebot}"
PY="${FABLEBOT_PYTHON:-python3}"
mkdir -p logs

# name|config|state dir|log
BOTS=(
  "mr|config_experiment.yaml|state_experiment|bot_mr.log"
  "trail|config_experiment_trail.yaml|state_experiment_trail|bot_trail.log"
  "regime|config_experiment_regime.yaml|state_experiment_regime|bot_regime.log"
  "breaker|config_experiment_adaptive.yaml|state_experiment_adaptive|bot_breaker.log"
  "tfboth|config_experiment_trendfilter.yaml|state_experiment_trendfilter|bot_trendfilter.log"
  "tfshorts|config_experiment_trendfilter_shorts.yaml|state_experiment_trendfilter_shorts|bot_trendshorts.log"
  "futures|config_experiment_futures.yaml|state_experiment_futures|bot_futures.log"
  "futuresrt|config_experiment_futures_rt.yaml|state_experiment_futures_rt|bot_futures_rt.log"
  "observed|config_experiment_observed.yaml|state_experiment_observed|bot_observed.log"
)

have_tmux() { command -v tmux >/dev/null 2>&1 || { echo "tmux not found — brew install tmux"; exit 1; }; }
session_exists() { tmux has-session -t "$SESSION" 2>/dev/null; }
window_exists() { tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx "$1"; }

start_one() {
  local name=$1 config=$2 state=$3 log=$4
  if window_exists "$name"; then echo "  $name: already running"; return; fi
  local cmd="cd '$ROOT' && $PY -u scripts/run_bot.py --config $config --state-dir $state 2>&1 | tee -a logs/$log"
  if session_exists; then
    tmux new-window -t "$SESSION" -n "$name" "$cmd"
  else
    tmux new-session -d -s "$SESSION" -n "$name" "$cmd"
  fi
  echo "  $name: started"
}

cmd_start() {
  have_tmux
  local only="${1:-}"
  echo "starting bots in tmux session '$SESSION':"
  for row in "${BOTS[@]}"; do
    IFS='|' read -r name config state log <<< "$row"
    [ -n "$only" ] && [ "$only" != "$name" ] && continue
    start_one "$name" "$config" "$state" "$log"
  done
  echo
  echo "attach with: scripts/bots.sh attach     (detach: Ctrl-b then d)"
}

cmd_status() {
  have_tmux
  # Always name every bot and show last journal write — even when the session
  # (and therefore every window) is down. WINDOW will just read "down".
  if ! session_exists; then
    echo "session '$SESSION' is not running (no bots up)"
  fi
  printf "%-11s %-9s %-22s %s\n" BOT WINDOW "LAST JOURNAL WRITE" STATE_DIR
  for row in "${BOTS[@]}"; do
    IFS='|' read -r name config state log <<< "$row"
    local win="down" wrote="-"
    session_exists && window_exists "$name" && win="UP"
    [ -f "$state/journal.jsonl" ] && wrote=$(date -r "$state/journal.jsonl" '+%Y-%m-%d %H:%M:%S')
    printf "%-11s %-9s %-22s %s\n" "$name" "$win" "$wrote" "$state"
  done
  echo
  echo "(WINDOW=UP means the tmux window is alive; LAST JOURNAL WRITE proves it's actually working)"
}

cmd_stop() {
  have_tmux
  local only="${1:-}"
  if [ -z "$only" ]; then
    session_exists && tmux kill-session -t "$SESSION" && echo "stopped all bots (session '$SESSION' killed)" \
      || echo "session '$SESSION' was not running"
  else
    window_exists "$only" && tmux kill-window -t "$SESSION:$only" && echo "stopped $only" \
      || echo "$only is not running"
  fi
}

# Resolve a bot name to its log basename from BOTS; empty if unknown.
lookup_log() {
  local want=$1 n c s l
  for row in "${BOTS[@]}"; do
    IFS='|' read -r n c s l <<< "$row"
    if [ "$n" = "$want" ]; then
      echo "$l"
      return 0
    fi
  done
  return 1
}

cmd_logs() {
  local name="${1:-}"
  if [ -z "$name" ]; then
    echo "usage: scripts/bots.sh logs <bot>" >&2
    echo "known bots: $(printf '%s\n' "${BOTS[@]}" | cut -d'|' -f1 | tr '\n' ' ')" >&2
    exit 1
  fi
  local log
  if ! log=$(lookup_log "$name"); then
    echo "unknown bot: $name" >&2
    echo "known bots: $(printf '%s\n' "${BOTS[@]}" | cut -d'|' -f1 | tr '\n' ' ')" >&2
    exit 1
  fi
  local path="logs/$log"
  if [ ! -f "$path" ]; then
    echo "no log yet for $name ($path) — has it been started?" >&2
    exit 1
  fi
  tail -f "$path"
}

case "${1:-status}" in
  start)   cmd_start "${2:-}" ;;
  stop)    cmd_stop  "${2:-}" ;;
  restart) cmd_stop "${2:-}"; sleep 1; cmd_start "${2:-}" ;;
  status)  cmd_status ;;
  attach)  have_tmux; tmux attach -t "$SESSION" ;;
  logs)    cmd_logs "${2:-}" ;;
  *) sed -n '2,18p' "$0" ;;
esac
