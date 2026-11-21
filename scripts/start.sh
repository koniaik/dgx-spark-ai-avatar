#!/usr/bin/env bash
# Start the avatar stack: Speaches container + OpenAvatarChat demo.
# Usage: start.sh [path-to-config.yaml]   (default: configs-local/chat_pl_liteavatar.yaml)
# Places this script (and stop.sh) in the OpenAvatarChat repo root if you run it as a service.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$ROOT/configs-local/chat_en_liteavatar.yaml}"
LOG_DIR="$ROOT/logs"
PIDFILE="$LOG_DIR/demo.pid"
mkdir -p "$LOG_DIR"

# already running?
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDFILE"))"
  exit 0
fi

# Speaches first; wait until it answers (container may be stopped or still loading models)
docker start speaches >/dev/null 2>&1 || true
for _ in $(seq 1 90); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  sleep 2
done

cd "$ROOT"
UV="$HOME/.local/bin/uv"
[ -x "$UV" ] || UV="$(command -v uv)"

# own process group, survive the parent shell; log to file; pid tracked in a file
# (never pkill -f over ssh — it matches its own command line)
setsid nohup "$UV" run src/demo.py --config "$CONFIG" >> "$LOG_DIR/demo.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE"); log: $LOG_DIR/demo.log"
