#!/usr/bin/env bash
# Stop the avatar stack: OpenAvatarChat demo (pidfile / port fallback) + Speaches container.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
PIDFILE="$LOG_DIR/demo.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  # kill the whole process group (setsid leader), then the pid itself
  PGID="$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || true)"
  if [ -n "${PGID:-}" ]; then kill -- "-$PGID" 2>/dev/null || true; fi
  kill "$PID" 2>/dev/null || true
  rm -f "$PIDFILE"
fi

# fallback: whatever still listens on the demo port
PORT="${AVATAR_PORT:-8899}"
PID_ON_PORT="$(ss -tlnp 2>/dev/null | awk -v p=":${PORT} " '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
if [ -n "${PID_ON_PORT:-}" ]; then
  kill "$PID_ON_PORT" 2>/dev/null || true
fi

docker stop speaches >/dev/null 2>&1 || true
echo "stopped"
