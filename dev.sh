#!/usr/bin/env bash
set -euo pipefail

SESSION="rlmdev"
DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Helpers ─────────────────────────────────────────────────────────────────
kill_port() {
  lsof -ti tcp:"$1" | xargs kill -9 2>/dev/null || true
}

# ── Kill existing session / stale server ────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true
kill_port 3000
sleep 1

# ── Start the RLM demo API server ───────────────────────────────────────────
tmux new-session -d -s "$SESSION" -n api \
  "cd $DIR && source .venv/bin/activate && uvicorn demo.backend.fastapi_server:app --host 0.0.0.0 --port 3000 --reload; read"

echo "RLM demo API server starting on http://localhost:3000"
echo "Attach with: tmux attach -t $SESSION"

# ── Attach if running interactively ─────────────────────────────────────────
if [ -t 0 ]; then
  tmux attach -t "$SESSION"
fi
