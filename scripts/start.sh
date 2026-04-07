#!/bin/bash
# ─────────────────────────────────────────────────────────────
# start.sh — start SearchEngine + WebUI
# Usage: ./scripts/start.sh
# ─────────────────────────────────────────────────────────────

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)/src"
CONDA_ENV="sync4"
SE_PORT=5050
WEBUI_PORT=5051

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Kill any leftover servers
pkill -f "uvicorn SearchEngine.app" 2>/dev/null
pkill -f "uvicorn main:app"         2>/dev/null
sleep 1

cd "$PROJECT_DIR"

uvicorn SearchEngine.app:app --host 0.0.0.0 --port $SE_PORT \
    > /tmp/searchengine.log 2>&1 &
SE_PID=$!
echo "SearchEngine started (PID $SE_PID)"

uvicorn main:app --host 0.0.0.0 --port $WEBUI_PORT \
    > /tmp/webui.log 2>&1 &
WEBUI_PID=$!
echo "WebUI started (PID $WEBUI_PID)"

echo ""
echo "Waiting for servers..."
for i in $(seq 1 20); do
    sleep 1
    curl -s "http://localhost:$SE_PORT/health"    > /dev/null 2>&1 && \
    curl -s "http://localhost:$WEBUI_PORT/health" > /dev/null 2>&1 && \
    break
    echo "  ($i) still starting..."
done

echo ""
echo "Open: http://localhost:$WEBUI_PORT"
echo "Logs: tail -f /tmp/searchengine.log /tmp/webui.log"
echo ""
echo "Ctrl+C to stop both servers"

trap "echo 'Stopping...'; kill $SE_PID $WEBUI_PID 2>/dev/null; exit" INT TERM
wait
