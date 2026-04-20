#!/bin/bash
# start.sh — start the single-process WebUI + RAG server

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)/src"
CONDA_ENV="${CONDA_ENV:-sync4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5050}"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "python.*main.py" 2>/dev/null
sleep 1

cd "$PROJECT_DIR"

# Read flash attention setting from config.yaml and export for Ollama
FLASH_ATTN=$(python3 -c "
import yaml, sys
try:
    c = yaml.safe_load(open('SearchEngine/config.yaml'))
    print('1' if c.get('ollama_flash_attention') else '0')
except Exception:
    print('0')
" 2>/dev/null || echo "0")
export OLLAMA_FLASH_ATTENTION="$FLASH_ATTN"
[ "$FLASH_ATTN" = "1" ] && echo "Flash Attention: enabled" || echo "Flash Attention: disabled"

echo "Starting WebUI + RAG on http://$HOST:$PORT"
exec uvicorn main:app --host "$HOST" --port "$PORT"
