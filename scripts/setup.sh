#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — First-time setup for AIIAB
# Run once after cloning to build the webui and download WebLLM assets.
#
# Usage:
#   bash scripts/setup.sh [model_id]
#
# Optional model_id (default: SmolLM2-360M-Instruct-q4f16_1-MLC):
#   SmolLM2-135M-Instruct-q4f16_1-MLC   ~200MB RAM, fastest,  minimal quality
#   SmolLM2-360M-Instruct-q4f16_1-MLC   ~400MB RAM, ~40tok/s, recommended
#   Qwen2.5-0.5B-Instruct-q4f16_1-MLC   ~500MB RAM, ~35tok/s, better reasoning
#   Phi-3.5-mini-instruct-q4f16_1-MLC   ~2.5GB RAM, ~15tok/s, strong (GPU only)
# ─────────────────────────────────────────────────────────────────────────────

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="${1:-SmolLM2-360M-Instruct-q4f16_1-MLC}"

echo "=== AIIAB Setup ==="
echo "Repo: $REPO_ROOT"
echo "Model: $MODEL_ID"
echo ""

# ── 1. Build the webui (creates webui/_app/) ──────────────────────────────────
echo "[1/2] Building webui…"
bash "$REPO_ROOT/scripts/build-webui.sh"
echo ""

# ── 2. Download WebLLM JS + MLC model (creates webui/webllm/ and webui/models/)
echo "[2/2] Downloading WebLLM assets…"
bash "$REPO_ROOT/scripts/download_webllm.sh" "$MODEL_ID"
echo ""

echo "=== Setup complete ==="
echo "Start the server with:"
echo "  cd src && python main.py"
