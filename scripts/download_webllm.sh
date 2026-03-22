#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_webllm.sh
# Run once (with internet) to download WebLLM JS and the SmolLM2 MLC model.
# After this, everything is served from the IIAB server — no internet needed.
#
# Usage:
#   cd /path/to/IIAB
#   bash scripts/download_webllm.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEBLLM_DIR="$REPO_ROOT/webui/webllm"
MODELS_DIR="$REPO_ROOT/webui/models"
# Default model — change to one of the alternatives below if needed:
#   SmolLM2-135M-Instruct-q4f16_1-MLC  ~200MB RAM, fastest,  minimal quality
#   SmolLM2-360M-Instruct-q4f16_1-MLC  ~400MB RAM, ~40tok/s, recommended
#   Qwen2.5-0.5B-Instruct-q4f16_1-MLC  ~500MB RAM, ~35tok/s, better reasoning
#   Phi-3.5-mini-instruct-q4f16_1-MLC  ~2.5GB RAM, ~15tok/s, strong (GPU only)
MODEL_ID="${1:-SmolLM2-360M-Instruct-q4f16_1-MLC}"
MODEL_REPO="mlc-ai/$MODEL_ID"

echo "=== IIAB WebLLM offline asset downloader ==="
echo "WebLLM dir : $WEBLLM_DIR"
echo "Models dir : $MODELS_DIR"
echo ""

# ── 1. Download WebLLM JS bundle from npm ─────────────────────────────────────
echo "[1/2] Downloading @mlc-ai/web-llm npm package…"

if command -v npm &>/dev/null; then
    # Use npm to get the exact package, then copy dist/
    TMP_DIR=$(mktemp -d)
    npm install --prefix "$TMP_DIR" @mlc-ai/web-llm --no-save --silent
    PKG="$TMP_DIR/node_modules/@mlc-ai/web-llm"
    SRC=$([ -d "$PKG/dist" ] && echo "$PKG/dist" || ([ -d "$PKG/lib" ] && echo "$PKG/lib" || echo "$PKG"))
    mkdir -p "$WEBLLM_DIR"
    cp -r "$SRC/." "$WEBLLM_DIR/"
    rm -rf "$TMP_DIR"
    echo "    WebLLM JS installed from npm."
else
    # Fallback: download bundled ESM from unpkg
    echo "    npm not found, downloading single bundle from unpkg…"
    curl -fsSL "https://unpkg.com/@mlc-ai/web-llm/dist/webllm.js" -o "$WEBLLM_DIR/webllm.js"
    # Also fetch the WASM file it references
    WASM_URL=$(curl -fsSL "https://unpkg.com/@mlc-ai/web-llm/dist/" | grep -oP 'tvmjs[^"]+\.wasm' | head -1)
    if [ -n "$WASM_URL" ]; then
        curl -fsSL "https://unpkg.com/@mlc-ai/web-llm/dist/$WASM_URL" -o "$WEBLLM_DIR/$WASM_URL"
    fi
    echo "    WebLLM JS downloaded from unpkg."
fi

# ── 2. Download MLC model from HuggingFace ────────────────────────────────────
echo ""
echo "[2/2] Downloading MLC model: $MODEL_ID…"
echo "      (SmolLM2-360M ~400MB, 135M ~200MB, Qwen2.5-0.5B ~500MB)"
MODEL_DIR="$MODELS_DIR/$MODEL_ID"
mkdir -p "$MODEL_DIR"

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
elif command -v git &>/dev/null && git lfs version &>/dev/null 2>&1; then
    git clone "https://huggingface.co/$MODEL_REPO" "$MODEL_DIR"
else
    # Manual file-by-file download using Python
    python3 - <<PYEOF
import urllib.request, json, os

repo = "$MODEL_REPO"
dest = "$MODEL_DIR"
api  = f"https://huggingface.co/api/models/{repo}"

print(f"  Fetching file list from HuggingFace API…")
with urllib.request.urlopen(api) as r:
    info = json.loads(r.read())

files = [s["rfilename"] for s in info.get("siblings", [])]
print(f"  Files to download: {len(files)}")

for fname in files:
    url   = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    fpath = os.path.join(dest, fname)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    print(f"  → {fname}")
    urllib.request.urlretrieve(url, fpath)

print("  Model download complete.")
PYEOF
fi

echo ""
echo "=== Done ==="
echo "WebLLM assets are in: $WEBLLM_DIR"
echo "Model files are in  : $MODELS_DIR/$MODEL_ID"
echo ""
echo "Restart the IIAB server and visit http://<server>:5050/exp01/"
echo ""
echo "To download a different model, pass the model ID as an argument:"
echo "  bash scripts/download_webllm.sh SmolLM2-135M-Instruct-q4f16_1-MLC"
echo "  bash scripts/download_webllm.sh Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
