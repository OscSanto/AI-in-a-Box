#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_llamacpp.sh — Install and configure llama.cpp for AIIAB
#
# What this does:
#   1. Installs build dependencies
#   2. Clones and compiles llama.cpp with ARM NEON optimizations (or downloads
#      a prebuilt binary if --prebuilt flag is passed)
#   3. Installs llama-server to /usr/local/bin
#   4. Creates the model directory
#   5. Optionally downloads a starter GGUF model
#   6. Patches config.yaml to enable llamacpp backend
#
# Usage:
#   bash scripts/setup_llamacpp.sh [options]
#
# Options:
#   --prebuilt        Download prebuilt ARM64 binary instead of compiling
#   --model TAG       Download a starter model (e.g. qwen2.5-1.5b, qwen2.5-3b)
#                     Supported: qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b,
#                                llama3.2-1b, llama3.2-3b, smollm2-1.7b
#   --model-dir PATH  Where to store GGUF files (default: /library/models)
#   --threads N       CPU threads for llama-server (default: 4)
#   --ctx N           Context window size (default: 2048)
#   --dry-run         Print what would happen without doing it
#
# Examples:
#   bash scripts/setup_llamacpp.sh --prebuilt --model qwen2.5-1.5b
#   bash scripts/setup_llamacpp.sh --model qwen2.5-3b --threads 4 --ctx 2048
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_YAML="$REPO_ROOT/src/SearchEngine/config.yaml"

# ── Defaults ─────────────────────────────────────────────────────────────────
PREBUILT=false
MODEL_TAG=""
MODEL_DIR="/library/models"
THREADS=4
CTX_SIZE=2048
DRY_RUN=false
BUILD_DIR="$HOME/llama.cpp"

# ── Starter model registry (HuggingFace GGUF URLs) ───────────────────────────
declare -A MODEL_URLS=(
    ["qwen2.5-0.5b"]="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ["qwen2.5-1.5b"]="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    ["qwen2.5-3b"]="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    ["llama3.2-1b"]="https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf"
    ["llama3.2-3b"]="https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf"
    ["smollm2-1.7b"]="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prebuilt)   PREBUILT=true ;;
        --model)      MODEL_TAG="$2"; shift ;;
        --model-dir)  MODEL_DIR="$2"; shift ;;
        --threads)    THREADS="$2"; shift ;;
        --ctx)        CTX_SIZE="$2"; shift ;;
        --dry-run)    DRY_RUN=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
info()  { echo "  [•] $*"; }
ok()    { echo "  [✓] $*"; }
warn()  { echo "  [!] $*"; }
run()   { if $DRY_RUN; then echo "  [dry] $*"; else "$@"; fi; }

echo ""
echo "=== AIIAB llama.cpp Setup ==="
echo ""

# ── 1. Build or download llama-server ─────────────────────────────────────────
if command -v llama-server &>/dev/null && llama-server --version &>/dev/null; then
    ok "llama-server already installed at $(command -v llama-server)"
elif $PREBUILT; then
    _ARCH="$(uname -m)"
    case "$_ARCH" in
        aarch64|arm64) _ARCH_TAG="arm64" ;;
        x86_64)        _ARCH_TAG="x64"   ;;
        *) echo "Unsupported arch: $_ARCH — use source build instead"; exit 1 ;;
    esac
    info "Downloading prebuilt llama-server binary for $_ARCH_TAG..."
    RELEASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/b5186/llama-b5186-bin-ubuntu-${_ARCH_TAG}.zip"
    TMP_DIR=$(mktemp -d)
    run curl -fsSL "$RELEASE_URL" -o "$TMP_DIR/llama.zip"
    run unzip -q "$TMP_DIR/llama.zip" -d "$TMP_DIR"
    run sudo mv "$TMP_DIR/build/bin/llama-server" /usr/local/bin/llama-server
    run sudo chmod +x /usr/local/bin/llama-server
    run rm -rf "$TMP_DIR"
    ok "llama-server installed"
else
    info "Compiling llama.cpp from source with ARM NEON optimizations..."
    info "This takes ~10–15 minutes on Pi 5. Get a coffee."
    echo ""

    run sudo apt-get update -qq
    run sudo apt-get install -y cmake build-essential git

    if [[ ! -d "$BUILD_DIR" ]]; then
        run git clone --depth 1 https://github.com/ggml-org/llama.cpp "$BUILD_DIR"
    else
        info "llama.cpp repo already exists at $BUILD_DIR, pulling latest..."
        run git -C "$BUILD_DIR" pull --ff-only
    fi

    run cmake -B "$BUILD_DIR/build" "$BUILD_DIR" \
        -DLLAMA_NATIVE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF
    run cmake --build "$BUILD_DIR/build" --target llama-server -j4
    run sudo mv "$BUILD_DIR/build/bin/llama-server" /usr/local/bin/llama-server
    run sudo chmod +x /usr/local/bin/llama-server
    ok "llama-server compiled and installed"
fi

echo ""

# ── 2. Create model directory ─────────────────────────────────────────────────
info "Model directory: $MODEL_DIR"
run sudo mkdir -p "$MODEL_DIR"
run sudo chown "$(whoami)" "$MODEL_DIR"
ok "Model directory ready"

echo ""

# ── 3. Download starter model ──────────────────────────────────────────────────
MODEL_FILE=""
if [[ -n "$MODEL_TAG" ]]; then
    if [[ -z "${MODEL_URLS[$MODEL_TAG]+_}" ]]; then
        warn "Unknown model tag '$MODEL_TAG'. Available: ${!MODEL_URLS[*]}"
        warn "Skipping model download."
    else
        MODEL_URL="${MODEL_URLS[$MODEL_TAG]}"
        MODEL_FILE="$(basename "$MODEL_URL")"
        MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

        if [[ -f "$MODEL_PATH" ]]; then
            ok "Model already exists: $MODEL_FILE"
        else
            info "Downloading $MODEL_TAG (~1–2 GB, this will take a while)..."
            run curl -fL --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
            ok "Downloaded: $MODEL_FILE"
        fi
    fi
else
    warn "No --model specified. Add GGUF files to $MODEL_DIR manually."
    warn "Example: bash scripts/setup_llamacpp.sh --model qwen2.5-1.5b"
fi

echo ""

# ── 4. Patch config.yaml ───────────────────────────────────────────────────────
info "Updating config.yaml..."

if $DRY_RUN; then
    echo "  [dry] Would set backend=llamacpp, model_dir=$MODEL_DIR, threads=$THREADS, ctx_size=$CTX_SIZE"
    [[ -n "$MODEL_FILE" ]] && echo "  [dry] Would set model=$MODEL_FILE"
else
    _MODEL_LINE=""
    [[ -n "$MODEL_FILE" ]] && _MODEL_LINE="text = replace_or_add(text, \"  model\", '\"$MODEL_FILE\"')"

    python3 - <<PYEOF
import re

path = "$CONFIG_YAML"
with open(path) as f:
    text = f.read()

def replace_or_add(text, key, value):
    pattern = rf'^({re.escape(key)}:\s*).*$'
    replacement = rf'\g<1>{value}'
    new, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    return new

text = replace_or_add(text, "backend", '"llamacpp"')
text = replace_or_add(text, "  url", '"http://localhost:8080"')
text = replace_or_add(text, "  model_dir", '"$MODEL_DIR"')
text = replace_or_add(text, "  threads", $THREADS)
text = replace_or_add(text, "  ctx_size", $CTX_SIZE)
$_MODEL_LINE

with open(path, "w") as f:
    f.write(text)
print("config.yaml updated")
PYEOF
fi

ok "config.yaml updated"

echo ""

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo "=== Setup complete ==="
echo ""
echo "  Backend:    llama.cpp"
echo "  Model dir:  $MODEL_DIR"
echo "  Threads:    $THREADS"
echo "  Context:    $CTX_SIZE tokens"
[[ -n "$MODEL_FILE" ]] && echo "  Model:      $MODEL_FILE"
echo ""
echo "Start the server:"
echo "  bash scripts/start.sh"
echo ""
echo "To switch back to Ollama, set in config.yaml:"
echo "  backend: \"ollama\""
echo ""
