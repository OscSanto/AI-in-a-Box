#!/bin/bash
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate sync4
export HF_HUB_OFFLINE=1   # never phone home — model must be cached locally
cd "$(dirname "$0")/../src"
uvicorn SearchEngine.app:app --host 0.0.0.0 --port 5050
