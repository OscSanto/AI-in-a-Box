#!/bin/bash
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate sync4
cd "$(dirname "$0")/../src"
uvicorn main:app --host 0.0.0.0 --port 5051
