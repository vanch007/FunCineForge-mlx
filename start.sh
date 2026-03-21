#!/bin/bash
# FunCineForge MLX WebUI — one-click start script
# Usage: bash start.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "🎬 FunCineForge MLX WebUI"
echo "========================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate FunCineForge

# Prevent MPS out-of-memory SIGKILL
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Launch WebUI
echo "🚀 Starting WebUI on http://localhost:7860"
echo ""
python app.py
