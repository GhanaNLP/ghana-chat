#!/usr/bin/env bash
# Runner script for Ghana Chat API on NVIDIA H200 GPU instance
set -e

PORT=${1:-8000}
HOST=${2:-"0.0.0.0"}

echo "=========================================================="
echo " Starting Ghana Chat API on NVIDIA H200"
echo " Model: Qwen/Qwen3.5-2B (bfloat16 on cuda:0)"
echo " Listening on http://${HOST}:${PORT}"
echo "=========================================================="

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run uvicorn server
python -m uvicorn ghana_chat.server:app --host "${HOST}" --port "${PORT}" --workers 1
