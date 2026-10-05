#!/usr/bin/env bash
# Supervisor for the Ghana Chat API.
# Restarts Uvicorn automatically if the process dies, so the Space never
# serves a dead endpoint after an unexpected crash or OOM.
set -uo pipefail

PROJ="/mnt/volume_d2wey28/projects/ghana-chat"
VENV="/mnt/volume_d2wey28/projects/map-nav/.venv/bin/activate"
LOG="$PROJ/server.log"

cd "$PROJ" || exit 1
# shellcheck disable=SC1090
source "$VENV"
export HF_HOME="/mnt/volume_d2wey28/hf_cache"
export PYTHONPATH="$PROJ"

while true; do
    echo "[supervisor] starting uvicorn at $(date -Is)" >> "$LOG"
    python -m uvicorn ghana_chat.server:app \
        --host 0.0.0.0 --port 8000 \
        --timeout-keep-alive 75 \
        >> "$LOG" 2>&1
    code=$?
    echo "[supervisor] uvicorn exited with code $code at $(date -Is); restarting in 5s" >> "$LOG"
    sleep 5
done
