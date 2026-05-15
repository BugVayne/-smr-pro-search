#!/usr/bin/env bash
# Start all microservices + UI in the background.
#
# Usage:
#   bash scripts/run_all.sh         # foreground (logs to terminal)
#   bash scripts/run_all.sh stop    # stop everything

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PIDS_FILE="$ROOT/.run_pids"
LOG_DIR="$ROOT/data/logs"
mkdir -p "$LOG_DIR"

# Make sure Python can import "common", "services", "training"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

start_service() {
    local name="$1"
    local module="$2"
    local logfile="$LOG_DIR/${name}.log"
    echo "Starting $name → $logfile"
    python -m "$module" >> "$logfile" 2>&1 &
    echo $! >> "$PIDS_FILE"
}

stop_all() {
    if [[ -f "$PIDS_FILE" ]]; then
        echo "Stopping services…"
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" || true
            fi
        done < "$PIDS_FILE"
        rm -f "$PIDS_FILE"
    fi
    echo "Done."
}

if [[ "${1:-start}" == "stop" ]]; then
    stop_all
    exit 0
fi

# Idempotent restart
stop_all
: > "$PIDS_FILE"

start_service preprocessing  services.preprocessing.app
start_service retrieval      services.retrieval.app
start_service ranking        services.ranking.app
start_service postprocessing services.postprocessing.app
start_service gateway        services.gateway.app
start_service ui             ui.app

echo
echo "All services started. PIDs in $PIDS_FILE, logs in $LOG_DIR/"
echo "UI:      http://localhost:8080"
echo "Gateway: http://localhost:5000"
echo
echo "Stop with: bash scripts/run_all.sh stop"
