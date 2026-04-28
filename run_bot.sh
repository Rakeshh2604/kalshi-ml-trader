#!/bin/bash
# Kalshi Paper Trading Bot — 24/7 runner with auto-restart
# Usage: bash run_bot.sh
# Stop:  Ctrl+C  (or kill the PID printed below)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/bot.log"
MAX_LOG_BYTES=52428800  # 50 MB — rotate when exceeded
RESTART_DELAY=10        # seconds to wait before restarting after a crash
PYTHON="$SCRIPT_DIR/.venv/bin/python"

echo "=========================================="
echo "  Kalshi Paper Trading Bot"
echo "  PAPER TRADING — NO REAL ORDERS"
echo "  Log: $LOG_FILE"
echo "  PID: $$"
echo "  Press Ctrl+C to stop"
echo "=========================================="

_rotate_log() {
    if [ -f "$LOG_FILE" ] && [ "$(wc -c < "$LOG_FILE")" -gt "$MAX_LOG_BYTES" ]; then
        mv "$LOG_FILE" "${LOG_FILE}.$(date +%Y%m%d_%H%M%S).bak"
        echo "$(date '+%Y-%m-%d %H:%M:%S') Log rotated" >> "$LOG_FILE"
    fi
}

_cleanup() {
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S') Bot stopped by user" | tee -a "$LOG_FILE"
    exit 0
}
trap _cleanup INT TERM

cd "$SCRIPT_DIR"

while true; do
    _rotate_log
    echo "$(date '+%Y-%m-%d %H:%M:%S') [RUNNER] Starting bot..." | tee -a "$LOG_FILE"

    set +e
    "$PYTHON" -m kalshi_bot.main 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [RUNNER] Bot exited cleanly. Restarting in ${RESTART_DELAY}s..." | tee -a "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [RUNNER] Bot crashed (exit $EXIT_CODE). Restarting in ${RESTART_DELAY}s..." | tee -a "$LOG_FILE"
    fi

    sleep $RESTART_DELAY
done
