#!/bin/bash
# run_weekly.sh — Weekly paper monitoring pipeline launcher.
#
# Designed to be invoked by cron via caffeinate:
#   0 18 * * 5 caffeinate -i /path/to/run_weekly.sh >> /path/to/cron.log 2>&1
#
# caffeinate -i prevents idle sleep only. It does NOT prevent sleep when the
# lid is closed. Run on a machine that will be open and awake on Friday at 6 PM.

set -euo pipefail

# Resolve the project root relative to this script, regardless of cwd or
# where cron calls it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/data/logs"

mkdir -p "$LOG_DIR"

echo "=== Weekly pipeline started at $(date) ==="
echo "    Project: $SCRIPT_DIR"

# --- Activate virtual environment ---
source "$VENV_DIR/bin/activate"

# --- Verify Ollama is running; start it if not ---
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is not running — attempting to start..."
    open -a Ollama
    # Give Ollama up to 30 seconds to become ready
    for i in $(seq 1 6); do
        sleep 5
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Ollama ready after $((i * 5))s"
            break
        fi
        if [ "$i" -eq 6 ]; then
            echo "FATAL: Ollama failed to start after 30s. Exiting."
            exit 1
        fi
    done
else
    echo "Ollama is already running"
fi

# --- Run the pipeline ---
cd "$SCRIPT_DIR"
python -m src.pipeline

echo "=== Weekly pipeline completed at $(date) ==="
