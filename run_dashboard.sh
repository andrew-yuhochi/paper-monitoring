#!/bin/bash
# run_dashboard.sh — Launch the Streamlit paper monitoring dashboard.
#
# Usage:  bash run_dashboard.sh
# Opens:  http://localhost:8501

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

source "$VENV_DIR/bin/activate"

echo "Starting Paper Monitoring dashboard at http://localhost:8501"
cd "$SCRIPT_DIR"
streamlit run src/dashboard/app.py \
    --server.headless false \
    --server.port 8501
