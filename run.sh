#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================================="
echo "Starting CrimeVision AI Forensic Command Center..."
echo "==================================================="

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "[WARNING] Virtual environment not found in venv/ or .venv/. Using system Python..."
fi

python app.py
