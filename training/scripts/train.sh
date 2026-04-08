#!/bin/bash
# Train LSTM model (Linux/macOS)

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        echo "Error: Virtual environment not found. Run setup.sh first."
        exit 1
    fi
fi

# Run training
echo "Starting LSTM training..."
python -m training.train "$@"
