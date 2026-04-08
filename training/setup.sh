#!/bin/bash
set -e

echo "========================================"
echo "Ferromode Training - Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python: $PYTHON_VERSION"

if ! python -c 'import sys; exit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "❌ Error: Python 3.12+ required"
    echo "Current version: $PYTHON_VERSION"
    echo "Install from: https://www.python.org/downloads/"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python -m venv .venv
    echo "✓ Virtual environment created"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"

echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip upgraded"

echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "✓ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run training: bash scripts/train.sh"
echo ""
