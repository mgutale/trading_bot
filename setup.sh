#!/bin/bash
# Fresh setup script for Trading Bot
# Creates virtual environment, installs dependencies, and runs backtest

set -e

echo "========================================"
echo "  Trading Bot - Fresh Setup"
echo "========================================"

# Remove existing venv if present
if [ -d "venv" ]; then
    echo "[1/5] Removing existing virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[3/5] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[4/6] Installing dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo "[5/6] Installing trading_bot package..."
pip install -e .

# Verify installation
echo "[6/6] Verifying installation..."
python -c "import pandas; import numpy; import hmmlearn; import yfinance; from trading_bot.main import main; print('All dependencies installed successfully!')"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To run a 1-year backtest:"
echo "  source venv/bin/activate"
echo "  python -m trading_bot.main backtest --years 1"
echo ""
