#!/bin/bash
set -e

echo "=============================================="
echo "OpenEnv-RL (SlideForge) Setup"
echo "=============================================="

cd "$(dirname "$0")"

# Remove existing venv if present
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create new venv
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate and upgrade pip
echo "Activating venv and upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip

# Install the package with rollouts dependencies
echo "Installing slideforge with rollouts dependencies..."
pip install -e ".[rollouts]"

# Create output directories
echo "Creating output directories..."
mkdir -p outputs

# Run test
echo ""
echo "=============================================="
echo "Running setup test..."
echo "=============================================="
python test_setup.py

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run rollouts:"
echo "  python training/rollouts.py --num-rollouts 1"
echo "=============================================="
