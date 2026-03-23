#!/bin/bash
# ML Engineering Environment Setup Script for WSL (CPU-only)
# Usage: bash installation/setup.sh

set -e

echo "=== ML Engineering Environment Setup ==="

# Check if running in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Warning: This script is designed for WSL. Proceeding anyway..."
fi

# Install python venv if needed
if ! python3 -m venv --help &>/dev/null; then
    echo "Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3.10-venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch (CPU)
echo "Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Hugging Face ecosystem
echo "Installing Hugging Face ecosystem..."
pip install transformers datasets accelerate peft trl sentencepiece

# Install experiment tracking and data tools
echo "Installing experiment tracking and data tools..."
pip install wandb tensorboard evaluate scikit-learn pandas matplotlib

# Install serving stack and dev tools
echo "Installing serving stack and dev tools..."
pip install fastapi uvicorn pydantic jupyter black ruff pytest

# Generate lock file
echo "Generating requirements.txt..."
pip freeze > requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers, datasets, peft, trl, wandb, fastapi; print('All key imports successful')"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source .venv/bin/activate"
