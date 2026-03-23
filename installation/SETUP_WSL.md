# ML Engineering Environment Setup Guide (WSL)

This guide documents how to set up a reproducible Python environment for ML engineering study on WSL (Windows Subsystem for Linux) without a dedicated NVIDIA GPU.

## Prerequisites

- WSL 2 with Ubuntu 22.04 (or similar Debian-based distro)
- Python 3.10+
- No NVIDIA GPU required (CPU-only setup)

## Hardware Check

### Check for NVIDIA GPU
```bash
nvidia-smi
```
If this returns "command not found" or no GPU info, you have no NVIDIA GPU available.

### Check for WSL GPU paravirtualization
```bash
ls /dev/dxg
```
If `/dev/dxg` exists, WSL GPU support is available but this is for DirectX compute, not CUDA.

## Environment Setup

### 1. Install Python venv support
```bash
sudo apt update
sudo apt install -y python3.10-venv
```

### 2. Create virtual environment
```bash
cd /path/to/becoming-an-ml-engineer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install PyTorch (CPU-only)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Hugging Face ecosystem
```bash
pip install transformers datasets accelerate peft trl sentencepiece
```

### 5. Install experiment tracking and data tools
```bash
pip install wandb tensorboard evaluate scikit-learn pandas matplotlib
```

### 6. Install serving stack and dev tools
```bash
pip install fastapi uvicorn pydantic jupyter black ruff pytest
```

### 7. Generate lock file
```bash
pip freeze > requirements.txt
```

## Verification

### Test PyTorch installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); x = torch.randn(3,3); print(f'Tensor test passed: {x.sum().item():.4f}')"
```

Expected output:
```
PyTorch: 2.11.0+cpu
CUDA available: False
Tensor test passed: <some float>
```

### Test key imports
```bash
python -c "import transformers, datasets, peft, trl, wandb, fastapi; print('All imports successful')"
```

## Daily Usage

Activate the environment before working:
```bash
cd /path/to/becoming-an-ml-engineer
source .venv/bin/activate
```

## Installed Packages Summary

| Category | Packages |
|----------|----------|
| Deep Learning | torch, torchvision, torchaudio (CPU) |
| LLM Tools | transformers, datasets, accelerate, peft, trl, sentencepiece |
| Experiment Tracking | wandb, tensorboard |
| Evaluation | evaluate, scikit-learn |
| Data | pandas, numpy, matplotlib |
| Serving | fastapi, uvicorn, pydantic |
| Dev Tools | jupyter, black, ruff, pytest |

## Notes

- **vLLM is NOT installed** - requires CUDA GPU
- **flash-attn, deepspeed, bitsandbytes are NOT installed** - require CUDA GPU
- For GPU-dependent work (fine-tuning, inference benchmarking), use cloud compute:
  - [Google Colab](https://colab.research.google.com/) - free tier
  - [Modal](https://modal.com/) - generous free tier
  - [RunPod](https://www.runpod.io/) - pay-per-hour
  - [Vast.ai](https://vast.ai/) - pay-per-hour
  - [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) - pay-per-hour

## Troubleshooting

### "ensurepip is not available"
```bash
sudo apt install python3.10-venv
```

### PyTorch can't find CUDA
This is expected on CPU-only setup. The environment is configured for CPU training which works for:
- Phase 1 foundations (optimization from scratch, tiny models)
- Understanding transformers and training loops
- Building evaluation pipelines
- RAG and application serving

Use cloud GPU for Phase 2+ work requiring actual model fine-tuning.
