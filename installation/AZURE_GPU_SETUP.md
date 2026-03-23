# Azure GPU Setup Guide

This guide covers setting up Azure GPU VMs for ML training and inference when local CPU isn't sufficient.

## When You Need GPU

| Task | Local CPU OK? | GPU Recommended |
|------|---------------|-----------------|
| Phase 1 foundations (NumPy, tiny models) | ✅ Yes | No |
| Phase 2 fine-tuning (LoRA/QLoRA) | ❌ Slow | ✅ Yes |
| Phase 3 inference benchmarking | ❌ Limited | ✅ Yes |
| vLLM serving | ❌ No | ✅ Required |

**Estimated GPU hours for this curriculum:** 20-40 hours total

---

## Microsoft Employee Access Options

### Option 1: MSDN/Visual Studio Subscription
1. Go to [my.visualstudio.com](https://my.visualstudio.com)
2. Check your Azure credits (typically $150/month for VS Enterprise)
3. Activate Azure subscription if not already done

### Option 2: Internal Azure Subscription
- Ask your team lead about shared ML/GPU quotas
- Check if your org has Azure ML Studio access

### Option 3: Azure for Students/Free Tier
- $200 credit for 30 days at [azure.microsoft.com/free](https://azure.microsoft.com/free)

---

## Recommended VM Sizes

### For Fine-Tuning (Week 5)
| VM Size | GPU | VRAM | Cost/hr | Use Case |
|---------|-----|------|---------|----------|
| `Standard_NC4as_T4_v3` | 1x T4 | 16GB | ~$0.50 | Budget option |
| `Standard_NC6s_v3` | 1x V100 | 16GB | ~$3.00 | Faster training |
| `Standard_NC24ads_A100_v4` | 1x A100 | 80GB | ~$3.50 | Large models |

### For Inference Benchmarking (Week 8-9)
| VM Size | GPU | VRAM | Cost/hr | Use Case |
|---------|-----|------|---------|----------|
| `Standard_NC4as_T4_v3` | 1x T4 | 16GB | ~$0.50 | Small models, vLLM |
| `Standard_NC6s_v3` | 1x V100 | 16GB | ~$3.00 | Larger models |

**Recommendation:** Start with T4 (cheapest), upgrade if needed.

---

## Quick Setup Steps

### 1. Create VM via Azure Portal
```
1. Portal → Create a resource → Virtual Machine
2. Image: Ubuntu 22.04 LTS
3. Size: Standard_NC4as_T4_v3 (or similar)
4. Allow SSH (port 22)
5. Create and download SSH key
```

### 2. Connect to VM
```bash
ssh -i ~/azure-key.pem azureuser@<VM_IP>
```

### 3. Install NVIDIA Drivers
```bash
# Ubuntu 22.04 with NVIDIA GPU
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### 4. Install CUDA Toolkit
```bash
# After reboot
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1
```

### 5. Set Up Python Environment
```bash
# Clone your repo or copy files
git clone <your-repo>
cd becoming-an-ml-engineer

# Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install rest of dependencies
pip install transformers datasets accelerate peft trl sentencepiece
pip install wandb tensorboard evaluate scikit-learn pandas matplotlib
pip install fastapi uvicorn pydantic
pip install vllm  # Now works with GPU!
pip install jupyter black ruff pytest
```

### 6. Verify GPU Access
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Cost Management Tips

1. **Stop VM when not using** - You only pay when running
2. **Use spot instances** - Up to 90% cheaper (can be evicted)
3. **Delete disk when done** - Storage costs money too
4. **Set budget alerts** in Azure Cost Management

---

## Alternative: Azure ML Studio

If you prefer managed infrastructure:

1. Go to [ml.azure.com](https://ml.azure.com)
2. Create workspace
3. Create compute instance with GPU
4. Use Jupyter notebooks directly in browser

Pros: No VM management, pre-configured
Cons: Slightly more overhead, less control

---

## Syncing Code Between Local and Azure

### Option A: Git (recommended)
```bash
# Local: push changes
git add . && git commit -m "update" && git push

# Azure VM: pull changes
git pull
```

### Option B: VS Code Remote SSH
1. Install "Remote - SSH" extension
2. Connect to Azure VM
3. Edit files directly on VM

### Option C: rsync
```bash
rsync -avz --exclude '.venv' ./ azureuser@<VM_IP>:~/becoming-an-ml-engineer/
```
