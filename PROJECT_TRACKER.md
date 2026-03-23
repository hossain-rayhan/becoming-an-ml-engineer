# ML Engineer Study Plan - Project Tracker

**Duration:** 12 weeks

---

## Stack Decisions (Phase 0)

| Component | Choice | Reason |
|-----------|--------|--------|
| Local Environment | CPU-only (WSL) | ThinkPad P16s has no NVIDIA GPU |
| GPU Compute | Azure VMs | Microsoft employee benefits |
| Experiment Tracker | TensorBoard | Local-first, no signup required |
| Serving Stack | FastAPI + Transformers | Works on CPU; vLLM for Azure GPU later |
| Base Model Family | Qwen | Best small models (0.5B-4B) for limited compute |
| Python Version | 3.10 | Stable, good library support |

---

## Portfolio Projects (max 3)

| # | Project | Phase | Status |
|---|---------|-------|--------|
| 1 | Minimal GPT / Transformer from Scratch | Phase 1 (Week 3) | Not Started |
| 2 | Fine-Tuning + Evaluation Pipeline | Phase 2 (Week 5-6) | Not Started |
| 3 | Inference/RAG System with Observability | Phase 3 (Week 8-9) | Not Started |

---

## Weekly Progress

### Phase 0: Setup (Days 1-3)
- [x] Set up reproducible Python environment
- [x] Confirm PyTorch works (CPU)
- [x] Create requirements lock file
- [x] Choose experiment tracker: TensorBoard
- [x] Choose serving stack: FastAPI + Transformers
- [x] Choose base model: Qwen
- [x] Plan GPU compute: Azure
- [ ] Verify Azure GPU access (when needed in Week 5)

### Phase 1: Foundations (Weeks 1-3)

#### Week 1: Math and Training Intuition
- [ ] Review linear algebra basics (3Blue1Brown or skip if strong)
- [ ] Review gradients, chain rule
- [ ] Implement linear regression from scratch (NumPy)
- [ ] Implement logistic regression from scratch (NumPy)
- [ ] Plot learning rate sensitivity curves
- [ ] Understand: vanishing/exploding gradients, softmax+cross-entropy

#### Week 2: Neural Network Mechanics
- [ ] Watch Karpathy Neural Networks Zero to Hero (implementation parts)
- [ ] Build minimal training loop in PyTorch
- [ ] Train on MNIST or CIFAR-10
- [ ] Experiment: SGD vs Adam
- [ ] Understand: initialization, regularization, batch training

#### Week 3: Attention and Transformers
- [ ] Read "Attention Is All You Need" paper
- [ ] Watch Karpathy "Let's Build GPT"
- [ ] Implement multi-head attention from scratch
- [ ] Implement transformer block (MLP, residuals, norm)
- [ ] Train tiny language model on small corpus
- [ ] **Deliverable: Project 1 - Minimal GPT repo**

### Phase 2: Practical LLM Engineering (Weeks 4-7)

#### Week 4: Modern LLM Architecture
- [ ] Read LLaMA paper
- [ ] Understand RoPE, RMSNorm, GQA
- [ ] Compare Qwen/Llama/Mistral architectures
- [ ] Trace model config → memory/latency

#### Week 5: Fine-Tuning (Azure GPU needed)
- [ ] Prepare instruction dataset
- [ ] Set up Azure GPU VM
- [ ] Fine-tune Qwen with LoRA/QLoRA
- [ ] Track with TensorBoard
- [ ] Document hyperparameters, cost, failures

#### Week 6: Evaluation Pipeline
- [ ] Build benchmark suite for fine-tuned model
- [ ] Add safety evals
- [ ] Record before/after outputs
- [ ] Create model card
- [ ] **Deliverable: Project 2 - Fine-tuning + Eval repo**

#### Week 7: Alignment Awareness
- [ ] Read InstructGPT, Constitutional AI, DPO papers
- [ ] Document RLHF/DPO pipeline design
- [ ] Understand reward hacking, specification errors

### Phase 3: Systems Engineering (Weeks 8-10)

#### Week 8: Inference Performance (Azure GPU)
- [ ] Benchmark latency, throughput, memory
- [ ] Understand KV cache, batching
- [ ] Compare baseline vs vLLM

#### Week 9: LLM Application Serving
- [ ] Build RAG or assistant service
- [ ] Add observability (logging, metrics)
- [ ] Define SLOs
- [ ] **Deliverable: Project 3 - Inference/RAG repo**

#### Week 10: Distributed Training Awareness
- [ ] Read ZeRO, Megatron-LM papers
- [ ] Write design note comparing DDP/FSDP/ZeRO
- [ ] Understand checkpointing, communication costs

### Phase 4: Interview Prep (Weeks 11-12)

#### Week 11: Interview Practice
- [ ] Practice transformer forward pass explanation
- [ ] Practice system design: LLM serving platform
- [ ] Practice coding: attention block, training loop

#### Week 12: Capstone & Packaging
- [ ] Clean up all 3 project repos
- [ ] Write READMEs with benchmarks
- [ ] Prepare 2-3 page interview summary
- [ ] Document limitations and tradeoffs

---

## Azure GPU Notes

**When needed:** Week 5 (fine-tuning), Week 8-9 (inference benchmarks)

**Recommended VMs:**
- `Standard_NC6s_v3` (1x V100 16GB) - good for LoRA fine-tuning
- `Standard_NC4as_T4_v3` (1x T4 16GB) - cheaper, good for inference

**Check access:**
1. MSDN subscription → Azure portal → check credits
2. Internal ML team → ask about shared GPU quota
3. Azure ML Studio → check compute quotas

**Estimated usage:** 20-40 GPU hours total for this curriculum
