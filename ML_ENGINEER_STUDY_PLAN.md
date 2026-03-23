# ML Engineer Study Plan: Senior Engineer Transitioning to LLM/AI Labs

**Target Roles:** ML Engineer, Applied ML Engineer, Training/Inference Engineer, Model Systems Engineer  
**Target Companies:** Anthropic, OpenAI, and similar frontier-model or high-caliber applied AI teams  
**Duration:** 12 weeks (3 months)  
**Commitment:** 20-25 hours/week  
**Audience:** Senior software engineers with strong implementation skills who want a realistic, industry-standard preparation path into ML engineering

---

## Who This Plan Is For

This guideline is for senior engineers who already know how to build production systems and now need to build ML depth fast enough to be credible in ML engineering interviews and on real projects.

This is **not** a core research scientist curriculum.

This plan optimizes for the actual work strong ML engineers do:
- understanding model training and inference mechanics well enough to debug them
- building reliable evaluation and serving systems
- making practical tradeoffs around latency, throughput, cost, quality, and safety
- reading important papers to extract implementation and systems implications
- shipping a small number of strong portfolio projects instead of touching everything once

---

## What Good Looks Like After 12 Weeks

By the end of this plan, you should be able to:
- explain transformer training and inference mechanics without hand-waving
- fine-tune an open-weight model using parameter-efficient methods
- build an evaluation pipeline for quality and safety regression testing
- serve an LLM-backed application with realistic observability and performance constraints
- discuss distributed training, inference optimization, and alignment at an engineering level
- present 2-3 credible ML engineering projects to hiring managers and interviewers

If you cannot do those six things, the plan is too broad and should be cut down further.

---

## Principles Behind This Revision

1. Depth over breadth.
A senior engineer should leave with a few defensible, interview-ready artifacts, not a long list of partially understood topics.

2. Engineering before ornamentation.
You do not need to reproduce frontier research. You do need to understand the systems, tradeoffs, and failure modes.

3. Papers are for implementation judgment.
Read papers to understand architecture, scaling, evaluation, and alignment decisions. Do not optimize for paper count.

4. Realistic compute assumptions.
A 3-month plan should work on a single decent GPU plus selective rented compute. Multi-GPU work is useful, but it should be optional or simulated unless you already have budget.

5. LLM-specific operations matter more than generic MLOps.
For target roles at Anthropic/OpenAI-class companies, evaluation, inference, safety, and training systems are more relevant than a generic feature-store-heavy MLOps curriculum.

---

## Program Structure

This version uses three layers:

- **Core:** Everyone should do this.
- **Optional:** Do this if you have time or need extra depth.
- **Stretch:** Do this if you have strong compute access or want to push toward systems specialization.

It also uses three portfolio anchors:

1. **Training Fundamentals Project**
2. **Fine-Tuning + Evaluation Project**
3. **Inference / LLM Systems Project**

These three projects matter more than the rest of the curriculum combined.

---

## Deliverables You Should Actually Produce

By week 12, have these ready:

1. A clean repository for a from-scratch transformer or minimal GPT implementation.
2. A fine-tuning repository with training logs, eval outputs, and reproducible instructions.
3. An inference or LLM application repository with benchmarks and observability.
4. A 2-3 page interview summary covering training, inference, evaluation, and alignment tradeoffs.
5. A short project portfolio note for each project: problem, architecture, bottlenecks, what failed, what you changed, and what you would do next.

---

## Phase 0: Setup and Scope Control (2-3 Days)

**Goal:** Prevent wasted time from environment churn and over-ambition.

### Core
- Set up one reproducible Python environment.
- Confirm PyTorch with GPU works.
- Choose one experiment tracker: Weights & Biases or TensorBoard.
- Choose one serving stack: FastAPI + vLLM, or FastAPI + Transformers.
- Choose one base model family for fine-tuning: Qwen, Llama, or Mistral.

### Output
- Environment lock file or requirements file
- One-page project tracker with weekly goals
- A hard rule: no more than 3 portfolio-grade builds in 12 weeks

### Project Tracker
- [PROJECT_TRACKER.md](PROJECT_TRACKER.md) - Weekly goals and progress tracking

### Recommended References
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [Weights & Biases Docs](https://docs.wandb.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [vLLM Documentation](https://docs.vllm.ai/)

### Notes
- Treat `flash-attn`, `deepspeed`, `bitsandbytes`, and `TensorRT-LLM` as optional advanced dependencies.
- Do not burn a week fighting CUDA or compiler issues unless low-level performance work is one of your explicit goals.

### Local Setup Files
- [Setup Guide (WSL)](installation/SETUP_WSL.md) - Step-by-step installation instructions
- [Automated Setup Script](installation/setup.sh) - Run `bash installation/setup.sh`
- [Requirements Lock File](installation/requirements.txt) - Pinned package versions
- [Azure GPU Setup](installation/AZURE_GPU_SETUP.md) - For Week 5+ when GPU is needed

---

## Phase 1: Foundations That Actually Matter (Weeks 1-3)

This phase is mandatory, but it should be right-sized. The goal is not to become a mathematician. The goal is to be able to reason clearly about training behavior and model mechanics.

### Week 1: Math and Training Intuition

**Goal:** Gain enough mathematical fluency to understand backpropagation, loss surfaces, and optimization behavior.

### Core Topics
- vectors, matrices, tensor shapes, broadcasting
- gradients, chain rule, Jacobian intuition
- softmax, cross-entropy, KL divergence
- probability basics needed for likelihood and sampling

### Skip or Compress If Already Strong
If you already remember the math, compress this week into 2-3 days and move to implementation quickly.

### Core Resources
- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Mathematics for Machine Learning](https://mml-book.github.io/): Chapter 2 (Linear Algebra), Chapter 5 (Vector Calculus), Chapter 7 (Optimization)
- [CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) and [CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)

### Additional References
- [Khan Academy: Calculus 1](https://www.khanacademy.org/math/calculus-1)

### Core Build
**Project 1A: Optimization from Scratch**
- linear regression with gradient descent
- logistic regression with gradient descent
- tiny autograd engine or computational graph
- plots of learning rate sensitivity and loss curves

### What You Must Be Able to Explain
- why gradients vanish or explode
- why softmax + cross-entropy pair well together
- why optimization is sensitive to scale and initialization

---

### Week 2: Neural Network Mechanics

**Goal:** Understand the mechanics of neural training well enough to debug basic models without framework magic.

### Core Topics
- MLP structure and forward/backward pass
- initialization and optimization
- regularization: dropout, weight decay, normalization
- mini-batch training and data pipeline basics

### Core Resources
- [Deep Learning Book](https://www.deeplearningbook.org/): selected chapters, not the whole book
- [Karpathy: Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ): focus on the implementation-heavy parts
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) and [PyTorch Tensor Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

### Additional References
- [CS231n Course Page](https://cs231n.stanford.edu/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Adam Paper](https://arxiv.org/abs/1412.6980)

### Core Build
**Project 1B: Minimal NN Framework or Minimal Trainer**
Choose one:
- a small NumPy autograd framework, or
- a minimal PyTorch training engine with clean abstractions

Train on MNIST or CIFAR-10. The point is not benchmark chasing. The point is understanding the training loop and the failure modes.

### Optional
- batch normalization paper
- Adam paper
- compare SGD vs Adam and document the observed differences

---

### Week 3: Attention and Transformers

**Goal:** Understand transformer blocks at implementation level.

### Core Topics
- self-attention mechanics
- causal masking
- positional information
- encoder-decoder vs decoder-only distinction
- tokenization basics and sequence length tradeoffs

### Core Resources
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Karpathy: Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

### Additional References
- [nanoGPT Repository](https://github.com/karpathy/nanoGPT)
- [Stanford CS224n](https://web.stanford.edu/class/cs224n/)

### Core Build
**Project 1C: Minimal GPT / Transformer Implementation**
- implement multi-head attention
- implement MLP block, residuals, normalization
- train a tiny language model on a small corpus
- inspect attention maps and generation samples

### Deliverable
A repository where you can explain every tensor transformation in the forward pass.

---

## Phase 2: Practical LLM Engineering (Weeks 4-7)

This is the center of the plan. If you only execute one phase well, execute this one well.

### Week 4: Modern LLM Architecture and Tokenization

**Goal:** Understand the components and tradeoffs in current open-weight LLMs.

### Core Topics
- decoder-only transformer design
- RoPE, RMSNorm, GQA/MQA
- BPE/SentencePiece tokenization
- context length and memory scaling
- pretraining objective vs instruction-tuning behavior

### Core Papers
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)

### Additional References
- [SentencePiece](https://github.com/google/sentencepiece)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- [Andrej Karpathy: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)

### Core Work
- read a modern open model config end-to-end
- trace how architecture parameters affect memory and latency
- compare 2-3 model families on context length, parameter count, tokenizer, and intended use

### Optional
- BERT and encoder models for contrast
- ALiBi and long-context variants

---

### Week 5: Fine-Tuning That a Real Team Would Respect

**Goal:** Learn one reproducible fine-tuning workflow well.

### Core Topics
- supervised fine-tuning data format
- prompt / chat template formatting
- LoRA and QLoRA
- batch size, effective batch size, gradient accumulation
- training instability and data-quality failures

### Core Papers
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

### Additional References
- [PEFT Quicktour](https://huggingface.co/docs/peft/quicktour)
- [Transformers Training Documentation](https://huggingface.co/docs/transformers/training)
- [TRL SFT Trainer Documentation](https://huggingface.co/docs/trl/sft_trainer)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)

### Core Build
**Project 2A: Fine-Tune an Open Model**
Recommended scope:
- choose a 3B-8B model that fits your setup
- prepare a small, clean instruction dataset
- fine-tune with LoRA or QLoRA
- track training loss, validation quality, and sample outputs
- document hyperparameters, cost, runtime, and what failed

### Required Outcome
You should be able to explain:
- why you chose LoRA or QLoRA
- what limited training quality most: data, compute, or hyperparameters
- what you would change in the next run

### Optional
- full fine-tuning comparison if you have compute
- adapter comparison
- experiment with chat templates and data cleaning rules

---

### Week 6: Evaluation, Regression Testing, and Safety Basics

**Goal:** Build the eval muscle that separates LLM engineering from casual model tinkering.

### Core Topics
- task-level evaluation vs human preference evaluation
- benchmark pitfalls
- LLM-as-judge strengths and weaknesses
- hallucination, refusal, instruction-following, and robustness checks
- safety and abuse-oriented eval categories

### Core Resources
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HELM](https://crfm.stanford.edu/helm/latest/)
- [Anthropic Research](https://www.anthropic.com/research) and [OpenAI Research](https://openai.com/research)
- [Hugging Face Model Cards](https://huggingface.co/docs/hub/model-cards)

### Additional References
- [Evaluating LLMs Course Chapter](https://huggingface.co/learn/llm-course/chapter3/3)

### Core Build
**Project 2B: Evaluation Pipeline**
- create a small benchmark suite for your fine-tuned model
- include quality evals and safety-oriented evals
- record before/after fine-tuning outputs
- define regression tests for future model changes
- produce a short model card or eval report

### Required Outcome
Be able to explain:
- why benchmark scores are not enough
- how you would prevent regressions before deploying a model update
- how you would separate quality regressions from safety regressions

### Optional Papers
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [TruthfulQA](https://arxiv.org/abs/2109.07958)
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)

---

### Week 7: Alignment and Post-Training Awareness

**Goal:** Understand alignment workflows at an engineering level, without pretending to become an alignment researcher in one week.

### Core Topics
- SFT vs preference optimization
- reward models at a high level
- PPO in RLHF: what it is, why it is operationally complex
- DPO as a simpler practical path
- Constitutional AI and RLAIF concepts

### Core Papers
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

### Additional References
- [TRL DPO Trainer Documentation](https://huggingface.co/docs/trl/dpo_trainer)
- [Hugging Face RLHF Blog](https://huggingface.co/blog/rlhf)
- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)

### Core Work
Choose one:
- implement a very small DPO-style or preference-optimization experiment on a tiny model, or
- do a deep design review of RLHF/DPO pipelines and document the operational components, data flow, and failure modes

For most engineers in 12 weeks, the second option is the better use of time.

### Required Outcome
You should be able to discuss:
- why post-training exists
- where preference data comes from
- why reward hacking and specification errors matter
- how alignment work affects engineering systems design

---

## Phase 3: Systems Engineering for LLMs (Weeks 8-10)

This phase is where strong backend and infrastructure engineers can create obvious differentiation.

### Week 8: Inference Fundamentals and Performance

**Goal:** Understand and measure the performance envelope of LLM serving.

### Core Topics
- prefill vs decode
- KV cache behavior
- batching and continuous batching
- throughput vs latency tradeoffs
- memory pressure and context-length costs
- quantization basics

### Core Papers / Tools
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [GPTQ](https://arxiv.org/abs/2210.17323) or [AWQ](https://arxiv.org/abs/2306.00978) at a conceptual level
- [vLLM](https://github.com/vllm-project/vllm) or [llama.cpp](https://github.com/ggerganov/llama.cpp) as practical baselines

### Additional References
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [TensorRT-LLM Repository](https://github.com/NVIDIA/TensorRT-LLM)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)

### Core Build
**Project 3A: Inference Benchmark Harness**
- deploy a model locally or on rented GPU
- benchmark latency, throughput, memory, and concurrency behavior
- compare baseline serving vs optimized serving stack
- document token throughput and bottlenecks

### Required Outcome
Be able to explain:
- why LLM inference is memory-bound so often
- why batching helps and when it hurts tail latency
- what KV cache buys you and what it costs you

### Optional
- quantized serving comparison
- speculative decoding
- FlashAttention integration details

---

### Week 9: LLM Application Serving and Observability

**Goal:** Build a production-style serving layer around a model-backed application.

### Core Topics
- request shaping and prompt management
- API contracts and structured output
- failure handling and retries
- caching layers
- observability: latency, tokens, quality regressions, refusal rates, cost
- online guardrails and abuse monitoring

### Core Build
**Project 3B: Serve an LLM Application**
Build one:
- RAG assistant with evals and tracing
- structured extraction or classification service
- internal copilot-style workflow assistant

### Additional References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Pinecone: RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [OpenTelemetry Python Instrumentation](https://opentelemetry.io/docs/languages/python/)
- [LangSmith Observability](https://docs.smith.langchain.com/)

The important point is not novelty. The important point is disciplined engineering.

### Minimum Requirements
- FastAPI or equivalent service layer
- inference backend
- request/response logging
- eval or regression test suite
- benchmark summary
- failure analysis note

### Required Outcome
You should be able to discuss:
- what metrics you would page on
- how you would roll out a new model version safely
- how you would attribute issues to prompting, retrieval, or model quality

---

### Week 10: Training and Distributed Systems Awareness

**Goal:** Understand large-scale training systems well enough to discuss tradeoffs credibly, even if you are not running a true frontier-scale cluster.

### Core Topics
- DDP, FSDP, ZeRO
- tensor and pipeline parallelism concepts
- checkpointing and fault tolerance
- mixed precision training
- communication costs and memory tradeoffs

### Core Resources
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- one practical distributed training walkthrough such as the [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)

### Additional References
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [NVIDIA Megatron-LM Repository](https://github.com/NVIDIA/Megatron-LM)
- [Hugging Face Distributed Training Guide](https://huggingface.co/docs/transformers/perf_train_gpu_many)

### Core Work
Choose one:
- run a small distributed training job across available GPUs, or
- build a design note comparing DDP, FSDP, and ZeRO for a hypothetical 7B model training setup

### Stretch
If you have serious compute access:
- run multi-GPU fine-tuning or pretraining on a small model
- compare memory footprint and throughput under different sharding strategies

### Required Outcome
Be able to explain:
- why data parallelism alone breaks down at scale
- what parameter sharding changes
- how checkpointing and communication affect end-to-end training throughput

---

## Phase 4: Interview Conversion and Portfolio Packaging (Weeks 11-12)

This phase converts learning into hiring signal.

### Week 11: Interview Preparation for ML Engineering Roles

**Goal:** Practice the kinds of questions strong ML engineering teams actually ask.

### Technical Topics to Prepare
- transformer forward pass and training loop
- gradient accumulation, mixed precision, optimizer state
- tokenization and context-length tradeoffs
- serving tradeoffs: latency, throughput, concurrency, batching
- evaluation strategy and regression protection
- basic alignment and safety concepts
- distributed training tradeoffs

### Interview References
- [Chip Huyen: Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [Made With ML](https://madewithml.com/)
- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/)
- [CS224n](https://web.stanford.edu/class/cs224n/)

### Coding Topics to Prepare
- attention block from scratch
- simple training loop in PyTorch
- LoRA wrapper intuition
- BPE or tokenizer basics
- evaluation harness design

### System Design Topics to Prepare
- design an LLM serving platform
- design a model evaluation pipeline
- design safe model release gates
- design a RAG service with observability
- design a distributed training stack for mid-sized models

### Behavioral Topics to Prepare
- why ML engineering, not general backend
- how you reason about reliability under uncertainty
- tradeoffs between speed, safety, and correctness
- how you handle ambiguous quality failures

---

### Week 12: Capstone and Final Packaging

**Goal:** Finish strong and package the work so that another engineer or hiring manager can assess it quickly.

### Capstone Options
Choose one and finish it well:

#### Option A: Fine-Tuned Assistant with Evals
- small instruction-tuned model
- eval suite with regression tests
- serving layer
- benchmark note
- model card

#### Option B: RAG System with Quality and Safety Checks
- ingestion and retrieval pipeline
- evals for answer quality and hallucination
- observability and tracing
- failure analysis across retrieval, prompt, and model

#### Option C: Inference Optimization Case Study
- benchmark baseline and optimized serving stack
- compare batching, quantization, and concurrency configurations
- produce an engineering writeup with recommendations

### Final Package Checklist
- clean README
- setup instructions
- benchmarks or eval tables
- limitations section
- future work section
- clear explanation of tradeoffs and failures

---

## Paper List: Reduced and Prioritized

Do not aim for maximum volume. Aim for strong recall of the important ones.

### Mandatory
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [GPT-3](https://arxiv.org/abs/2005.14165)
3. [LoRA](https://arxiv.org/abs/2106.09685)
4. [QLoRA](https://arxiv.org/abs/2305.14314)
5. [InstructGPT](https://arxiv.org/abs/2203.02155)
6. [Constitutional AI](https://arxiv.org/abs/2212.08073)
7. [FlashAttention](https://arxiv.org/abs/2205.14135)
8. [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
9. [ZeRO](https://arxiv.org/abs/1910.02054)
10. [LLaMA](https://arxiv.org/abs/2302.13971)

### Optional, High Value
1. [DPO](https://arxiv.org/abs/2305.18290)
2. [Chinchilla](https://arxiv.org/abs/2203.15556)
3. [TruthfulQA](https://arxiv.org/abs/2109.07958)
4. [HELM](https://arxiv.org/abs/2211.09110)
5. [CLIP](https://arxiv.org/abs/2103.00020)
6. [RAG](https://arxiv.org/abs/2005.11401)
7. [Mamba](https://arxiv.org/abs/2312.00752)

### What You Should Extract From Each Paper
For every paper, write down:
- the core problem it solves
- the key idea
- the engineering consequence
- the main tradeoff or failure mode
- where it would matter in a real system

If you cannot produce those five bullets, you did not extract the value from the paper.

---

## Compute Guidance: Honest Version

### Track A: Single Consumer GPU or Limited Budget
Good for most of this plan.

You can still do:
- tiny transformer training
- LoRA / QLoRA fine-tuning on smaller models
- evaluation pipelines
- inference benchmarking
- RAG and application serving

Avoid claiming frontier-scale training experience from this setup.

### Track B: Intermittent Rented GPU Compute
Best practical path.

Use this for:
- short fine-tuning runs
- inference benchmarking
- occasional multi-GPU experiments
- one strong systems case study

### Track C: Serious Multi-GPU Access
Only relevant if you already have budget or employer resources.

Use this for:
- actual sharded fine-tuning experiments
- memory/throughput comparisons across DDP, FSDP, and ZeRO
- more credible distributed systems measurements

### Budget Guidance
A realistic 3-month learning budget is often enough for:
- targeted fine-tuning runs
- inference experiments
- one or two distributed systems experiments

It is usually not enough for repeated large-model training mistakes. Design experiments to be small and diagnostic.

### Compute References
- [Lambda Labs Cloud](https://lambdalabs.com/service/gpu-cloud)
- [RunPod](https://www.runpod.io/)
- [Vast.ai](https://vast.ai/)
- [Google Colab](https://colab.research.google.com/)
- [Modal](https://modal.com/)

---

## Weekly Time Budget

| Activity | Hours/Week |
|----------|------------|
| Implementation and experiments | 10-12 |
| Paper reading and notes | 3-4 |
| Debugging and tooling | 2-3 |
| Benchmarking and evaluation | 2-3 |
| Interview prep and packaging | 3-4 |
| **Total** | **20-26** |

The time budget is realistic only if you keep the number of active projects small.

---

## What Senior Engineers Usually Get Wrong

1. They read too much and ship too little.
2. They chase distributed training too early.
3. They optimize for paper count instead of systems understanding.
4. They treat evaluation as an afterthought.
5. They build demos without measuring latency, cost, or regressions.
6. They overclaim on model training scale based on small experiments.
7. They underestimate data quality and prompt-formatting effects.

Avoid all seven.

---

## Interview Positioning Guidance

If you are targeting Anthropic or OpenAI style ML engineering roles, your story should sound like this:

- I understand the mechanics of model training and inference.
- I can fine-tune and evaluate models in a disciplined, reproducible way.
- I can build systems around models that are reliable, observable, and safe.
- I know where scaling and alignment introduce engineering complexity.
- I do not confuse toy demos with production-ready ML systems.

That positioning is stronger than trying to sound like a research scientist without research output.

---

## Recommended Final Portfolio

If you only complete three things, complete these:

1. **Minimal GPT / Transformer from Scratch**
Shows real understanding of the architecture.

2. **Fine-Tuning + Evaluation Project**
Shows you can work with actual model adaptation and quality measurement.

3. **Inference or RAG System with Observability**
Shows you can ship something production-shaped.

This trio is enough to create a credible ML engineer transition narrative.

---

## Final Advice

For senior engineers, the biggest advantage is not knowing more buzzwords. It is showing better judgment.

Good judgment in ML engineering looks like:
- choosing measurable experiments
- keeping scope under control
- understanding where model quality really comes from
- building evaluation before deployment
- knowing when systems bottlenecks, not model architecture, are the actual problem

Use this plan to build that judgment.
