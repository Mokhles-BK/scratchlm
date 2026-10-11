# ScratchLM

A language model built entirely from scratch in PyTorch: our own byte-level BPE
tokenizer, our own decoder-only transformer architecture, our own training loop,
and our own text generation. No pretrained models, no Hugging Face `transformers`
model classes, no fine-tuning of someone else's checkpoint.

## What's inside

| Path | Purpose |
|---|---|
| `scratchlm/` | Source modules: `tokenizer.py`, `model.py`, `data.py`, `train.py`, `generate.py` |
| `scripts/` | Setup and data-preparation scripts (`setup.ps1`, `download_data.py`, `prepare_data.py`) |
| `tests/` | Unit tests for each module |
| `data/` | Raw text and tokenized `.bin` files (gitignored) |
| `checkpoints/` | Trained model checkpoints (gitignored) |

## Hardware

- GPU: NVIDIA GeForce RTX 3050 (Laptop), 4 GB VRAM
- Target model size: 10–40M parameters, context length 256–512
- bf16 autocast + gradient accumulation + small micro-batches so training fits in 4 GB
- Windows has no Triton: attention uses `torch.nn.functional.scaled_dot_product_attention`

## Roadmap

0. Setup — repo, venv, PyTorch CUDA build, download TinyStories, GPU sanity check
1. Tokenizer — byte-level BPE from scratch
2. Data pipeline — tokenize to `.bin`, train/val split, memory-mapped batch loader
3. Model — decoder-only GPT (RoPE, RMSNorm, causal MHA, SwiGLU, weight tying)
4. Training loop — AdamW, warmup + cosine decay, grad accumulation, bf16, checkpoints
5. Generation — temperature / top-k / top-p sampling with KV cache
6. Evaluation — perplexity, sample quality, size vs. context experiments
7. Optional — instruction/chat fine-tuning

## Running

```powershell
# One-time setup (creates .venv, installs PyTorch CUDA build + requirements)
scripts\setup.ps1

# Sanity check that CUDA + bf16 attention work on the GPU
python scripts\gpu_check.py

# Download the TinyStories training text into data/raw/
python scripts\download_data.py

# After Phases 1-2:
python scripts\prepare_data.py
```