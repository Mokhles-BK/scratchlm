r"""Training loop for ScratchLM (Phase 4).

WHAT TRAINING IS
================
We repeat this one step thousands of times:

    1. Take a batch of text windows (x) and the same windows shifted by one (y).
    2. Forward pass: the model predicts, at every position, a score for each
       possible next token. The loss (cross-entropy) is small when the true
       next token gets a high score.
    3. Backward pass: autograd computes d(loss)/d(weight) for EVERY weight.
    4. The optimiser (AdamW) nudges every weight a little in the direction
       that lowers the loss.

On top of that basic recipe this file adds the tricks that make it work on a
4 GB laptop GPU:

    * gradient accumulation : to simulate a big batch, run several small
      micro-batches, ADD their gradients, and only then take one optimiser
      step. Same maths as one big batch, but memory of one small batch.
    * bf16 autocast         : matrix multiplies run in bfloat16 (half the memory
      traffic, faster tensor cores) while the weights stay float32.
    * learning-rate schedule: warm up from ~0, then decay along a cosine.
    * gradient clipping     : cap the size of the update so one bad batch
      cannot wreck the model.
    * checkpoints           : save everything needed to stop and resume.

Run (from the project root):
    python -m scratchlm.train --model small --name small-run --max-steps 3000
    python -m scratchlm.train --model small --name small-run --resume
"""

import argparse
import contextlib
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from scratchlm.data import open_splits
from scratchlm.model import BASE, SMALL, GPT, GPTConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Everything about HOW we train (the model's shape lives in GPTConfig)."""

    data_dir: str = str(REPO_ROOT / "data")
    out_dir: str = str(REPO_ROOT / "checkpoints" / "run")
    device: str = "cuda"

    micro_batch_size: int = 8      # sequences per forward/backward pass
    grad_accum_steps: int = 8      # micro-batches per optimiser step
    max_steps: int = 3000          # optimiser steps in the whole run

    lr: float = 6e-4               # peak learning rate
    min_lr_ratio: float = 0.1      # final lr = lr * this
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_interval: int = 250       # validate every N steps
    eval_batches: int = 50         # batches averaged per validation
    ckpt_interval: int = 500       # write latest.pt every N steps
    milestone_interval: int = 0    # also keep step_XXXXXX.pt every N steps (0 = off)
    log_interval: int = 10         # print + CSV row every N steps
    seed: int = 1337


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(step, cfg: TrainConfig):
    """Learning rate to use for optimiser step number `step` (0-based).

        step < warmup      : rises linearly from ~0 up to lr
                             (at the start the weights are random and the
                             gradients are noisy; big steps would do damage)
        warmup .. max_steps: follows half a cosine from lr down to min_lr
        step >= max_steps  : stays at min_lr
    """
    min_lr = cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))                                # 1 -> 0
    return min_lr + coeff * (cfg.lr - min_lr)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def make_optimizer(model, cfg: TrainConfig, device_type):
    """AdamW with two parameter groups.

    Weight decay shrinks weights a little every step (a regulariser). It makes
    sense for the big matrices (linear layers, embedding: tensors with >= 2
    dims) but NOT for the RMSNorm gains (1-D vectors), which should be free to
    sit near 1.
    """
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # The fused CUDA kernel is faster; it only exists on GPU.
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                             fused=(device_type == "cuda"))


# ---------------------------------------------------------------------------
# One optimiser step worth of forward/backward passes
# ---------------------------------------------------------------------------

def run_micro_batches(model, batches, autocast_ctx):
    """Forward+backward over several micro-batches, ADDING up the gradients.

    Each micro-batch loss is divided by the number of micro-batches before
    backward(). Gradients accumulate (+=) in p.grad, so the final gradient is
    the gradient of the AVERAGE loss, exactly what one big batch would give.

    Returns the average loss as a detached tensor.
    """
    n = len(batches)
    total = None
    for x, y in batches:
        with autocast_ctx:
            _, loss = model(x, targets=y)
        (loss / n).backward()
        total = loss.detach() / n if total is None else total + loss.detach() / n
    return total


@torch.no_grad()
def estimate_val_loss(model, val_ds, batch_size, n_batches, device, autocast_ctx):
    """Average loss on n_batches random validation batches (model never trains on these)."""
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = val_ds.get_batch(batch_size, device)
        with autocast_ctx:
            _, loss = model(x, targets=y)
        total += loss.item()
    model.train()
    return total / n_batches


# ---------------------------------------------------------------------------
# GPU safety check (project rule: the GPU is shared)
# ---------------------------------------------------------------------------

def check_gpu_free(max_used_mb=1024):
    """Abort if other programs already use more than `max_used_mb` of VRAM."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        used = int(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as e:
        raise RuntimeError(f"could not read GPU memory with nvidia-smi: {e}")
    if used > max_used_mb:
        raise RuntimeError(
            f"GPU already uses {used} MB (limit {max_used_mb} MB). Close the other "
            f"program (e.g. a llama-server process) and try again."
        )
    return used


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, opt, step, best_val, tcfg, model_cfg, train_ds, val_ds):
    """Write everything needed to continue training later.

    Saved to a temporary file first, then renamed: if the PC dies mid-write we
    never end up with a half-written checkpoint.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "best_val": best_val,
        "train_cfg": asdict(tcfg),
        "model_cfg": asdict(model_cfg),
        # Random-number-generator states of the two data loaders, so after
        # resuming we get the SAME batches an uninterrupted run would have.
        "train_rng": train_ds.rng.bit_generator.state,
        "val_rng": val_ds.rng.bit_generator.state,
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_model_from_checkpoint(path, device="cpu"):
    """Rebuild a trained model from a checkpoint file (used later for generation)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------

CSV_FIELDS = ["step", "train_loss", "val_loss", "lr", "grad_norm",
              "tokens_seen", "tokens_per_sec", "peak_vram_mb"]


def train(tcfg: TrainConfig, model_cfg: GPTConfig, resume=False, stop_at_step=None, log=print):
    """Train (or resume training). Returns a small summary dict.

    stop_at_step: stop early at this step (still saves latest.pt). Used by the
    resume test and handy for benchmarking; max_steps still defines the lr schedule.
    """
    device = torch.device(tcfg.device)
    if device.type == "cuda":
        used = check_gpu_free()
        log(f"GPU check ok: {used} MB already in use by other programs")

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest, best = out_dir / "latest.pt", out_dir / "best.pt"
    csv_path = out_dir / "log.csv"
    if latest.exists() and not resume:
        raise FileExistsError(f"{latest} exists. Use --resume to continue it, or pick a new --name.")
    if resume and not latest.exists():
        raise FileNotFoundError(f"nothing to resume: {latest} does not exist")

    torch.manual_seed(tcfg.seed)
    train_ds, val_ds = open_splits(tcfg.data_dir, model_cfg.block_size, seed=tcfg.seed)
    model = GPT(model_cfg).to(device)
    opt = make_optimizer(model, tcfg, device.type)

    step, best_val = 0, float("inf")
    if resume:
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        if ckpt["model_cfg"] != asdict(model_cfg):
            raise ValueError("model settings differ from the checkpoint being resumed")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        train_ds.rng.bit_generator.state = ckpt["train_rng"]
        val_ds.rng.bit_generator.state = ckpt["val_rng"]
        log(f"Resumed from step {step} (best val loss so far {best_val:.4f})")

    # bf16 autocast only on GPU; on CPU (tests) plain float32.
    if device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    tokens_per_step = tcfg.micro_batch_size * tcfg.grad_accum_steps * model_cfg.block_size
    end_step = min(tcfg.max_steps, stop_at_step) if stop_at_step else tcfg.max_steps
    log(f"Model: {model.num_params():,} parameters | {tokens_per_step:,} tokens per step | "
        f"steps {step} -> {end_step} of {tcfg.max_steps}")

    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    history = []                 # (step, val_loss) at every validation
    last_loss = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_last, tokens_since_log = time.time(), 0

    def checkpoint(path):
        save_checkpoint(path, model, opt, step, best_val, tcfg, model_cfg, train_ds, val_ds)

    model.train()
    opt.zero_grad(set_to_none=True)
    try:
        while step < end_step:
            # --- one optimiser step ------------------------------------
            lr = get_lr(step, tcfg)
            for group in opt.param_groups:
                group["lr"] = lr

            batches = [train_ds.get_batch(tcfg.micro_batch_size, device)
                       for _ in range(tcfg.grad_accum_steps)]
            loss = run_micro_batches(model, batches, autocast_ctx)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens_since_log += tokens_per_step

            # --- logging, validation, checkpoints ----------------------
            do_eval = step % tcfg.eval_interval == 0 or step == end_step
            do_log = step % tcfg.log_interval == 0 or do_eval
            if do_log:
                if device.type == "cuda":
                    torch.cuda.synchronize()     # so the timing below is honest
                now = time.time()
                tok_s = tokens_since_log / max(now - t_last, 1e-9)
                t_last, tokens_since_log = now, 0
                last_loss = loss.item()
                peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
                val_loss = ""
                if do_eval:
                    val_loss = estimate_val_loss(model, val_ds, tcfg.micro_batch_size,
                                                 tcfg.eval_batches, device, autocast_ctx)
                    history.append((step, val_loss))
                    if val_loss < best_val:
                        best_val = val_loss
                        checkpoint(best)
                    t_last = time.time()         # do not count validation time in tok/s
                writer.writerow({
                    "step": step, "train_loss": f"{last_loss:.5f}",
                    "val_loss": "" if val_loss == "" else f"{val_loss:.5f}",
                    "lr": f"{lr:.6g}", "grad_norm": f"{float(grad_norm):.4f}",
                    "tokens_seen": step * tokens_per_step,
                    "tokens_per_sec": f"{tok_s:.0f}", "peak_vram_mb": f"{peak:.0f}",
                })
                csv_file.flush()
                msg = (f"step {step:>5}/{tcfg.max_steps} | loss {last_loss:.4f} | lr {lr:.2e} | "
                       f"gnorm {float(grad_norm):.2f} | {tok_s:,.0f} tok/s | vram {peak:,.0f} MB")
                if val_loss != "":
                    msg += f" | VAL {val_loss:.4f}"
                log(msg)

            if step % tcfg.ckpt_interval == 0 or step == end_step:
                checkpoint(latest)
            # Milestones are never overwritten: they let us look at the SAME
            # run at different ages (Phase 6: how do samples improve?).
            if tcfg.milestone_interval and step % tcfg.milestone_interval == 0:
                checkpoint(out_dir / f"step_{step:06d}.pt")
    except KeyboardInterrupt:
        log(f"\nInterrupted at step {step}: saving checkpoint ...")
        checkpoint(latest)
    finally:
        csv_file.close()

    log(f"Finished at step {step}. Best val loss {best_val:.4f}. Checkpoint: {latest}")
    return {"step": step, "best_val": best_val, "history": history, "last_train_loss": last_loss}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train ScratchLM")
    p.add_argument("--model", choices=["small", "base"], default="small")
    p.add_argument("--name", default="run", help="run name; files go to checkpoints/<name>/")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--stop-at-step", type=int, default=None,
                   help="stop early at this step (e.g. 20 for a benchmark)")
    p.add_argument("--lr", type=float, default=None, help="peak lr (default 1e-3 small, 6e-4 base)")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--ckpt-interval", type=int, default=500)
    p.add_argument("--milestone-interval", type=int, default=0,
                   help="also keep step_XXXXXX.pt every N steps (0 = off)")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    a = p.parse_args()

    model_cfg = replace(SMALL if a.model == "small" else BASE, block_size=a.block_size)
    tcfg = TrainConfig(
        data_dir=a.data_dir, out_dir=str(REPO_ROOT / "checkpoints" / a.name), device=a.device,
        micro_batch_size=a.micro_batch_size, grad_accum_steps=a.grad_accum_steps,
        max_steps=a.max_steps, lr=a.lr if a.lr else (1e-3 if a.model == "small" else 6e-4),
        warmup_steps=a.warmup_steps, eval_interval=a.eval_interval, eval_batches=a.eval_batches,
        ckpt_interval=a.ckpt_interval, milestone_interval=a.milestone_interval,
        log_interval=a.log_interval, seed=a.seed,
    )
    try:
        train(tcfg, model_cfg, resume=a.resume, stop_at_step=a.stop_at_step)
    except (RuntimeError, FileExistsError, FileNotFoundError, ValueError) as e:
        print(f"\nSTOPPED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
