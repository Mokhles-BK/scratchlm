r"""Evaluation for ScratchLM (Phase 6): how good is a checkpoint, as numbers?

WHY A SEPARATE EVALUATOR
========================
During training, "VAL" is the average loss on 50 RANDOM batches. Random means
two evaluations of the same model give slightly different numbers (noise).
For comparing runs we want a number that is the same every time:

    * walk through the validation file from start to end in
      non-overlapping windows of block_size tokens (every token is predicted
      exactly once, nothing is sampled), and average the loss.

WHAT THE NUMBERS MEAN
=====================
loss        : average cross-entropy in nats per token. Lower is better.
perplexity  : exp(loss). "The model is as unsure as if it chose uniformly
              between `perplexity` tokens at every step".
bits/byte   : loss / ln(2) / (bytes per token). Unlike loss it does NOT depend
              on the tokenizer, so it is the fair way to compare models that
              use different vocabularies. (We have one tokenizer, but this is
              the standard number in papers.)
by position : loss at token position 0, 1, 2, ... inside the window. Position 0
              has no context at all, so its loss is high; later positions have
              more context, so loss falls. The curve shows HOW MUCH the model
              uses its context.

Run (from the project root):
    python -m scratchlm.evaluate --checkpoint checkpoints\small-3k\best.pt
    python -m scratchlm.evaluate --checkpoint checkpoints\a\best.pt checkpoints\b\best.pt
"""

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scratchlm.data import TokenDataset
from scratchlm.tokenizer import BPETokenizer
from scratchlm.train import check_gpu_free, load_model_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent

# Position ranges reported in the "loss by position" summary: [start, end)
POSITION_BUCKETS = [(0, 1), (1, 16), (16, 64), (64, 256), (256, 10**9)]


# ---------------------------------------------------------------------------
# Walking through the data
# ---------------------------------------------------------------------------

def iter_windows(tokens, block_size, batch_size, max_windows=None):
    """Yield (x, y) int64 arrays of shape (b, block_size), covering `tokens` in order.

    Window i holds tokens[i*T : i*T + T] as input and the same range shifted by
    one as target. Windows do not overlap, so each token (except the very first)
    is predicted exactly once. The last few tokens that do not fill a window are
    left out.
    """
    T = block_size
    n_windows = (len(tokens) - 1) // T      # window i needs i*T + T + 1 <= N
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    for first in range(0, n_windows, batch_size):
        starts = [i * T for i in range(first, min(first + batch_size, n_windows))]
        x = np.stack([tokens[s: s + T] for s in starts]).astype(np.int64)
        y = np.stack([tokens[s + 1: s + 1 + T] for s in starts]).astype(np.int64)
        yield x, y


# ---------------------------------------------------------------------------
# Loss over a whole token array
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_tokens(model, tokens, block_size=None, batch_size=16, device="cpu",
                    autocast_ctx=None, max_windows=None):
    """Deterministic average loss of `model` over `tokens` (a 1-D uint16 array).

    Returns a dict: loss, perplexity, n_tokens, n_windows, loss_by_position
    (a numpy array of length block_size: mean loss at each position).
    """
    block_size = block_size or model.cfg.block_size
    if block_size > model.cfg.block_size:
        raise ValueError(f"block_size {block_size} exceeds the model's {model.cfg.block_size}")
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    was_training = model.training
    model.eval()

    pos_sum = torch.zeros(block_size, dtype=torch.float64)
    n_windows = 0
    for x, y in iter_windows(tokens, block_size, batch_size, max_windows):
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with ctx:
            logits, _ = model(x)                                        # (b, T, V)
        # reduction="none": keep one loss per token instead of the average,
        # so we can average over windows AND look at each position.
        per_token = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(),
                                    reduction="none").view(y.shape)     # (b, T)
        pos_sum += per_token.sum(dim=0).double().cpu()
        n_windows += x.size(0)
    model.train(was_training)

    if n_windows == 0:
        raise ValueError("no complete window fits in the data")
    by_pos = (pos_sum / n_windows).numpy()
    loss = float(by_pos.mean())
    return {"loss": loss, "perplexity": math.exp(loss), "n_tokens": n_windows * block_size,
            "n_windows": n_windows, "loss_by_position": by_pos}


def bucket_summary(by_pos):
    """Mean loss inside each POSITION_BUCKETS range -> {"0": ..., "1-15": ...}."""
    out = {}
    for lo, hi in POSITION_BUCKETS:
        if lo >= len(by_pos):
            continue
        hi = min(hi, len(by_pos))
        name = f"{lo}" if hi - lo == 1 else f"{lo}-{hi - 1}"
        out[name] = float(np.mean(by_pos[lo:hi]))
    return out


# ---------------------------------------------------------------------------
# Tokenizer-independent number
# ---------------------------------------------------------------------------

def bytes_per_token(tokens, tokenizer, sample=200_000):
    """Average number of UTF-8 bytes one token stands for (estimated on a prefix)."""
    ids = [int(t) for t in tokens[:sample]]
    return len(tokenizer.decode(ids).encode("utf-8")) / len(ids)


def bits_per_byte(loss, bytes_per_tok):
    return loss / math.log(2) / bytes_per_tok


# ---------------------------------------------------------------------------
# Repetition: a simple number for "does the text loop?"
# ---------------------------------------------------------------------------

def distinct_ngram_ratio(ids, n=3):
    """Unique n-grams / all n-grams in a token list. 1.0 = nothing repeats,
    small = the text keeps repeating itself."""
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Deterministic validation loss of checkpoints")
    p.add_argument("--checkpoint", nargs="+", required=True, help="one or more .pt files")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p.add_argument("--tokenizer", default=str(REPO_ROOT / "data" / "tokenizer.json"))
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=None,
                   help="evaluate with a SHORTER context than the model was trained on")
    p.add_argument("--max-windows", type=int, default=None, help="only the first N windows (quick check)")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--no-save", action="store_true", help="do not write eval_<split>.json next to the checkpoint")
    a = p.parse_args()

    try:
        if a.device.startswith("cuda"):
            check_gpu_free()
        tokenizer = BPETokenizer.load(a.tokenizer)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        sys.exit(1)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if a.device.startswith("cuda") else contextlib.nullcontext())

    rows = []
    for path in a.checkpoint:
        model = load_model_from_checkpoint(path, device=a.device).eval()
        T = a.block_size or model.cfg.block_size
        ds = TokenDataset(Path(a.data_dir) / f"{a.split}.bin", T)
        r = evaluate_tokens(model, ds.tokens, T, a.batch_size, a.device, ctx, a.max_windows)
        bpt = bytes_per_token(ds.tokens, tokenizer)
        r["bits_per_byte"] = bits_per_byte(r["loss"], bpt)
        r["bytes_per_token"] = bpt
        r["params"] = model.num_params()
        r["block_size"] = T
        r["by_position"] = bucket_summary(r.pop("loss_by_position"))
        r["checkpoint"] = str(path)
        rows.append(r)

        print(f"\n{path}")
        print(f"  params {r['params']:,} | context {T} | {r['n_tokens']:,} tokens in {r['n_windows']:,} windows")
        print(f"  loss {r['loss']:.4f} | perplexity {r['perplexity']:.2f} | bits/byte {r['bits_per_byte']:.4f}")
        print("  loss by position: " + "  ".join(f"[{k}] {v:.3f}" for k, v in r["by_position"].items()))
        if not a.no_save:
            out = Path(path).with_name(f"eval_{a.split}_{Path(path).stem}.json")
            out.write_text(json.dumps(r, indent=2))
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
