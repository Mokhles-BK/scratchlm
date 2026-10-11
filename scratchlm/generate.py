r"""Text generation for ScratchLM (Phase 5): sampling and the KV cache.

HOW A LANGUAGE MODEL WRITES TEXT
================================
The model only does one thing: given tokens so far, output a score (logit) for
every possible NEXT token. To write a story we loop:

    1. run the model on the tokens so far          -> logits for the next token
    2. turn the logits into ONE chosen token        (this file's "sampling")
    3. append the token, go back to 1

HOW A TOKEN IS CHOSEN (sampling)
================================
logits (V numbers) --/temperature--> --top-k--> --top-p--> softmax --> random draw

    temperature : divide logits by T before softmax.
                  T < 1 sharpens the distribution (safer, more repetitive),
                  T > 1 flattens it (more surprising, more mistakes),
                  T = 0 means "always take the best token" (greedy).
    top-k       : keep only the k highest-scoring tokens, drop the rest.
    top-p       : "nucleus" sampling: sort tokens by probability and keep the
                  smallest group whose probabilities add up to p. The group is
                  small when the model is sure, large when it is unsure.

SPEED: THE KV CACHE (see KVCache in model.py)
    Without it, every new token needs a forward pass over the WHOLE text so
    far. With it, only the one new token goes through the model.

Run (from the project root):
    python -m scratchlm.generate --checkpoint checkpoints/small-3k/best.pt ^
        --prompt "Once upon a time" --max-new-tokens 200
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from scratchlm.tokenizer import BPETokenizer
from scratchlm.train import check_gpu_free, load_model_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Choosing the next token
# ---------------------------------------------------------------------------

def filter_logits(logits, top_k=None, top_p=None):
    """Set the logits of tokens we do NOT want to -inf (their probability becomes 0).

    logits: (B, V). Returns a new tensor of the same shape.
    """
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        # Value of the k-th best logit in each row; everything below it goes.
        # (Tokens tied with the k-th value are kept.)
        kth_best = torch.topk(logits, k, dim=-1).values[..., -1, None]    # (B, 1)
        logits[logits < kth_best] = float("-inf")

    if top_p is not None and top_p < 1.0:
        if top_p <= 0.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Drop a token if the probability mass of the tokens BEFORE it already
        # reaches top_p. (So the best token is always kept, and the group is
        # the smallest one whose mass is >= top_p.)
        drop = (cumulative - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        # Undo the sorting so each logit goes back to its own token id.
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return logits


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """Pick one next token per row. logits: (B, V) -> token ids (B, 1)."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)            # greedy

    logits = logits.float() / temperature                     # sharpen or flatten
    logits = filter_logits(logits, top_k, top_p)
    probs = F.softmax(logits, dim=-1)                         # (B, V), rows sum to 1
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ---------------------------------------------------------------------------
# The generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
             eos_id=None, use_cache=True, generator=None, autocast_ctx=None):
    """Continue the token sequences in `idx`.

    idx           : (B, T) prompt token ids
    returns       : (B, T + n) prompt followed by n new tokens, n <= max_new_tokens
    eos_id        : if given, a sequence that produces this id is finished; the
                    loop stops once every sequence in the batch is finished
    use_cache     : True = KV cache (fast). False = recompute everything every
                    step (slow; kept so tests can prove both give the same text)

    The model can only handle block_size tokens in total, so generation stops
    when prompt + new tokens reach block_size.
    """
    block_size = model.cfg.block_size
    B, T = idx.shape
    if T == 0:
        raise ValueError("prompt must contain at least one token")
    if T > block_size:
        raise ValueError(f"prompt has {T} tokens but block_size is {block_size}")
    max_new_tokens = min(max_new_tokens, block_size - T)

    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    was_training = model.training
    model.eval()
    caches = model.make_caches() if use_cache else None

    out = idx                                       # everything so far
    feed = idx                                      # what the model sees this step
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)

    for _ in range(max_new_tokens):
        with ctx:
            logits, _ = model(feed, caches=caches)  # (B, len(feed), V)
        next_logits = logits[:, -1, :]              # scores for the token AFTER the last one
        nxt = sample_next_token(next_logits, temperature, top_k, top_p, generator)  # (B, 1)

        if eos_id is not None:
            # Finished sequences just keep repeating the end token.
            nxt = torch.where(finished[:, None], torch.full_like(nxt, eos_id), nxt)
            finished = finished | (nxt[:, 0] == eos_id)

        out = torch.cat([out, nxt], dim=1)
        # With a cache the past is remembered, so feed ONLY the new token.
        feed = nxt if use_cache else out
        if eos_id is not None and bool(finished.all()):
            break

    model.train(was_training)
    return out


def generate_text(model, tokenizer, prompt="", max_new_tokens=200, temperature=0.8,
                  top_k=40, top_p=0.95, seed=None, use_cache=True, stop_at_eot=True):
    """Prompt string in, story string out (the prompt followed by the continuation).

    Training stories always started right after an <|endoftext|> token, so we
    begin the token sequence with one. That tells the model "a new story
    starts here".
    """
    device = next(model.parameters()).device
    ids = [tokenizer.eot_id] + tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    if device.type == "cuda":
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = contextlib.nullcontext()

    out = generate(model, idx, max_new_tokens, temperature, top_k, top_p,
                   eos_id=tokenizer.eot_id if stop_at_eot else None,
                   use_cache=use_cache, generator=generator, autocast_ctx=ctx)

    all_ids = out[0].tolist()
    prompt_ids, new_ids = all_ids[1:len(ids)], all_ids[len(ids):]   # drop the leading <|endoftext|>
    if stop_at_eot and tokenizer.eot_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.eot_id)]        # the story ended here
    return tokenizer.decode(prompt_ids + new_ids)                   # prompt + continuation


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate text with a trained ScratchLM checkpoint")
    p.add_argument("--checkpoint", required=True, help="e.g. checkpoints/small-3k/best.pt")
    p.add_argument("--tokenizer", default=str(REPO_ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40, help="0 = off")
    p.add_argument("--top-p", type=float, default=0.95, help="1.0 = off")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu",
                   help="cpu (default, safe while training runs) or cuda")
    p.add_argument("--no-cache", action="store_true", help="disable the KV cache")
    p.add_argument("--compare-speed", action="store_true",
                   help="time generation with and without the KV cache")
    a = p.parse_args()

    try:
        if a.device.startswith("cuda"):
            check_gpu_free()
        tokenizer = BPETokenizer.load(a.tokenizer)
        model = load_model_from_checkpoint(a.checkpoint, device=a.device).eval()
    except (RuntimeError, OSError, ValueError) as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        sys.exit(1)

    top_k = a.top_k if a.top_k > 0 else None
    top_p = a.top_p if a.top_p < 1.0 else None

    if a.compare_speed:
        print("Speed test: 128 greedy tokens from an empty prompt ...")
        texts = {}
        for use_cache in (False, True):
            t0 = time.time()
            texts[use_cache] = generate_text(model, tokenizer, "", 128, temperature=0,
                                             top_k=None, top_p=None, use_cache=use_cache,
                                             stop_at_eot=False)
            dt = time.time() - t0
            print(f"  KV cache {'ON ' if use_cache else 'OFF'}: {128 / dt:7.1f} tokens/s")
        print(f"  same output both ways: {texts[False] == texts[True]}\n")

    for i in range(a.num_samples):
        seed = None if a.seed is None else a.seed + i
        text = generate_text(model, tokenizer, a.prompt, a.max_new_tokens, a.temperature,
                             top_k, top_p, seed=seed, use_cache=not a.no_cache)
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
