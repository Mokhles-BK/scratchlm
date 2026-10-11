r"""Train the ScratchLM byte-level BPE tokenizer on a slice of TinyStories.

Reads the first N characters of data/raw/train.txt (default 50 million, about
50 MB), learns merges until the vocabulary has 4096 tokens, and saves
data/tokenizer.json. Then prints a few sanity checks.

Run:  python scripts\train_tokenizer.py
Opts: --chars 50000000  --vocab-size 4096  --out data\tokenizer.json
"""

import argparse
import sys
import time
from pathlib import Path

# Let this script import the scratchlm package that lives one folder up.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scratchlm.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "raw" / "train.txt"))
    parser.add_argument("--val", default=str(REPO_ROOT / "data" / "raw" / "validation.txt"))
    parser.add_argument("--chars", type=int, default=50_000_000,
                        help="how many characters of the input to train on")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "tokenizer.json"))
    args = parser.parse_args()

    # newline="" = read the file exactly as stored, no newline translation.
    print(f"Reading first {args.chars:,} characters of {args.input}")
    with open(args.input, "r", encoding="utf-8", newline="") as f:
        text = f.read(args.chars)

    print(f"Training BPE to vocab size {args.vocab_size} ...")
    start = time.time()
    tok = BPETokenizer.train(text, args.vocab_size, verbose=True)
    print(f"Training took {time.time() - start:.1f} s, vocab size = {tok.vocab_size}")

    tok.save(args.out)
    print(f"Saved tokenizer -> {args.out}")

    # --- Sanity checks ---------------------------------------------------
    print("\nFirst 15 merges (these are the most frequent pairs):")
    for rank in range(min(15, len(tok.merges))):
        a, b = tok.merges[rank]
        token_bytes = tok.vocab[tok.first_merge_id + rank]
        print(f"  {rank:>3}: {tok.vocab[a]!r} + {tok.vocab[b]!r} -> {token_bytes!r}")

    sample = "Once upon a time, a little girl named Lily had a red ball. <|endoftext|>"
    ids = tok.encode(sample)
    print(f"\nSample : {sample!r}")
    print(f"ids    : {ids}")
    print(f"tokens : {[tok.vocab[i].decode('utf-8', 'replace') for i in ids]}")
    print(f"round trip ok: {tok.decode(ids) == sample}")

    # Compression ratio on UNSEEN text (the validation file): bytes per token.
    # Higher is better; 1.0 would mean no compression at all.
    with open(args.val, "r", encoding="utf-8", newline="") as f:
        val_text = f.read(1_000_000)
    val_ids = tok.encode(val_text)
    n_bytes = len(val_text.encode("utf-8"))
    print(f"\nValidation slice: {n_bytes:,} bytes -> {len(val_ids):,} tokens "
          f"= {n_bytes / len(val_ids):.2f} bytes per token")
    print(f"Lossless on this slice: {tok.decode(val_ids) == val_text}")


if __name__ == "__main__":
    main()
