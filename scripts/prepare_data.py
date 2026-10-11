r"""Turn the raw TinyStories text into token-id files for training (Phase 2).

Reads   data/raw/train.txt  and  data/raw/validation.txt
Writes  data/train.bin      and  data/val.bin

A .bin file is just a long row of uint16 numbers (2 bytes per token id), one
after another, with NO header. Token ids are below 4096, which fits easily in
uint16 (max 65535). The end-of-text token (id 256) sits between stories.

Speed: encoding 1.9 GB of text in pure Python is slow on one core, so the text
is cut into chunks and a pool of worker processes encodes chunks in parallel.
Chunks are cut ONLY right after an <|endoftext|> marker, so no word is ever
split in half and the result is identical to encoding everything in one go.

Run:  python scripts\prepare_data.py
Opts: --workers 12  --chunk-chars 2000000  --limit-chars 0 (0 = whole file)
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scratchlm.tokenizer import END_OF_TEXT, BPETokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Worker side: each worker process loads the tokenizer ONCE, then encodes
# whatever text chunks it is given.
# ---------------------------------------------------------------------------
_worker_tokenizer = None


def _init_worker(tokenizer_path):
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer.load(tokenizer_path)


def _encode_chunk(text):
    """Encode one text chunk -> raw bytes of a uint16 array."""
    ids = _worker_tokenizer.encode(text)
    return np.asarray(ids, dtype=np.uint16).tobytes()


# ---------------------------------------------------------------------------
# Main-process side
# ---------------------------------------------------------------------------

def iter_chunks(f, chunk_chars, limit_chars):
    """Yield text chunks of about `chunk_chars` characters from an open file.

    Every chunk ends exactly after an <|endoftext|> marker (except possibly
    the very last one), so chunk borders never fall inside a story.
    """
    carry = ""          # text read but not yet handed out
    total_read = 0
    while True:
        want = chunk_chars
        if limit_chars:
            want = min(want, limit_chars - total_read)
            if want <= 0:
                break
        block = f.read(want)
        if not block:
            break
        total_read += len(block)
        carry += block
        # Hand out everything up to and including the LAST marker we have.
        cut = carry.rfind(END_OF_TEXT)
        if cut != -1:
            cut += len(END_OF_TEXT)
            yield carry[:cut]
            carry = carry[cut:]
    if carry:
        yield carry  # leftover tail (normally empty)


def encode_file(in_path, out_path, tokenizer_path, workers, chunk_chars, limit_chars):
    """Encode one text file into one .bin file. Returns the number of tokens."""
    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".bin.part")   # never leave a half-written .bin
    total_tokens = 0
    total_bytes = in_path.stat().st_size
    if limit_chars:
        total_bytes = min(total_bytes, limit_chars)

    print(f"Encoding {in_path.name} -> {out_path.name} with {workers} workers")
    with open(in_path, "r", encoding="utf-8", newline="") as f, \
         open(tmp_path, "wb") as out, \
         Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool, \
         tqdm(total=total_bytes, unit="B", unit_scale=True, desc=in_path.name) as bar:

        chunks = iter_chunks(f, chunk_chars, limit_chars)
        while True:
            # Take a small batch of chunks (not the whole file: that would fill
            # RAM), encode them in parallel, write results IN ORDER.
            batch = []
            for chunk in chunks:
                batch.append(chunk)
                if len(batch) == workers * 2:
                    break
            if not batch:
                break
            for chunk, data in zip(batch, pool.map(_encode_chunk, batch)):
                out.write(data)
                total_tokens += len(data) // 2          # 2 bytes per token
                bar.update(len(chunk.encode("utf-8")))

    tmp_path.replace(out_path)
    print(f"  {total_tokens:,} tokens, {out_path.stat().st_size / 1e6:,.1f} MB")
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--tokenizer", default=str(REPO_ROOT / "data" / "tokenizer.json"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--chunk-chars", type=int, default=2_000_000)
    parser.add_argument("--limit-chars", type=int, default=0,
                        help="only encode the first N chars of each file (0 = all)")
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    jobs = [("train.txt", "train.bin"), ("validation.txt", "val.bin")]
    counts = {}
    for src, dst in jobs:
        counts[dst] = encode_file(raw_dir / src, out_dir / dst, args.tokenizer,
                                  args.workers, args.chunk_chars, args.limit_chars)

    # --- Sanity check: read the start of train.bin back and decode it -------
    tok = BPETokenizer.load(args.tokenizer)
    ids = np.fromfile(out_dir / "train.bin", dtype=np.uint16, count=60)
    print("\nFirst 60 tokens of train.bin decode to:")
    print(repr(tok.decode(ids.tolist())))
    print(f"max token id in this sample: {ids.max()} (vocab size {tok.vocab_size})")
    print("\nDone. Tokens: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))


if __name__ == "__main__":
    # This guard is REQUIRED on Windows: worker processes re-import this file,
    # and without it each worker would start its own pool and loop forever.
    main()
