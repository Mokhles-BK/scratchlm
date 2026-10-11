"""Training-data loader: random batches from a memory-mapped token file (Phase 2).

THE IDEA
========
After prepare_data.py, a whole split lives in one file: a long row of uint16
token ids,   [t0, t1, t2, ..., t_{N-1}].

A language model learns NEXT-TOKEN PREDICTION: given tokens, predict the one
that follows. So one training example is a window of `block_size` tokens
(the input x) plus the same window shifted one step to the right (the target y):

    tokens :  t5  t6  t7  t8  t9 ...
    x      :  t5  t6  t7  t8            (block_size = 4)
    y      :  t6  t7  t8  t9            y[j] is the token that follows x[j]

One window therefore trains block_size predictions at once: after seeing t5
predict t6, after t5 t6 predict t7, and so on (the causal mask in the model
makes sure position j cannot peek at tokens after j).

MEMORY MAPPING
    np.memmap does not load the file into RAM. It lets us index the file like
    an array while the operating system loads only the pages we touch. That is
    how a 1 GB file works comfortably with 16 GB of RAM.

RANDOM WINDOWS
    Each batch picks `batch_size` random start positions. A window may span
    the boundary between two stories; the <|endoftext|> token (id 256) marks
    that boundary, so the model learns that a new story starts after it.
"""

from pathlib import Path

import numpy as np


class TokenDataset:
    """One split (train or val) stored as a uint16 .bin file."""

    def __init__(self, path, block_size, seed=0):
        """
        path       : the .bin file written by scripts/prepare_data.py
        block_size : context length T (how many tokens the model sees)
        seed       : seed for the random generator, so batches are reproducible
        """
        self.path = Path(path)
        self.block_size = block_size
        # mode="r": read-only. dtype must match how the file was written.
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        # A window needs block_size inputs plus 1 extra token for the last target.
        if len(self.tokens) < block_size + 2:
            raise ValueError(
                f"{self.path} has only {len(self.tokens)} tokens; need at least "
                f"block_size + 2 = {block_size + 2}"
            )
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """Number of tokens in the file."""
        return len(self.tokens)

    def sample(self, batch_size):
        """Draw a random batch as NumPy arrays (x, y), both int64, shape (B, T)."""
        T = self.block_size
        # Largest legal start s satisfies s + T + 1 <= N  (y ends at s + T).
        # rng.integers excludes its upper bound, so the bound is N - T.
        starts = self.rng.integers(0, len(self.tokens) - T, size=batch_size)
        # Each slice copies T tokens from disk into RAM.
        x = np.stack([self.tokens[s : s + T] for s in starts])          # (B, T)
        y = np.stack([self.tokens[s + 1 : s + 1 + T] for s in starts])  # (B, T)
        # Embedding layers and cross-entropy want int64 indices; uint16 is just
        # the storage format.
        return x.astype(np.int64), y.astype(np.int64)

    def get_batch(self, batch_size, device):
        """Like `sample`, but returns torch tensors already on `device`."""
        import torch  # imported here so the NumPy part works without torch

        x, y = self.sample(batch_size)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if torch.device(device).type == "cuda":
            # Pinned (page-locked) memory lets the copy to the GPU run
            # asynchronously while the GPU is busy with the previous step.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def open_splits(data_dir, block_size, seed=0):
    """Open data/train.bin and data/val.bin. Returns (train_ds, val_ds)."""
    data_dir = Path(data_dir)
    train = TokenDataset(data_dir / "train.bin", block_size, seed=seed)
    # Different seed so validation batches are not a copy of training batches.
    val = TokenDataset(data_dir / "val.bin", block_size, seed=seed + 1)
    return train, val
