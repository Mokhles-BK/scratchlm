"""Tests for the data pipeline (Phase 2): prepare_data.py and scratchlm/data.py.

Run:  python -m pytest tests\test_data.py -v
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scratchlm.data import TokenDataset, open_splits
from scratchlm.tokenizer import END_OF_TEXT, BPETokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

STORY = ("Once upon a time, a little girl named Lily had a red ball. "
         "She played in the park with her dog.\nIt was fun!")


# --- loader ---------------------------------------------------------------

@pytest.fixture
def bin_path(tmp_path):
    """A token file whose content is 0,1,2,... so shifts are easy to check."""
    path = tmp_path / "toy.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    return path


def test_shapes_and_dtype(bin_path):
    ds = TokenDataset(bin_path, block_size=16, seed=0)
    x, y = ds.sample(batch_size=8)
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == np.int64 and y.dtype == np.int64
    assert len(ds) == 1000


def test_target_is_input_shifted_by_one(bin_path):
    ds = TokenDataset(bin_path, block_size=16, seed=0)
    x, y = ds.sample(batch_size=32)
    # In the toy file token value == position, so y must equal x + 1 ...
    assert np.array_equal(y, x + 1)
    # ... and, in general, y[:, :-1] == x[:, 1:] (shifted window).
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_windows_stay_inside_file(bin_path):
    ds = TokenDataset(bin_path, block_size=16, seed=0)
    for _ in range(300):
        x, y = ds.sample(batch_size=64)
        assert x.min() >= 0 and y.max() <= 999


def test_last_window_is_reachable(tmp_path):
    # With N = block_size + 2 there are exactly 2 legal windows: start 0 and 1.
    path = tmp_path / "tiny.bin"
    np.arange(10, dtype=np.uint16).tofile(path)
    ds = TokenDataset(path, block_size=8, seed=0)
    starts = {int(ds.sample(1)[0][0, 0]) for _ in range(200)}
    assert starts == {0, 1}


def test_same_seed_same_batches_different_seed_differs(bin_path):
    a = TokenDataset(bin_path, 16, seed=5).sample(8)[0]
    b = TokenDataset(bin_path, 16, seed=5).sample(8)[0]
    c = TokenDataset(bin_path, 16, seed=6).sample(8)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_too_short_file_raises(tmp_path):
    path = tmp_path / "short.bin"
    np.arange(5, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        TokenDataset(path, block_size=8)


def test_open_splits(tmp_path):
    for name in ("train.bin", "val.bin"):
        np.arange(100, dtype=np.uint16).tofile(tmp_path / name)
    train, val = open_splits(tmp_path, block_size=8)
    assert len(train) == 100 and len(val) == 100


def test_get_batch_torch(bin_path):
    torch = pytest.importorskip("torch")
    ds = TokenDataset(bin_path, block_size=16, seed=0)
    x, y = ds.get_batch(4, device="cpu")
    assert isinstance(x, torch.Tensor)
    assert x.shape == (4, 16) and x.dtype == torch.int64
    assert torch.equal(y, x + 1)


# --- prepare_data.py end to end ---------------------------------------------

def test_prepare_data_matches_direct_encoding(tmp_path):
    """Run the real script on tiny files; chunked parallel output must be
    identical to encoding the whole text in one call."""
    train_text = (STORY + END_OF_TEXT) * 40
    val_text = (STORY + END_OF_TEXT) * 5
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.txt").write_text(train_text, encoding="utf-8", newline="")
    (raw / "validation.txt").write_text(val_text, encoding="utf-8", newline="")

    tok = BPETokenizer.train(train_text, vocab_size=300)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "prepare_data.py"),
         "--raw-dir", str(raw), "--out-dir", str(tmp_path),
         "--tokenizer", str(tok_path),
         "--workers", "2", "--chunk-chars", "300"],   # tiny chunks: many borders
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    for name, text in (("train.bin", train_text), ("val.bin", val_text)):
        got = np.fromfile(tmp_path / name, dtype=np.uint16)
        assert got.tolist() == tok.encode(text)
        assert tok.decode(got.tolist()) == text
        assert (got == tok.eot_id).sum() == text.count(END_OF_TEXT)
