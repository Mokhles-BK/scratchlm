"""Tests for Phase 6: evaluate.py, milestone checkpoints, compare_runs, sample_grid helpers.

Run:  python -m pytest tests\test_evaluate.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scratchlm.evaluate import (
    bits_per_byte, bucket_summary, distinct_ngram_ratio, evaluate_tokens, iter_windows,
)
from scratchlm.model import GPT, GPTConfig
from scratchlm.train import TrainConfig, train

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_runs import read_log, summarise  # noqa: E402


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


@pytest.fixture
def model():
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32, d_ff=64)
    return GPT(cfg).eval()


# --- window iteration ----------------------------------------------------------

def test_windows_cover_data_in_order_without_overlap():
    tokens = np.arange(100, dtype=np.uint16)
    wins = [(x, y) for x, y in iter_windows(tokens, block_size=8, batch_size=4)]
    xs = np.concatenate([x for x, _ in wins])
    ys = np.concatenate([y for _, y in wins])
    assert xs.shape == (12, 8)                     # (100 - 1) // 8 = 12 windows
    assert np.array_equal(xs.flatten(), np.arange(96))
    assert np.array_equal(ys, xs + 1)              # target = input shifted by one
    assert wins[-1][0].shape[0] == 4 and len(wins) == 3


def test_windows_last_batch_can_be_smaller_and_max_windows_works():
    tokens = np.arange(100, dtype=np.uint16)
    sizes = [x.shape[0] for x, _ in iter_windows(tokens, 8, batch_size=5)]
    assert sizes == [5, 5, 2]
    assert sum(x.shape[0] for x, _ in iter_windows(tokens, 8, 5, max_windows=7)) == 7


def test_windows_never_read_past_the_end():
    tokens = np.arange(17, dtype=np.uint16)        # exactly one window of 16 plus its target
    (x, y), = list(iter_windows(tokens, 16, 4))
    assert y[0, -1] == 16


# --- evaluation ------------------------------------------------------------------

def test_random_model_loss_is_about_ln_vocab(model):
    tokens = np.random.default_rng(0).integers(0, 64, 2000).astype(np.uint16)
    r = evaluate_tokens(model, tokens, batch_size=8)
    assert abs(r["loss"] - math.log(64)) < 0.3
    assert r["perplexity"] == pytest.approx(math.exp(r["loss"]))


def test_evaluation_is_deterministic(model):
    tokens = np.random.default_rng(1).integers(0, 64, 1000).astype(np.uint16)
    a = evaluate_tokens(model, tokens, batch_size=4)
    b = evaluate_tokens(model, tokens, batch_size=7)       # batch size must not matter
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)


def test_matches_models_own_loss(model):
    """Our per-token loss averaged must equal the loss the model computes with targets."""
    tokens = np.random.default_rng(2).integers(0, 64, 500).astype(np.uint16)
    r = evaluate_tokens(model, tokens, batch_size=100)     # one batch
    (x, y), = list(iter_windows(tokens, 16, 100))
    with torch.no_grad():
        _, ref = model(torch.from_numpy(x), targets=torch.from_numpy(y))
    assert r["loss"] == pytest.approx(ref.item(), abs=1e-5)


def test_loss_by_position_has_right_shape_and_mean(model):
    tokens = np.random.default_rng(3).integers(0, 64, 500).astype(np.uint16)
    r = evaluate_tokens(model, tokens)
    assert r["loss_by_position"].shape == (16,)
    assert r["loss_by_position"].mean() == pytest.approx(r["loss"])


def test_shorter_context_allowed_longer_rejected(model):
    tokens = np.random.default_rng(4).integers(0, 64, 500).astype(np.uint16)
    assert evaluate_tokens(model, tokens, block_size=8)["loss_by_position"].shape == (8,)
    with pytest.raises(ValueError):
        evaluate_tokens(model, tokens, block_size=32)


def test_too_little_data_raises(model):
    with pytest.raises(ValueError):
        evaluate_tokens(model, np.arange(10, dtype=np.uint16))


def test_evaluation_restores_training_mode(model):
    model.train()
    evaluate_tokens(model, np.zeros(100, dtype=np.uint16))
    assert model.training


def test_trained_model_beats_random_on_learnable_data(tmp_path):
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32, d_ff=64)
    pattern = np.tile(np.arange(16, dtype=np.uint16), 400)
    pattern.tofile(tmp_path / "train.bin")
    pattern.tofile(tmp_path / "val.bin")
    tcfg = TrainConfig(data_dir=str(tmp_path), out_dir=str(tmp_path / "run"), device="cpu",
                       micro_batch_size=4, grad_accum_steps=2, max_steps=40, lr=3e-3,
                       warmup_steps=5, eval_interval=20, eval_batches=3, ckpt_interval=20,
                       log_interval=10)
    train(tcfg, cfg, log=lambda *_: None)
    from scratchlm.train import load_model_from_checkpoint
    m = load_model_from_checkpoint(tmp_path / "run" / "latest.pt")
    r = evaluate_tokens(m, pattern)
    assert r["loss"] < math.log(64) - 1.0


# --- helpers ------------------------------------------------------------------------

def test_bucket_summary_names_and_values():
    by_pos = np.arange(300, dtype=float)
    s = bucket_summary(by_pos)
    assert s["0"] == 0.0
    assert s["1-15"] == pytest.approx(np.mean(np.arange(1, 16)))
    assert "256-299" in s
    assert list(bucket_summary(np.zeros(8))) == ["0", "1-7"]


def test_bits_per_byte():
    # loss ln(2) nats = 1 bit per token; 4 bytes per token -> 0.25 bits per byte
    assert bits_per_byte(math.log(2), 4.0) == pytest.approx(0.25)


def test_distinct_ngram_ratio():
    assert distinct_ngram_ratio([1, 2, 3, 4, 5], 3) == 1.0
    assert distinct_ngram_ratio([1, 2] * 10, 3) == pytest.approx(2 / 18)
    assert distinct_ngram_ratio([1, 2], 3) == 1.0


# --- milestones ------------------------------------------------------------------------

def test_milestone_checkpoints_are_kept(tmp_path):
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, d_model=32, d_ff=64)
    np.tile(np.arange(16, dtype=np.uint16), 200).tofile(tmp_path / "train.bin")
    np.tile(np.arange(16, dtype=np.uint16), 200).tofile(tmp_path / "val.bin")
    tcfg = TrainConfig(data_dir=str(tmp_path), out_dir=str(tmp_path / "run"), device="cpu",
                       micro_batch_size=2, grad_accum_steps=1, max_steps=12, lr=1e-3,
                       warmup_steps=2, eval_interval=6, eval_batches=2, ckpt_interval=6,
                       milestone_interval=4, log_interval=4)
    train(tcfg, cfg, log=lambda *_: None)
    names = sorted(p.name for p in (tmp_path / "run").glob("step_*.pt"))
    assert names == ["step_000004.pt", "step_000008.pt", "step_000012.pt"]


def test_no_milestones_by_default(tmp_path):
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, d_model=32, d_ff=64)
    np.tile(np.arange(16, dtype=np.uint16), 200).tofile(tmp_path / "train.bin")
    np.tile(np.arange(16, dtype=np.uint16), 200).tofile(tmp_path / "val.bin")
    tcfg = TrainConfig(data_dir=str(tmp_path), out_dir=str(tmp_path / "run"), device="cpu",
                       micro_batch_size=2, grad_accum_steps=1, max_steps=4, lr=1e-3,
                       warmup_steps=2, eval_interval=4, eval_batches=2, ckpt_interval=4, log_interval=4)
    train(tcfg, cfg, log=lambda *_: None)
    assert not list((tmp_path / "run").glob("step_*.pt"))


# --- compare_runs ------------------------------------------------------------------------

def test_read_log_and_summarise(tmp_path):
    csv_path = tmp_path / "log.csv"
    csv_path.write_text(
        "step,train_loss,val_loss,lr,grad_norm,tokens_seen,tokens_per_sec,peak_vram_mb\n"
        "10,3.0,,0.001,1.0,1000,500,100\n"
        "20,2.5,2.6,0.001,1.0,2000,700,120\n"
        "30,2.0,2.1,0.001,1.0,3000,600,120\n")
    d = read_log(csv_path)
    assert d["val_step"] == [20, 30] and d["val"] == [2.6, 2.1]
    s = summarise(d)
    assert s["final_val"] == 2.1 and s["best_val"] == 2.1
    assert s["tokens"] == 3000 and s["vram"] == 120 and s["tok_s"] == 600
