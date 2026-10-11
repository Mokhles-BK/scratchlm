"""Tests for the training loop (Phase 4). All run on CPU.

Run:  python -m pytest tests\test_train.py -v
"""

import math
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scratchlm import train as train_mod
from scratchlm.model import GPT, GPTConfig
from scratchlm.train import (
    TrainConfig, check_gpu_free, get_lr, load_model_from_checkpoint,
    make_optimizer, run_micro_batches, train,
)

import contextlib


@pytest.fixture
def model_cfg():
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32, d_ff=64)


@pytest.fixture
def data_dir(tmp_path):
    """Toy 'language': the sequence 0,1,2,...,15 repeated. Perfectly learnable."""
    pattern = np.tile(np.arange(16, dtype=np.uint16), 400)
    pattern.tofile(tmp_path / "train.bin")
    pattern.tofile(tmp_path / "val.bin")
    return tmp_path


def make_tcfg(data_dir, out_dir, **kw):
    base = dict(data_dir=str(data_dir), out_dir=str(out_dir), device="cpu",
                micro_batch_size=4, grad_accum_steps=2, max_steps=40, lr=3e-3,
                warmup_steps=5, eval_interval=10, eval_batches=5, ckpt_interval=10,
                log_interval=5, seed=7)
    base.update(kw)
    return TrainConfig(**base)


# --- learning-rate schedule ----------------------------------------------------

def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1e-3, min_lr_ratio=0.1, warmup_steps=100, max_steps=1000)
    assert get_lr(0, cfg) == pytest.approx(1e-3 / 100)          # tiny at the start
    assert get_lr(99, cfg) == pytest.approx(1e-3)               # reaches the peak
    assert get_lr(100, cfg) == pytest.approx(1e-3)              # cosine starts at the peak
    mid = 100 + (1000 - 100) // 2
    assert get_lr(mid, cfg) == pytest.approx((1e-3 + 1e-4) / 2, rel=1e-3)  # half-way
    assert get_lr(1000, cfg) == pytest.approx(1e-4)             # floor
    assert get_lr(5000, cfg) == pytest.approx(1e-4)             # stays at the floor
    after = [get_lr(s, cfg) for s in range(100, 1001)]
    assert all(a >= b - 1e-12 for a, b in zip(after, after[1:]))  # never goes up again


# --- optimiser groups ------------------------------------------------------------

def test_weight_decay_groups(model_cfg):
    model = GPT(model_cfg)
    opt = make_optimizer(model, TrainConfig(weight_decay=0.1), "cpu")
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])      # matrices decay
    assert all(p.dim() == 1 for p in no_decay["params"])   # norm gains do not
    n = len(decay["params"]) + len(no_decay["params"])
    assert n == len(list(model.parameters()))              # every parameter, once
    assert len({id(p) for g in opt.param_groups for p in g["params"]}) == n


# --- gradient accumulation ---------------------------------------------------------

def test_accumulation_equals_one_big_batch(model_cfg):
    torch.manual_seed(0)
    model_a = GPT(model_cfg)
    model_b = GPT(model_cfg)
    model_b.load_state_dict(model_a.state_dict())
    x = torch.randint(0, 64, (8, 16))
    y = torch.randint(0, 64, (8, 16))
    ctx = contextlib.nullcontext()

    loss_a = run_micro_batches(model_a, [(x, y)], ctx)                       # 1 x 8
    micro = [(x[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2)]
    loss_b = run_micro_batches(model_b, micro, ctx)                          # 4 x 2

    assert torch.allclose(loss_a, loss_b, atol=1e-5)
    for (name, pa), pb in zip(model_a.named_parameters(), model_b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5), name


# --- end to end ------------------------------------------------------------------------

def test_training_lowers_loss_and_writes_files(data_dir, model_cfg, tmp_path):
    out = tmp_path / "run"
    result = train(make_tcfg(data_dir, out), model_cfg, log=lambda *_: None)
    first, last = result["history"][0][1], result["history"][-1][1]
    assert last < first - 0.5, f"val loss did not fall: {first:.3f} -> {last:.3f}"
    assert (out / "latest.pt").exists() and (out / "best.pt").exists()
    lines = (out / "log.csv").read_text().strip().splitlines()
    assert lines[0].startswith("step,train_loss,val_loss")
    assert len(lines) > 5


def test_refuses_to_overwrite_existing_run(data_dir, model_cfg, tmp_path):
    out = tmp_path / "run"
    train(make_tcfg(data_dir, out, max_steps=5, warmup_steps=2, eval_interval=5),
          model_cfg, log=lambda *_: None)
    with pytest.raises(FileExistsError):
        train(make_tcfg(data_dir, out), model_cfg, log=lambda *_: None)


def test_resume_without_checkpoint_raises(data_dir, model_cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        train(make_tcfg(data_dir, tmp_path / "none"), model_cfg, resume=True, log=lambda *_: None)


def test_resume_matches_uninterrupted_run(data_dir, model_cfg, tmp_path):
    quiet = lambda *_: None
    # Run A: 20 steps straight through.
    out_a = tmp_path / "a"
    train(make_tcfg(data_dir, out_a, max_steps=20, warmup_steps=4), model_cfg, log=quiet)
    # Run B: stop at step 8, then resume to step 20.
    out_b = tmp_path / "b"
    cfg_b = make_tcfg(data_dir, out_b, max_steps=20, warmup_steps=4)
    train(cfg_b, model_cfg, stop_at_step=8, log=quiet)
    res = train(cfg_b, model_cfg, resume=True, log=quiet)
    assert res["step"] == 20

    a = torch.load(out_a / "latest.pt", weights_only=True)["model"]
    b = torch.load(out_b / "latest.pt", weights_only=True)["model"]
    assert a.keys() == b.keys()
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-6), f"{name} differs after resume"


def test_checkpoint_loads_into_a_model(data_dir, model_cfg, tmp_path):
    out = tmp_path / "run"
    train(make_tcfg(data_dir, out, max_steps=5, warmup_steps=2, eval_interval=5),
          model_cfg, log=lambda *_: None)
    model = load_model_from_checkpoint(out / "latest.pt")
    logits, _ = model(torch.randint(0, 64, (1, 8)))
    assert logits.shape == (1, 8, 64)


# --- GPU guard --------------------------------------------------------------------------------

def fake_smi(stdout):
    def run(*args, **kwargs):
        return SimpleNamespace(stdout=stdout)
    return run


def test_gpu_check_passes_when_gpu_is_free(monkeypatch):
    monkeypatch.setattr(train_mod.subprocess, "run", fake_smi("506\n"))
    assert check_gpu_free() == 506


def test_gpu_check_stops_when_gpu_is_busy(monkeypatch):
    monkeypatch.setattr(train_mod.subprocess, "run", fake_smi("2048\n"))
    with pytest.raises(RuntimeError, match="2048"):
        check_gpu_free()


def test_gpu_check_stops_when_nvidia_smi_missing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(train_mod.subprocess, "run", boom)
    with pytest.raises(RuntimeError):
        check_gpu_free()
