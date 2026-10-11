"""Tests for the transformer model (Phase 3). All run on CPU.

Run:  python -m pytest tests\test_model.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

from scratchlm.model import (
    BASE, SMALL, CausalSelfAttention, GPT, GPTConfig, RMSNorm,
    apply_rope, build_rope_cache,
)


@pytest.fixture
def tiny_cfg():
    """A very small model so tests run in a blink."""
    return GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=4,
                     d_model=64, d_ff=96)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


# --- shapes ----------------------------------------------------------------

def test_output_shapes(tiny_cfg):
    model = GPT(tiny_cfg)
    idx = torch.randint(0, 100, (3, 20))
    logits, loss = model(idx)
    assert logits.shape == (3, 20, 100)
    assert loss is None
    logits, loss = model(idx, targets=idx)
    assert loss.shape == ()                      # a single number


def test_sequence_longer_than_block_size_raises(tiny_cfg):
    model = GPT(tiny_cfg)
    with pytest.raises(ValueError):
        model(torch.randint(0, 100, (1, tiny_cfg.block_size + 1)))


def test_bad_config_raises():
    with pytest.raises(ValueError):
        GPTConfig(d_model=100, n_head=8)         # 100 not divisible by 8
    with pytest.raises(ValueError):
        GPTConfig(d_model=24, n_head=8)          # head size 3 is odd


# --- parameter count --------------------------------------------------------

def expected_params(c):
    embedding = c.vocab_size * c.d_model                      # tied with output layer
    per_block = (4 * c.d_model ** 2                           # qkv (3) + out_proj (1)
                 + 3 * c.d_model * c.d_ff                     # gate, up, down
                 + 2 * c.d_model)                             # two RMSNorm gains
    return embedding + c.n_layer * per_block + c.d_model      # + final RMSNorm


def test_param_count_matches_formula(tiny_cfg):
    assert GPT(tiny_cfg).num_params() == expected_params(tiny_cfg)


@pytest.mark.parametrize("cfg", [SMALL, BASE], ids=["small", "base"])
def test_real_configs_are_in_target_range(cfg):
    model = GPT(cfg)
    n = model.num_params()
    assert n == expected_params(cfg)
    assert 10e6 <= n <= 40e6
    print(f"\n  {n:,} parameters")


def test_weight_tying(tiny_cfg):
    model = GPT(tiny_cfg)
    assert model.lm_head.weight is model.tok_emb.weight


# --- initial loss -------------------------------------------------------------

def test_initial_loss_is_about_log_vocab(tiny_cfg):
    # A model that knows nothing guesses uniformly: loss = ln(V).
    model = GPT(tiny_cfg)
    idx = torch.randint(0, 100, (8, 32))
    _, loss = model(idx, targets=torch.randint(0, 100, (8, 32)))
    assert abs(loss.item() - math.log(100)) < 0.3


# --- the key property: causality -------------------------------------------------

def test_causality_future_tokens_do_not_change_past_logits(tiny_cfg):
    model = GPT(tiny_cfg).eval()
    idx = torch.randint(0, 100, (2, 24))
    changed = idx.clone()
    k = 10
    changed[:, k:] = torch.randint(0, 100, (2, 24 - k))     # rewrite the future
    assert not torch.equal(idx[:, k:], changed[:, k:])

    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    # Positions 0..k-1 cannot see positions >= k, so their logits are identical.
    assert torch.allclose(a[:, :k], b[:, :k], atol=1e-5)
    # Sanity check that the test can fail: later positions DO change.
    assert not torch.allclose(a[:, k:], b[:, k:], atol=1e-5)


def test_changing_a_past_token_changes_later_logits(tiny_cfg):
    model = GPT(tiny_cfg).eval()
    idx = torch.randint(0, 100, (1, 16))
    changed = idx.clone()
    changed[0, 3] = (changed[0, 3] + 1) % 100
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-5)    # before: unchanged
    assert not torch.allclose(a[:, 3:], b[:, 3:], atol=1e-5)  # from there on: changed


# --- components ---------------------------------------------------------------

def test_rmsnorm_output_has_unit_rms():
    norm = RMSNorm(64)
    x = torch.randn(4, 10, 64) * 7 + 3
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_scores_depend_only_on_relative_distance():
    # Put the SAME q vector and the SAME k vector at every position. After
    # RoPE, score(m, n) = q_m . k_n must depend only on m - n.
    D, T = 16, 12
    cos, sin = build_rope_cache(D, T, base=10000.0)
    q = torch.randn(1, 1, 1, D).expand(1, 1, T, D).contiguous()
    k = torch.randn(1, 1, 1, D).expand(1, 1, T, D).contiguous()
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (qr @ kr.transpose(-1, -2))[0, 0]               # (T, T)
    for m in range(T - 1):
        for n in range(m + 1):
            assert torch.isclose(scores[m, n], scores[m + 1, n + 1], atol=1e-4)
    # And rotation keeps vector length unchanged.
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    cos, sin = build_rope_cache(8, 4, 10000.0)
    x = torch.randn(1, 1, 4, 8)
    assert torch.allclose(apply_rope(x, cos, sin)[:, :, 0], x[:, :, 0], atol=1e-6)


def test_attention_matches_manual_implementation(tiny_cfg):
    """Our fused-kernel attention == the textbook formula written out by hand."""
    attn = CausalSelfAttention(tiny_cfg).eval()
    B, T, C = 2, 12, tiny_cfg.d_model
    H, D = tiny_cfg.n_head, tiny_cfg.head_dim
    cos, sin = build_rope_cache(D, tiny_cfg.block_size, 10000.0)
    x = torch.randn(B, T, C)

    with torch.no_grad():
        got = attn(x, cos, sin)

        # --- manual version ---
        q, k, v = attn.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)       # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))     # hide the future
        weights = scores.softmax(dim=-1)
        y = (weights @ v).transpose(1, 2).reshape(B, T, C)
        want = attn.out_proj(y)

    assert torch.allclose(got, want, atol=1e-5)


# --- training signal -----------------------------------------------------------

def test_all_parameters_receive_gradients(tiny_cfg):
    model = GPT(tiny_cfg)
    idx = torch.randint(0, 100, (4, 16))
    _, loss = model(idx, targets=idx)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has bad gradient"


def test_bf16_autocast_forward_is_finite(tiny_cfg):
    model = GPT(tiny_cfg)
    idx = torch.randint(0, 100, (2, 16))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, loss = model(idx, targets=idx)
    assert torch.isfinite(loss)


def test_ignore_index_skips_positions(tiny_cfg):
    model = GPT(tiny_cfg).eval()
    idx = torch.randint(0, 100, (2, 16))
    targets = idx.clone()
    with torch.no_grad():
        logits, _ = model(idx)
        targets[:, 8:] = -100                                 # ignore the second half
        _, loss = model(idx, targets=targets)
        want = F.cross_entropy(logits[:, :8].reshape(-1, 100), idx[:, :8].reshape(-1))
    assert torch.allclose(loss, want, atol=1e-5)


def test_overfit_one_batch(tiny_cfg):
    """If the model, loss and optimiser are wired correctly, it must be able to
    memorise one small batch: loss should fall from ln(100)=4.6 to almost 0."""
    model = GPT(tiny_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    idx = torch.randint(0, 100, (4, 33))
    x, y = idx[:, :-1], idx[:, 1:]                            # next-token pairs
    first = None
    for step in range(300):
        _, loss = model(x, targets=y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"\n  loss {first:.3f} -> {loss.item():.4f}")
    assert first > 4.0
    assert loss.item() < 0.05
