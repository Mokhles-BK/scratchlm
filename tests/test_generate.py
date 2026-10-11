"""Tests for generation (Phase 5): KV cache and sampling. All run on CPU.

Run:  python -m pytest tests\test_generate.py -v
"""

import math

import pytest
import torch

from scratchlm.generate import filter_logits, generate, generate_text, sample_next_token
from scratchlm.model import GPT, GPTConfig, KVCache
from scratchlm.tokenizer import BPETokenizer

NEG_INF = float("-inf")


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


@pytest.fixture
def model():
    cfg = GPTConfig(vocab_size=100, block_size=48, n_layer=2, n_head=4, d_model=64, d_ff=96)
    return GPT(cfg).eval()


# --- KV cache --------------------------------------------------------------------

def test_cached_logits_match_full_forward(model):
    """Feeding tokens in pieces through the cache must give the SAME logits as
    one full forward pass. Covers all three attention paths: prompt (start=0),
    chunk after a cache, and one token after a cache."""
    x = torch.randint(0, 100, (2, 20))
    with torch.no_grad():
        full, _ = model(x)
        caches = model.make_caches()
        a, _ = model(x[:, :5], caches=caches)                 # prompt
        assert torch.allclose(a, full[:, :5], atol=1e-4)
        b, _ = model(x[:, 5:9], caches=caches)                # 4 new tokens after a cache
        assert torch.allclose(b, full[:, 5:9], atol=1e-4)
        for t in range(9, 20):                                # one token at a time
            c, _ = model(x[:, t:t + 1], caches=caches)
            assert torch.allclose(c[:, 0], full[:, t], atol=1e-4), f"position {t}"
    assert caches[0].length == 20


def test_cache_overflow_raises(model):
    caches = model.make_caches()
    with torch.no_grad():
        model(torch.randint(0, 100, (1, 48)), caches=caches)   # exactly fills it
        with pytest.raises(ValueError):
            model(torch.randint(0, 100, (1, 1)), caches=caches)


def test_kvcache_returns_everything_so_far():
    cache = KVCache(max_len=8)
    k1, v1 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k2, v2 = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    k, v = cache.update(k1, v1)
    assert k.shape == (1, 2, 3, 4)
    k, v = cache.update(k2, v2)
    assert k.shape == (1, 2, 4, 4) and cache.length == 4
    assert torch.equal(k[:, :, :3], k1) and torch.equal(k[:, :, 3:], k2)
    assert torch.equal(v[:, :, 3:], v2)


# --- sampling functions --------------------------------------------------------------

def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = filter_logits(logits, top_k=2)
    kept = (out > NEG_INF).nonzero()[:, 1].tolist()
    assert sorted(kept) == [1, 3]                             # the two largest


def test_top_p_keeps_smallest_group_reaching_p():
    # probabilities 0.5, 0.3, 0.2 (in this order)
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    assert (filter_logits(logits, top_p=0.4) > NEG_INF).sum() == 1   # 0.5 alone >= 0.4
    assert (filter_logits(logits, top_p=0.6) > NEG_INF).sum() == 2   # 0.5 + 0.3
    assert (filter_logits(logits, top_p=0.9) > NEG_INF).sum() == 3   # needs all three
    assert (filter_logits(logits, top_p=1.0) > NEG_INF).sum() == 3   # off


def test_top_p_keeps_tokens_in_the_right_places():
    # Unsorted input: best token is at index 2, second best at index 0.
    logits = torch.log(torch.tensor([[0.3, 0.1, 0.5, 0.1]]))
    out = filter_logits(logits, top_p=0.6)
    assert (out > NEG_INF).nonzero()[:, 1].tolist() == [0, 2]


def test_filter_does_not_modify_its_input():
    logits = torch.randn(2, 10)
    before = logits.clone()
    filter_logits(logits, top_k=3, top_p=0.5)
    assert torch.equal(logits, before)


def test_invalid_top_p_and_temperature_raise():
    with pytest.raises(ValueError):
        filter_logits(torch.randn(1, 5), top_p=0.0)
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(1, 5), temperature=-1.0)


def test_temperature_zero_is_greedy():
    logits = torch.tensor([[0.1, 2.0, 0.5], [3.0, 0.0, 1.0]])
    assert sample_next_token(logits, temperature=0).flatten().tolist() == [1, 0]


def test_top_k_1_is_greedy():
    logits = torch.randn(4, 50)
    got = sample_next_token(logits, temperature=1.0, top_k=1)
    assert torch.equal(got, logits.argmax(-1, keepdim=True))


def test_samples_match_softmax_distribution():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    probs = torch.softmax(logits, dim=-1)[0]
    g = torch.Generator().manual_seed(1)
    draws = torch.stack([sample_next_token(logits, 1.0, generator=g) for _ in range(20000)])
    freq = torch.bincount(draws.flatten(), minlength=4).float() / 20000
    assert torch.allclose(freq, probs, atol=0.02)


def test_temperature_changes_spread():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    g = torch.Generator().manual_seed(2)
    cold = {sample_next_token(logits, 0.05, generator=g).item() for _ in range(300)}
    hot = {sample_next_token(logits, 20.0, generator=g).item() for _ in range(300)}
    assert cold == {0}                                         # nearly always the best token
    assert len(hot) == 4                                       # all tokens show up


def test_sampling_respects_top_k():
    logits = torch.randn(1, 30)
    allowed = set(torch.topk(logits, 3).indices[0].tolist())
    g = torch.Generator().manual_seed(3)
    seen = {sample_next_token(logits, 1.0, top_k=3, generator=g).item() for _ in range(500)}
    assert seen <= allowed


# --- generate() -------------------------------------------------------------------------

def test_greedy_with_and_without_cache_are_identical(model):
    prompt = torch.randint(0, 100, (2, 6))
    a = generate(model, prompt, 25, temperature=0, use_cache=True)
    b = generate(model, prompt, 25, temperature=0, use_cache=False)
    assert a.shape == (2, 31)
    assert torch.equal(a, b)


def test_sampling_with_same_seed_is_identical_with_and_without_cache(model):
    prompt = torch.randint(0, 100, (1, 4))
    outs = []
    for use_cache in (True, False):
        g = torch.Generator().manual_seed(11)
        outs.append(generate(model, prompt, 20, temperature=0.9, top_k=20, top_p=0.9,
                             use_cache=use_cache, generator=g))
    assert torch.equal(outs[0], outs[1])


def test_prompt_is_kept_and_output_is_longer(model):
    prompt = torch.randint(0, 100, (3, 5))
    out = generate(model, prompt, 10)
    assert out.shape == (3, 15)
    assert torch.equal(out[:, :5], prompt)


def test_generation_stops_at_block_size(model):
    prompt = torch.randint(0, 100, (1, 40))
    out = generate(model, prompt, 100)                          # block_size is 48
    assert out.shape == (1, 48)


def test_prompt_too_long_or_empty_raises(model):
    with pytest.raises(ValueError):
        generate(model, torch.randint(0, 100, (1, 49)), 5)
    with pytest.raises(ValueError):
        generate(model, torch.zeros(1, 0, dtype=torch.long), 5)


def test_stops_when_every_sequence_hit_eos(model):
    # Zero the (tied) embedding: every logit becomes 0, so greedy always picks
    # token 0. Declare 0 the end token: both sequences finish after ONE step.
    with torch.no_grad():
        model.lm_head.weight[:] = 0.0
    prompt = torch.randint(1, 100, (2, 4))
    out = generate(model, prompt, 30, temperature=0, eos_id=0)
    assert out.shape == (2, 5)
    assert out[:, -1].tolist() == [0, 0]


def test_generation_restores_training_mode(model):
    model.train()
    generate(model, torch.randint(0, 100, (1, 3)), 3)
    assert model.training


def test_generate_under_bf16_autocast_runs(model):
    ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    out = generate(model, torch.randint(0, 100, (1, 4)), 8, autocast_ctx=ctx)
    assert out.shape == (1, 12)


# --- text level ------------------------------------------------------------------------

def test_generate_text_returns_prompt_plus_continuation():
    corpus = "Once upon a time there was a little girl named Lily. <|endoftext|>" * 50
    tok = BPETokenizer.train(corpus, vocab_size=300)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, d_model=32, d_ff=64)
    model = GPT(cfg).eval()

    a = generate_text(model, tok, "Once upon", max_new_tokens=15, seed=5)
    b = generate_text(model, tok, "Once upon", max_new_tokens=15, seed=5)
    c = generate_text(model, tok, "Once upon", max_new_tokens=15, seed=5, use_cache=False)
    assert a.startswith("Once upon")
    assert a == b == c                                          # reproducible, cache-independent
    assert "<|endoftext|>" not in a
    assert isinstance(generate_text(model, tok, "", max_new_tokens=5, seed=1), str)
