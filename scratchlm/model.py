r"""Decoder-only transformer (GPT/Llama style), written from scratch (Phase 3).

WHAT THE MODEL DOES
===================
Input : a batch of token ids, shape (B, T)        B = batch, T = sequence length
Output: logits, shape (B, T, V)                   V = vocab size (4096)
        logits[b, t] = a score for every possible NEXT token after position t.
Training makes the score of the true next token high (cross-entropy loss).

ARCHITECTURE (one picture)
==========================

    token ids (B, T)
        |
    token embedding            id -> vector of size d_model         (B, T, C)
        |
    Block x n_layer            each block does:
        |                        x = x + Attention(RMSNorm(x))      "look at the past"
        |                        x = x + SwiGLU(RMSNorm(x))         "think about it"
        |
    RMSNorm                    final normalisation
        |
    output layer (tied)        vector -> V scores                   (B, T, V)

Notation used in shape comments:
    B = batch size, T = sequence length, C = d_model (embedding size),
    H = number of attention heads, D = C / H = size of one head,
    V = vocab size, F = hidden size of the MLP (d_ff).

Choices (all standard in modern LLMs):
    * RMSNorm instead of LayerNorm        (simpler, no mean subtraction, no bias)
    * RoPE instead of learned positions   (positions are encoded by ROTATING q and k)
    * SwiGLU MLP instead of GELU MLP      (a gated MLP, works a bit better)
    * No biases in any linear layer
    * Pre-norm: normalise BEFORE each sub-layer, keep the residual path clean
    * Weight tying: the output layer reuses the token embedding matrix
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """All the numbers that define the model's shape. Defaults = the BASE model."""

    vocab_size: int = 4096     # V: must match the tokenizer
    block_size: int = 512      # T_max: longest context the model supports
    n_layer: int = 8           # number of transformer blocks
    n_head: int = 8            # H: attention heads per block
    d_model: int = 512         # C: width of the residual stream
    d_ff: int = 1408           # F: hidden width of the SwiGLU MLP (about 8/3 * C)
    rope_base: float = 10000.0 # RoPE frequency base
    norm_eps: float = 1e-5     # RMSNorm epsilon (avoids dividing by zero)

    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("head size (d_model / n_head) must be even for RoPE")

    @property
    def head_dim(self):
        return self.d_model // self.n_head


# Two ready-made sizes (about 15.7M and 27.8M parameters).
SMALL = GPTConfig(d_model=384, n_layer=8, n_head=6, d_ff=1024)
BASE = GPTConfig(d_model=512, n_layer=8, n_head=8, d_ff=1408)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square normalisation.

    For each token vector x (size C):   y = x / sqrt(mean(x^2) + eps) * g
    It rescales the vector to have root-mean-square 1, then multiplies each
    channel by a learned gain g. (LayerNorm also subtracts the mean; RMSNorm
    skips that and works just as well.)
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # g, shape (C,)

    def forward(self, x):                      # x: (B, T, C)
        # Do the math in float32 even under bf16 autocast: squares and means
        # lose too much precision in bf16.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight     # (B, T, C)


# ---------------------------------------------------------------------------
# RoPE (rotary position embedding)
# ---------------------------------------------------------------------------

def build_rope_cache(head_dim, max_len, base):
    """Precompute cos/sin tables for every position and frequency.

    IDEA: attention compares a query q (at position m) with a key k (at
    position n) using a dot product. RoPE first ROTATES q and k by an angle
    proportional to their position. Rotating both makes the dot product
    depend only on the DISTANCE m - n, so the model learns relative positions
    without any learned position table.

    The head vector (size D) is cut into D/2 pairs. Pair i is rotated with its
    own frequency   theta_i = base^(-2i / D)   (fast for small i, slow for
    large i), by angle   position * theta_i.

    Returns cos, sin, each of shape (max_len, D/2).
    """
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (D/2,)
    positions = torch.arange(max_len).float()                                  # (T_max,)
    angles = torch.outer(positions, theta)                                     # (T_max, D/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair of channels of x by its position-dependent angle.

    x      : (B, H, T, D)
    cos/sin: (T, D/2)   (already cut to the first T positions)

    Pair i is made of channels (i, i + D/2). A 2D rotation by angle a is
        new1 = x1 * cos(a) - x2 * sin(a)
        new2 = x1 * sin(a) + x2 * cos(a)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]                  # each (B, H, T, D/2)
    rotated = torch.cat([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)     # (B, H, T, D)
    return rotated.type_as(x)


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token only sees tokens before it.

    For every token we make three vectors with learned projections:
        query q : "what am I looking for?"
        key   k : "what do I contain?"
        value v : "what do I hand over if someone attends to me?"
    The attention weight of token t on token s is softmax(q_t . k_s / sqrt(D)),
    and the output is the weighted sum of the values. "Causal" means s <= t:
    a token may never look at the future (otherwise predicting the next token
    would be cheating).

    "Multi-head": we do this H times in parallel with smaller vectors (size D),
    so different heads can specialise (e.g. one tracks the previous word,
    another tracks the subject of the sentence). Results are concatenated.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        # One matrix produces q, k and v at once: C -> 3C.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # Mixes the concatenated heads back together: C -> C.
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):                      # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)           # each (B, T, C)

        # Cut channels into heads: (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Inject position information by rotating q and k (NOT v).
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # Fused attention kernel. is_causal=True applies the lower-triangular
        # mask for us. Mathematically:
        #   scores = q @ k^T / sqrt(D)          (B, H, T, T)
        #   scores[t, s] = -inf for s > t       (the causal mask)
        #   y = softmax(scores) @ v             (B, H, T, D)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Glue the heads back together: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)                          # (B, T, C)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Gated feed-forward network applied to every token independently.

        out = down( silu(gate(x)) * up(x) )

    `gate` and `up` expand C -> F. silu(z) = z * sigmoid(z) is a smooth
    ReLU-like function. Multiplying silu(gate) by up lets the network decide,
    channel by channel, how much of `up` to let through (the "gate").
    `down` then shrinks F -> C. Attention moves information BETWEEN tokens;
    this MLP processes the information WITHIN each token.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):                                # x: (B, T, C)
        return self.down(F.silu(self.gate(x)) * self.up(x))  # (B, T, C)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """attention + MLP, each wrapped in a residual connection.

    The residual form  x = x + f(norm(x))  means every sub-layer only ADDS a
    correction to the stream. Gradients flow straight through the "+",
    which is what makes deep networks trainable.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):                      # x: (B, T, C)
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Token embedding -> n_layer blocks -> RMSNorm -> output scores."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # (V, C)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # (C -> V)

        # WEIGHT TYING: the output layer uses the SAME matrix as the embedding.
        # Intuition: the vector that stands for token "cat" on the way in is
        # also the direction the model should output to predict "cat". It
        # saves V*C parameters and usually helps small models.
        self.lm_head.weight = self.tok_emb.weight

        # RoPE tables are constants, not parameters. persistent=False keeps
        # them out of checkpoints (they are rebuilt from the config).
        cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)   # (T_max, D/2)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual-branch output projections get a smaller init. Every block
        # adds two things to the residual stream; with n_layer blocks the
        # variance would otherwise grow with depth. Scaling by 1/sqrt(2*n_layer)
        # keeps the stream's size roughly constant at the start (GPT-2 trick).
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(module):
        """Small random start: every weight ~ Normal(0, 0.02)."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        """Total trainable parameters. The tied matrix is counted once."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, ignore_index=-100):
        """
        idx     : (B, T) int64 token ids
        targets : (B, T) int64 next-token ids, or None
        returns : logits (B, T, V), and the mean cross-entropy loss if
                  targets were given (else None). Positions whose target equals
                  ignore_index are skipped in the loss (used later for chat
                  fine-tuning).
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")

        x = self.tok_emb(idx)                            # (B, T, C)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)   # (B, T, C)
        x = self.norm_f(x)
        logits = self.lm_head(x)                         # (B, T, V)

        loss = None
        if targets is not None:
            # cross_entropy wants (N, V) scores and (N,) class ids.
            # float() so the softmax/log runs in full precision under autocast.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss
