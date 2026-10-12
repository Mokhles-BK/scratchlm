# ScratchLM findings

Every claim points at a number or a sample. Commands are in `docs/PHASE6.md`.

## Setup (same for all runs)
- Data: TinyStories. Tokenizer: byte-level BPE trained from scratch, vocab 4096, 4.07 bytes/token.
- 32,768 tokens per optimiser step (micro-batch 8 x accum 8 x context 512). 3000 steps = 98.3M tokens.
- AdamW (betas 0.9/0.95, wd 0.1), 200 warmup steps, cosine to 10% of peak, grad clip 1.0, bf16 autocast.
- GPU: RTX 3050 Laptop 4 GB.
- Validation: `evaluate.py` walks the whole val file (4,873,728 tokens, 9,519 windows) in order.
  The "VAL" printed during training uses 50 random batches and is noisy (about +-0.02).

## Runs

| run | model | params | context | lr | steps | tokens | training VAL (noisy) | full val loss | perplexity | bits/byte | tok/s | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small-3k (A) | small | 15.7M | 512 | 1e-3 | 3000 | 98.3M | 1.5541 | **1.5719** | 4.82 | 0.5592 | 38.8k | 1,216 MB |
| base-3k (C) | base | 27.8M | 512 | 6e-4 | 3000 | 98.3M | 1.5276 | **1.5449** | 4.69 | 0.5496 | 30.1k | 1,681 MB |
| base-10k (D) | base | 27.8M | 512 | 6e-4 | stopped at ~210 of 10000 | | | not finished, resumable | | | | |

Run B (small at context 256) was started, stopped at step 303 and skipped (see Q2).

## Questions and answers

1. **Does a bigger model help at the same steps? By how much?**
   Yes, a little. Full val loss 1.5719 -> 1.5449 (-0.027, about 1.7%) for 77% more parameters,
   at 23% lower speed and +465 MB VRAM. Caveat: base also used a lower peak lr (6e-4 vs 1e-3), so
   this is "base with its default lr vs small", not size alone. The gap is the same in every
   position bucket (about 0.027).

2. **Does a shorter context cost quality?**
   Scoring run A with 256-token windows gives 1.6263 vs 1.5719 at 512. This looks like a gap but is a
   measurement artifact: per-position loss is the same (bucket 64-255: 1.535 vs 1.533). Shorter
   windows restart more often, and every restart pays the expensive early positions again.
   Lesson: when context lengths differ, compare per-position buckets, not the mean.
   Run B (training at 256) was therefore not worth GPU time.

3. **Does the model use its context?**
   Mostly the first ~64 tokens. Loss by position (run A): pos 0 = 3.858, 1-15 = 2.347, 16-63 = 1.719,
   64-255 = 1.533, 256-511 = 1.519. Run C: 3.832, 2.322, 1.693, 1.506, 1.492.
   After about 64 tokens the curve is nearly flat: stories are short, so long context adds little.

4. **How do samples change with training age?** (run C, same seed, prompt "Once upon a time")
   - step 500: grammatical sentences, no logic ("her mom said she was too cold ... not too late ... hotel").
   - step 1500: a consistent topic (beach, wet clothes) but it loops ("clothes were dirty / wet / clean / clean again").
   - step 3000: a full arc (seed -> garden -> monster -> mom protects -> happy ending), small oddities
     ("the monster was okay").
   Order of learning: grammar, then topic, then plot.

5. **Sampling settings** (`sample_grid.py`, distinct-3-gram table)
   - TODO: run `scripts\sample_grid.py --checkpoint checkpoints\base-3k\best.pt` and paste the table.

6. **What does the model still get wrong?**
   - Logic drift inside a story ("the seed is very important to me" after "small and fragile").
   - Repetition loops ("keep his special rock safe so he could keep it safe").
   - Similar openers across samples ("a little girl named Lily ... play outside"): TinyStories bias plus T=0.8.

## Other measurements
- KV cache: 77.1 -> 226.5 tokens/s (2.9x), identical greedy output with and without it.
- Both models are undertrained: train loss (about 1.5) is close to val loss, no overfitting,
  and val was still falling at step 3000. More tokens would help more than more parameters.

## Next things to test
- Run D: base for 10,000 steps (328M tokens). Prediction (a guess): full val about 1.45 and less looping.
- Same lr for small and base, to separate size from lr.
