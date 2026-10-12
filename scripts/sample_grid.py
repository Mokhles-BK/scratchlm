r"""Write what a checkpoint says under many sampling settings, for side-by-side reading.

Numbers (loss) say how good a model is on average. Reading text says WHAT is
good or bad about it. This script runs the same prompts under a grid of
settings and writes a Markdown file you can read and quote in FINDINGS.md.
Same seeds everywhere, so differences come from the settings, not from luck.

Run:  python scripts\sample_grid.py --checkpoint checkpoints\small-3k\best.pt
      python scripts\sample_grid.py --checkpoint checkpoints\small-3k\best.pt --out docs\grid-small.md
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scratchlm.evaluate import distinct_ngram_ratio  # noqa: E402
from scratchlm.generate import generate_text  # noqa: E402
from scratchlm.tokenizer import BPETokenizer  # noqa: E402
from scratchlm.train import check_gpu_free, load_model_from_checkpoint  # noqa: E402

PROMPTS = [
    "Once upon a time",
    "Tom and his dog went to the park. Suddenly",
    "The little robot wanted to",
]

# (label, temperature, top_k, top_p)
SETTINGS = [
    ("greedy (T=0)", 0.0, None, None),
    ("T=0.5", 0.5, 40, 0.95),
    ("T=0.8 (default)", 0.8, 40, 0.95),
    ("T=1.0", 1.0, 40, 0.95),
    ("T=1.0 no filters", 1.0, None, None),
    ("T=1.3", 1.3, 40, 0.95),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=str(REPO_ROOT / "data" / "tokenizer.json"))
    p.add_argument("--out", default=None, help="Markdown output (default: next to the checkpoint)")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--samples", type=int, default=2, help="samples per (prompt, setting)")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    try:
        if a.device.startswith("cuda"):
            check_gpu_free()
        tok = BPETokenizer.load(a.tokenizer)
        model = load_model_from_checkpoint(a.checkpoint, device=a.device).eval()
    except (RuntimeError, OSError, ValueError) as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        sys.exit(1)

    out = Path(a.out) if a.out else Path(a.checkpoint).with_name(f"grid_{Path(a.checkpoint).stem}.md")
    lines = [f"# Sampling grid: `{a.checkpoint}`", ""]
    summary = []
    for label, temp, top_k, top_p in SETTINGS:
        ratios = []
        lines += [f"## {label}", ""]
        for prompt in PROMPTS:
            # greedy has no randomness: one sample is enough
            for i in range(1 if temp == 0 else a.samples):
                text = generate_text(model, tok, prompt, a.max_new_tokens, temp, top_k, top_p,
                                     seed=a.seed + i)
                ratios.append(distinct_ngram_ratio(tok.encode(text), 3))
                lines += [f"**{prompt!r}** (sample {i + 1})", "", "> " + text.replace("\n", "\n> "), ""]
        mean = sum(ratios) / len(ratios)
        summary.append((label, mean))
        lines += [f"*distinct 3-grams: {mean:.3f}*", ""]

    lines += ["## Summary: distinct 3-gram ratio (1.0 = no repetition)", "",
              "| setting | distinct 3-grams |", "|---|---|"]
    lines += [f"| {l} | {m:.3f} |" for l, m in summary]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'setting':<20}distinct 3-grams")
    for l, m in summary:
        print(f"{l:<20}{m:.3f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
