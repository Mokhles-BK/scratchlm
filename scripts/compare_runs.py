r"""Compare several training runs: table + overlay plots.

Reads checkpoints/<name>/log.csv for each run and writes one PNG with two
panels: validation loss against optimiser step, and against tokens seen.
(Against TOKENS is the fair x-axis when runs use different batch or context sizes.)

Run:  python scripts\compare_runs.py --names small-3k small-256 base-3k
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_log(path):
    """log.csv -> dict of lists. Validation columns only hold rows where val_loss was measured."""
    d = {"step": [], "tokens": [], "val_step": [], "val_tokens": [], "val": [],
         "tok_s": [], "vram": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            step, tokens = int(row["step"]), int(row["tokens_seen"])
            d["step"].append(step)
            d["tokens"].append(tokens)
            if row["tokens_per_sec"]:
                d["tok_s"].append(float(row["tokens_per_sec"]))
            d["vram"].append(float(row["peak_vram_mb"]))
            if row["val_loss"]:
                d["val_step"].append(step)
                d["val_tokens"].append(tokens)
                d["val"].append(float(row["val_loss"]))
    return d


def summarise(d):
    """One row of facts about a run."""
    speeds = sorted(d["tok_s"])
    median_speed = speeds[len(speeds) // 2] if speeds else 0.0
    return {
        "steps": d["step"][-1] if d["step"] else 0,
        "tokens": d["tokens"][-1] if d["tokens"] else 0,
        "final_val": d["val"][-1] if d["val"] else float("nan"),
        "best_val": min(d["val"]) if d["val"] else float("nan"),
        "tok_s": median_speed,
        "vram": max(d["vram"]) if d["vram"] else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True, help="run names under checkpoints/")
    p.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "compare.png"))
    a = p.parse_args()

    runs = {n: read_log(REPO_ROOT / "checkpoints" / n / "log.csv") for n in a.names}

    print(f"{'run':<16}{'steps':>7}{'tokens(M)':>11}{'final val':>11}{'best val':>10}{'tok/s':>9}{'VRAM MB':>9}")
    for n, d in runs.items():
        s = summarise(d)
        print(f"{n:<16}{s['steps']:>7}{s['tokens'] / 1e6:>11.1f}{s['final_val']:>11.4f}"
              f"{s['best_val']:>10.4f}{s['tok_s']:>9,.0f}{s['vram']:>9,.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for n, d in runs.items():
        ax1.plot(d["val_step"], d["val"], "o-", label=n, ms=3)
        ax2.plot([t / 1e6 for t in d["val_tokens"]], d["val"], "o-", label=n, ms=3)
    ax1.set_xlabel("optimiser step")
    ax2.set_xlabel("tokens seen (millions)")
    ax1.set_ylabel("validation loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.set_title("by step")
    ax2.set_title("by tokens seen")
    # Early validation losses are huge (random model); zoom on the useful part.
    ax1.set_ylim(top=3.0)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
