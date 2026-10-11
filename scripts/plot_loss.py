r"""Plot the loss curve of a training run.

Reads checkpoints/<name>/log.csv (written by scratchlm.train) and saves
checkpoints/<name>/loss.png: training loss and validation loss against step.

Run:  python scripts\plot_loss.py --name small-run
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run", help="run name under checkpoints/")
    parser.add_argument("--csv", default=None, help="explicit path to a log.csv")
    parser.add_argument("--out", default=None, help="explicit output png path")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else REPO_ROOT / "checkpoints" / args.name / "log.csv"
    out_path = Path(args.out) if args.out else csv_path.with_name("loss.png")

    steps, train_loss, val_steps, val_loss = [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_steps.append(int(row["step"]))
                val_loss.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_loss, label="train loss", alpha=0.6)
    if val_loss:
        ax.plot(val_steps, val_loss, "o-", label="validation loss")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"ScratchLM training: {csv_path.parent.name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")
    if val_loss:
        print(f"Last validation loss: {val_loss[-1]:.4f} (best {min(val_loss):.4f})")


if __name__ == "__main__":
    main()
