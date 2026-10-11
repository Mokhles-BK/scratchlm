r"""Download the TinyStories training text into data/raw/.

Uses the Hugging Face `datasets` library ONLY as a downloader (no models).
Fetches the official TinyStories dataset (roneneldan/TinyStories), which has a
"train" split (~2.1M stories) and a "validation" split (~22k stories), and
writes each as one UTF-8 text file:

    data/raw/train.txt
    data/raw/validation.txt

Stories are separated by the marker  <|endoftext|>  (the same marker the
original TinyStories text files use). Stories themselves contain single
newlines, so a blank line would NOT be a safe separator. Phase 1 will turn
this marker into one special "end of text" token.

The script skips the download if both files already exist.

Run:  python scripts\download_data.py
"""

from pathlib import Path

from datasets import load_dataset

DATASET = "roneneldan/TinyStories"
SPLITS = ("train", "validation")

# Written between stories. See the module docstring for why it is not "\n\n".
STORY_SEPARATOR = "<|endoftext|>"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_files = {split: raw_dir / f"{split}.txt" for split in SPLITS}

    # Skip if everything is already present.
    if all(p.exists() and p.stat().st_size > 0 for p in out_files.values()):
        print("Already present - nothing to download:")
        for split, p in out_files.items():
            print(f"  {split:10s} {p.stat().st_size:>14,} bytes  {p}")
        return

    for split in SPLITS:
        out = out_files[split]
        print(f"Loading {DATASET} [{split}] (first run downloads; can take a few minutes)...")

        # One load_dataset call PER split: each returns a single Dataset whose
        # rows look like {"text": "Once upon a time, ..."}.
        split_ds = load_dataset(DATASET, split=split)

        n_stories = 0
        n_chars = 0
        # Write to a temp name first so an interrupted run never leaves a
        # half-written file that the "already present" check would trust.
        tmp = out.with_suffix(".txt.part")
        # newline="\n": stop Windows from turning every "\n" into "\r\n",
        # which would silently change the training text.
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for row in split_ds:
                text = row["text"].strip()
                if not text:
                    continue
                f.write(text)
                f.write(STORY_SEPARATOR)
                n_stories += 1
                n_chars += len(text)
        tmp.replace(out)

        print(f"  {split:10s} {n_stories:>10,} stories  {n_chars:>14,} chars  ->  {out}")

    print("\nDone. Quick look at the start of train.txt:")
    with open(out_files["train"], "r", encoding="utf-8", newline="") as f:
        print(repr(f.read(300)))


if __name__ == "__main__":
    main()
