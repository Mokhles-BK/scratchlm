r"""Byte-level BPE tokenizer, written from scratch (Phase 1).

THE BIG IDEA
============
A language model works on integers, not text. A tokenizer is the translator:

    text  --encode-->  [list of integer token ids]  --decode-->  text

Why "byte-level"?
    Every string can be turned into bytes with UTF-8 (values 0..255). If our
    base vocabulary is exactly those 256 byte values, then ANY text (English,
    Arabic, emoji, garbage) can be encoded. There is no "unknown token" and
    decode(encode(text)) == text always holds.

Why "BPE" (Byte Pair Encoding)?
    Raw bytes make sequences long (1 token per character). BPE shortens them:
    1. Start with every byte as its own token.
    2. Find the PAIR of neighbouring tokens that occurs most often.
    3. Give that pair a brand-new token id ("merge" them).
    4. Repeat until the vocabulary has the size we want (4096).
    Frequent chunks like " the", "ing", " and" end up as single tokens.

TOKEN ID LAYOUT (vocab size 4096)
    0   .. 255   the 256 raw bytes
    256          <|endoftext|>  (special token: marks the end of one story)
    257 .. 4095  learned merges, in the order they were learned

The learned merges list IS the tokenizer. Merge number i creates token id
(FIRST_MERGE_ID + i). Earlier merges are "stronger" (more frequent pairs) and
are applied first when encoding.

PRE-TOKENIZATION
    Before merging, text is split into "words" with a regex, and merges only
    happen INSIDE a word. Without this, BPE could glue the end of one word to
    the start of the next (e.g. "e t" from "the time"), wasting vocabulary.
    A word keeps its leading space: " cat" is one pre-token, "cat" another.
    That is why tokens like " the" exist.
"""

import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The one special token. In the raw text files it separates stories.
END_OF_TEXT = "<|endoftext|>"

# Simplified GPT-2 style pre-tokenizer pattern. Alternatives are tried left to
# right at each position, and together they cover EVERY possible character,
# so joining all matches gives back the original text exactly:
#   's|'t|...        English contractions: "don't" -> "don", "'t"
#    ?[A-Za-z]+      optional space + letters:  " hello"
#    ?[0-9]+         optional space + digits:   " 2024"
#    ?[^\sA-Za-z0-9]+ optional space + anything else (punctuation, emoji, é...)
#   \s+(?!\S)        whitespace run, except the last space if a word follows
#   \s+              any remaining whitespace
PRETOKEN_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


# ---------------------------------------------------------------------------
# Small helper used by both training and encoding
# ---------------------------------------------------------------------------

def merge_pair(ids, pair, new_id):
    """Return a copy of `ids` where every adjacent `pair` is replaced by `new_id`.

    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), new_id=9  ->  [9, 3, 9]
    """
    out = []
    i = 0
    while i < len(ids):
        # If the pair starts here, emit the new token and skip both members.
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE tokenizer: train, encode, decode, save, load."""

    def __init__(self, merges=None):
        """Build a tokenizer from an ordered list of merges.

        merges: list of (id_a, id_b) pairs. merges[i] creates token
                FIRST_MERGE_ID + i. An empty list gives a pure byte tokenizer.
        """
        self.merges = [tuple(m) for m in (merges or [])]

        # Special tokens sit right after the 256 bytes.
        self.special_tokens = {END_OF_TEXT: 256}
        self.first_merge_id = 256 + len(self.special_tokens)  # = 257

        # merge_ranks: pair -> (rank, new_id). Lower rank = learned earlier =
        # applied earlier when encoding.
        self.merge_ranks = {
            pair: (rank, self.first_merge_id + rank)
            for rank, pair in enumerate(self.merges)
        }

        # vocab: token id -> the bytes that token stands for. Used by decode.
        # A merged token's bytes are just its two parts glued together.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for text, tok_id in self.special_tokens.items():
            self.vocab[tok_id] = text.encode("utf-8")
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[self.first_merge_id + rank] = self.vocab[a] + self.vocab[b]

        # Regex that cuts the text at special tokens (kept as separate pieces).
        self._special_split = re.compile(
            "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        )
        # Encoding the same word again and again is wasteful; remember results.
        self._word_cache = {}

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def eot_id(self):
        """Token id of <|endoftext|>."""
        return self.special_tokens[END_OF_TEXT]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        """Learn merges from `text` until the vocabulary reaches `vocab_size`.

        Steps:
          1. Cut text into pre-tokens ("words") and count how often each
             distinct word appears. Training on distinct words with counts is
             MUCH faster than on the raw text, and gives the same result.
          2. Represent each word as a tuple of byte ids.
          3. Loop: count adjacent pairs (weighted by word frequency), merge
             the most frequent pair everywhere, record it.
        """
        first_merge_id = 256 + 1  # 256 bytes + 1 special token
        num_merges = vocab_size - first_merge_id
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {first_merge_id}")

        # The special token is never learned from, so remove it from the text.
        text = text.replace(END_OF_TEXT, "\n")

        # 1 + 2. word -> count, with each word as a tuple of byte ids.
        word_counts = Counter(PRETOKEN_PATTERN.findall(text))
        words = {tuple(w.encode("utf-8")): c for w, c in word_counts.items()}
        if verbose:
            print(f"  {len(word_counts):,} distinct words from {len(text):,} chars")

        merges = []
        for rank in range(num_merges):
            # 3a. Count every adjacent pair. A word seen c times adds c to
            #     each of its pairs.   shape: {(id_a, id_b): total_count}
            pair_counts = Counter()
            for word, count in words.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break  # every word is already a single token; nothing to merge

            # 3b. Pick the most frequent pair. Ties are broken by the smaller
            #     ids so training is deterministic (same text -> same merges).
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            new_id = first_merge_id + rank
            merges.append(best)

            # 3c. Apply the merge to every word.
            new_words = {}
            for word, count in words.items():
                merged = tuple(merge_pair(word, best, new_id))
                new_words[merged] = new_words.get(merged, 0) + count
            words = new_words

            if verbose and (rank + 1) % 250 == 0:
                print(f"  merge {rank + 1:>5}/{num_merges}  "
                      f"pair={best}  count={pair_counts[best]:,}")

        return cls(merges)

    # ------------------------------------------------------------------
    # Encoding: text -> ids
    # ------------------------------------------------------------------

    def _encode_word(self, word):
        """Encode ONE pre-token (a str) into token ids using the learned merges."""
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        ids = list(word.encode("utf-8"))  # start: one token per byte
        while len(ids) >= 2:
            # Among the adjacent pairs present, find the one learned EARLIEST
            # (lowest rank). Pairs we never learned get infinity.
            best_pair = None
            best_rank = float("inf")
            for pair in zip(ids, ids[1:]):
                entry = self.merge_ranks.get(pair)
                if entry is not None and entry[0] < best_rank:
                    best_rank = entry[0]
                    best_pair = pair
            if best_pair is None:
                break  # no learned merge applies any more
            ids = merge_pair(ids, best_pair, self.merge_ranks[best_pair][1])

        self._word_cache[word] = ids
        return ids

    def encode(self, text):
        """Turn a string into a list of token ids.

        The literal text "<|endoftext|>" becomes the single special id 256.
        """
        ids = []
        # re.split with a capture group keeps the special tokens as pieces.
        for piece in self._special_split.split(text):
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
            else:
                for word in PRETOKEN_PATTERN.findall(piece):
                    ids.extend(self._encode_word(word))
        return ids

    # ------------------------------------------------------------------
    # Decoding: ids -> text
    # ------------------------------------------------------------------

    def decode(self, ids):
        """Turn token ids back into a string.

        Look up the bytes of every id, glue them, then decode as UTF-8.
        errors="replace": a model may generate ids whose bytes split a
        multi-byte character; that shows as U+FFFD instead of crashing.
        """
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        """Write the tokenizer to a JSON file (the merges are all we need)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "scratchlm-byte-bpe",
            "version": 1,
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):
        """Read a tokenizer saved by `save`."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("type") != "scratchlm-byte-bpe":
            raise ValueError(f"{path} is not a ScratchLM tokenizer file")
        return cls([tuple(m) for m in payload["merges"]])
