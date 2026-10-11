"""Tests for the byte-level BPE tokenizer (Phase 1).

Run:  python -m pytest tests\test_tokenizer.py -v
"""

import random

import pytest

from scratchlm.tokenizer import END_OF_TEXT, PRETOKEN_PATTERN, BPETokenizer, merge_pair

# A tiny TinyStories-like training text, repeated so merges have real counts.
TRAIN_TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved her red ball. "
    "One day, Lily and her dog went to the park. \"Look, Mom!\" she said. "
    "The dog didn't want to go home, but it was late. <|endoftext|>"
) * 200


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TRAIN_TEXT, vocab_size=330)


# --- helper ---------------------------------------------------------------

def test_merge_pair_basic():
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]  # left to right, no overlap
    assert merge_pair([], (1, 2), 9) == []


# --- pre-tokenizer ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Hello world", "  two  spaces ", "tabs\tand\nnewlines\n\n", "don't stop",
    "numbers 123 and 4.5", "é à ü 你好 🙂", "", "   ",
])
def test_pretokenizer_covers_every_character(text):
    # Joining the pieces must give the text back, or we would lose characters.
    assert "".join(PRETOKEN_PATTERN.findall(text)) == text


# --- round trip (the most important property) -----------------------------

ROUND_TRIP_CASES = [
    "",
    "a",
    "Hello, world!",
    "Once upon a time, there was a little girl.",
    "  leading and trailing spaces  ",
    "multiple\n\nnewlines\r\nand\ttabs",
    "Unicode: café, naïve, 你好, مرحبا, 🙂🚀",
    "Numbers: 0 1 22 333 4444 and punctuation !?.,;:'\"()",
    "It's we're they've I'm you'll he'd",
    "A" * 500,
    "<|endoftext|>",
    "story one<|endoftext|>story two<|endoftext|>",
]


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_random_unicode(tok):
    rng = random.Random(0)
    for _ in range(200):
        # Random code points, skipping the surrogate range (not valid UTF-8).
        chars = [chr(rng.choice([rng.randint(32, 126), rng.randint(160, 0x2FFF),
                                 rng.randint(0x1F300, 0x1F64F)]))
                 for _ in range(rng.randint(0, 40))]
        text = "".join(chars)
        assert tok.decode(tok.encode(text)) == text


def test_untrained_tokenizer_is_pure_bytes():
    raw = BPETokenizer()
    ids = raw.encode("hi é")
    assert ids == list("hi é".encode("utf-8"))
    assert raw.decode(ids) == "hi é"


# --- vocabulary layout -------------------------------------------------------

def test_vocab_layout(tok):
    assert tok.vocab_size == 330
    assert tok.eot_id == 256
    assert tok.first_merge_id == 257
    assert len(tok.merges) == 330 - 257
    assert tok.vocab[65] == b"A"                      # bytes are ids 0..255
    assert tok.vocab[256] == END_OF_TEXT.encode()     # special token
    # Every merged token is exactly its two parts glued together.
    a, b = tok.merges[0]
    assert tok.vocab[257] == tok.vocab[a] + tok.vocab[b]


def test_all_ids_are_in_range(tok):
    ids = tok.encode(TRAIN_TEXT[:2000] + "é🙂")
    assert all(0 <= i < tok.vocab_size for i in ids)


# --- special token -----------------------------------------------------------

def test_end_of_text_is_one_token(tok):
    assert tok.encode(END_OF_TEXT) == [256]
    ids = tok.encode("hello" + END_OF_TEXT + "world")
    assert ids.count(256) == 1
    assert tok.decode(ids) == "hello<|endoftext|>world"


# --- BPE behaviour --------------------------------------------------------------

def test_compression(tok):
    text = "Once upon a time, there was a little girl named Lily."
    assert len(tok.encode(text)) < len(text.encode("utf-8")) * 0.6


def test_frequent_word_becomes_one_token(tok):
    # " Lily" appears 400 times in the training text; it must be learned.
    assert len(tok.encode(" Lily")) == 1


def test_merges_do_not_cross_word_boundaries(tok):
    # Every token (except special) must lie inside ONE pre-token.
    for tok_id, data in tok.vocab.items():
        if tok_id == tok.eot_id or len(data) < 2:
            continue
        pieces = PRETOKEN_PATTERN.findall(data.decode("utf-8", errors="replace"))
        # A token may be a partial UTF-8 sequence; only check clean text.
        if "�" not in data.decode("utf-8", errors="replace"):
            assert len(pieces) == 1, f"token {data!r} spans several pre-tokens"


def test_training_is_deterministic():
    a = BPETokenizer.train(TRAIN_TEXT, vocab_size=350)
    b = BPETokenizer.train(TRAIN_TEXT, vocab_size=350)
    assert a.merges == b.merges


def test_decode_handles_split_utf8_bytes(tok):
    # Ids 0xC3 (195) alone is half of "é". Must not raise.
    assert tok.decode([195]) == "�"


# --- save / load ---------------------------------------------------------------------

def test_save_and_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    text = "Once upon a time, Lily had a red ball. é🙂<|endoftext|>"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_vocab_size_too_small_raises():
    with pytest.raises(ValueError):
        BPETokenizer.train("hello", vocab_size=100)
