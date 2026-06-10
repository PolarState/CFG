"""Tests for CFGCharacterTokenizer.__init__.

Block 1 of the cfg.grammar.* test coverage plan: pin down the vocab + id
layout assumptions that the cfg3b dataset pipeline depends on.
"""

import pytest

from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer


# ── Default id layout (the cfg3b invariant) ───────────────────────────────


def test_default_vocab_id_layout():
    """Terminals get ids 0..V-1, then eos at V, then bos at V+1.

    This is the layout the cfg3b downstream tools (SA queries, analysis,
    visualizations) all assume.
    """
    tok = CFGCharacterTokenizer(vocab=["1", "2", "3"])
    # Terminals first, in vocab order.
    assert tok.encode_vocab["1"] == 0
    assert tok.encode_vocab["2"] == 1
    assert tok.encode_vocab["3"] == 2
    # Then eos (default "E"), then bos (default "B").
    assert tok.encode_vocab["E"] == 3
    assert tok.encode_vocab["B"] == 4
    # Decode map is the inverse.
    assert tok.decode_vocab[0] == "1"
    assert tok.decode_vocab[4] == "B"


# ── Custom bos/eos chars ──────────────────────────────────────────────────


def test_custom_bos_eos_chars():
    """Caller-supplied bos_char and eos_char override the defaults."""
    tok = CFGCharacterTokenizer(vocab=["a", "b"], bos_char="<", eos_char=">")
    assert tok.encode_vocab["a"] == 0
    assert tok.encode_vocab["b"] == 1
    assert tok.encode_vocab[">"] == 2  # eos first
    assert tok.encode_vocab["<"] == 3  # bos second
    # The string and token attrs reflect the overrides too.
    assert tok.bos_string == "<"
    assert tok.eos_string == ">"


# ── Vocab collision with special tokens ───────────────────────────────────


def test_vocab_collision_with_default_bos_raises():
    """Including the bos char in the vocab is rejected at construction."""
    with pytest.raises(ValueError, match="bos_char"):
        CFGCharacterTokenizer(vocab=["1", "2", "B"])


def test_vocab_collision_with_default_eos_raises():
    """Including the eos char in the vocab is rejected at construction."""
    with pytest.raises(ValueError, match="eos_char"):
        CFGCharacterTokenizer(vocab=["1", "2", "E"])


def test_vocab_collision_with_custom_bos_raises():
    """The collision check uses the *configured* bos_char, not the default."""
    with pytest.raises(ValueError, match="bos_char"):
        CFGCharacterTokenizer(vocab=["1", "<"], bos_char="<")


# ── Multi-character special tokens ────────────────────────────────────────


def test_multi_char_bos_raises():
    """bos_char must be a single character (the tokenizer is char-level)."""
    with pytest.raises(ValueError):
        CFGCharacterTokenizer(vocab=["1", "2"], bos_char="BOS")


def test_multi_char_eos_raises():
    """eos_char must be a single character."""
    with pytest.raises(ValueError):
        CFGCharacterTokenizer(vocab=["1", "2"], eos_char="EOS")


# ── Special-token attribute consistency ───────────────────────────────────


def test_special_token_attrs_consistent():
    """bos_string / bos_token (and eos counterparts) agree with the maps.

    bos_string is the character; bos_token is the result of encode() on
    that character, i.e. a single-element list of the assigned id.
    """
    tok = CFGCharacterTokenizer(vocab=["1", "2", "3"])

    assert tok.bos_string == "B"
    assert tok.eos_string == "E"

    # bos_token / eos_token are the encoded lists; verify they match a
    # fresh encode() call and resolve to the id stored in encode_vocab.
    assert tok.bos_token == [tok.encode_vocab["B"]]
    assert tok.eos_token == [tok.encode_vocab["E"]]
    assert tok.bos_token == tok.encode("B")
    assert tok.eos_token == tok.encode("E")
