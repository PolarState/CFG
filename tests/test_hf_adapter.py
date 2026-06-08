# Tests for the HuggingFace tokenizer adapter.

import torch

from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer
from cfg.grammar.hf_adapter import HFTokenizerAdapter


def _make_adapter():
    """Helper: build an adapter around a simple character tokenizer."""
    char_tok = CFGCharacterTokenizer(vocab=["0", "1", "2"])
    return HFTokenizerAdapter(char_tok)


def test_special_tokens_are_strings():
    """HF expects bos_token / eos_token / pad_token to be strings."""
    tok = _make_adapter()
    assert isinstance(tok.bos_token, str)
    assert isinstance(tok.eos_token, str)
    assert isinstance(tok.pad_token, str)


def test_vocab_size():
    """vocab_size includes the data tokens plus bos and eos."""
    tok = _make_adapter()
    # 3 data tokens + eos + bos = 5
    assert tok.vocab_size == 5


def test_encode_decode_roundtrip():
    """Encoding then decoding should recover the original string."""
    tok = _make_adapter()
    text = "012"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded == text


def test_call_returns_input_ids():
    """__call__ with return_tensors='pt' should return a dict with input_ids."""
    tok = _make_adapter()
    output = tok("012", return_tensors="pt")
    assert "input_ids" in output
    assert isinstance(output["input_ids"], torch.Tensor)


def test_batch_decode():
    """batch_decode should handle a batch of id sequences."""
    tok = _make_adapter()
    ids = tok("012", return_tensors="pt")["input_ids"]
    # batch_decode expects a list/tensor of sequences.
    decoded = tok.batch_decode(ids, skip_special_tokens=True)
    assert isinstance(decoded, list)
    assert len(decoded) == 1
    assert "012" in decoded[0]


def test_call_with_bos():
    """Encoding the bos token string should produce the bos_token_id."""
    tok = _make_adapter()
    ids = tok.encode(tok.bos_token, add_special_tokens=False)
    assert tok.bos_token_id in ids


def test_unknown_token_fallback():
    """Unknown characters should map to the eos id rather than crashing."""
    tok = _make_adapter()
    unk_id = tok._convert_token_to_id("Z")
    assert unk_id == tok.eos_token_id
