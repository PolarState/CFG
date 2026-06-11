"""Tests for cfg.analysis.model_probs against a hand-built toy model.

Runs only where torch is installed (the gpt2 env); skipped in the
torch-free cfg env via importorskip. The toy model is a fixed-table
"bigram" LM whose next-token distribution depends only on the last
input token, so every level of the extracted joint is computable in
closed form with the chain rule.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cfg.analysis.model_probs import extract_model_bos_ngrams

V = 5
BOS = 4


class BigramTableModel(torch.nn.Module):
    """Causal-LM stand-in: logits at each position depend only on the
    token at that position, read from a fixed (V, V) table."""

    def __init__(self, table):
        super().__init__()
        # Minimal config surface model_probs touches.
        self.config = type("Cfg", (), {"vocab_size": V})()
        self.register_buffer("table", torch.tensor(table, dtype=torch.float32))
        # A dummy parameter so next(model.parameters()) yields a device.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        # attention_mask is accepted (decision_points passes one for
        # padded batches) but irrelevant here: each position's logits
        # depend only on its own token.
        logits = self.table[input_ids]
        return type("Out", (), {"logits": logits})()


def make_model_and_probs(seed=0):
    """Build a random logit table and the row-softmax probabilities it
    implies, so tests can chain expectations analytically."""
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(V, V)).astype(np.float64)
    # Row-wise softmax in float64 — the same arithmetic the extractor
    # performs, so comparisons are exact up to float64 rounding.
    exp = np.exp(table - table.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return BigramTableModel(table), probs


def test_levels_sum_to_one_and_nest():
    model, _ = make_model_and_probs()
    arrays = extract_model_bos_ngrams(model, max_n=3, vocab_size=V, bos_id=BOS)
    for k in range(1, 4):
        assert abs(arrays[f"n{k}"].sum() - 1.0) < 1e-9
    for k in range(2, 4):
        np.testing.assert_allclose(
            arrays[f"n{k}"].sum(axis=-1), arrays[f"n{k - 1}"], atol=1e-12
        )
    assert arrays["prefix"].tolist() == [BOS]


def test_joints_match_chain_rule():
    model, probs = make_model_and_probs()
    arrays = extract_model_bos_ngrams(model, max_n=3, vocab_size=V, bos_id=BOS)

    # For a bigram model the conditional after [bos, a, b] is row b of
    # the table (last token only), so the joint factorizes as
    # P(a|bos) P(b|a) P(c|b).
    expected_n1 = probs[BOS]
    np.testing.assert_allclose(arrays["n1"], expected_n1, atol=1e-9)

    expected_n2 = expected_n1[:, None] * probs
    np.testing.assert_allclose(arrays["n2"], expected_n2, atol=1e-9)

    expected_n3 = expected_n2[:, :, None] * probs[None, :, :]
    np.testing.assert_allclose(arrays["n3"], expected_n3, atol=1e-9)


def test_batch_size_does_not_change_result():
    model, _ = make_model_and_probs(seed=1)
    a = extract_model_bos_ngrams(model, max_n=3, vocab_size=V, bos_id=BOS,
                                 batch_size=1, verbose=False)
    b = extract_model_bos_ngrams(model, max_n=3, vocab_size=V, bos_id=BOS,
                                 batch_size=64, verbose=False)
    for k in range(1, 4):
        np.testing.assert_allclose(a[f"n{k}"], b[f"n{k}"], atol=1e-12)


def test_vocab_mismatch_raises():
    model, _ = make_model_and_probs()
    with pytest.raises(ValueError, match="vocab_size"):
        extract_model_bos_ngrams(model, max_n=2, vocab_size=7, bos_id=BOS)
