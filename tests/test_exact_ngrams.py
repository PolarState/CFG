"""Tests for cfg.analysis.exact — exact BOS-rooted n-gram probabilities.

Strategy: the exact computation must agree with (a) structural
invariants that hold for any probability table, and (b) Monte Carlo
estimates from the actual sampling path (CFGrammar.generate + the
bos/eos stream layout of build_char_dataset), within sampling error.
The MC comparison uses a deliberately tiny grammar with 1-3 token
yields so that the eos -> bos -> next-sentence continuation path is
exercised inside the n-gram window — cfg3b's sentences are far longer
than any window we test, so it alone would never cover that branch.
"""

import random

import numpy as np

from cfg.analysis.exact import exact_bos_ngrams, symbol_prefix_distribution
from cfg.grammar.cfg_grammar import CFGrammar
from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer


# A tiny acyclic grammar with yields of length 1-3. Start symbol is "S"
# (the only NT that never appears on a RHS). Short yields force the
# stream model to cross eos/bos boundaries within small windows.
TINY = {
    "S": [["1"], ["2", "3"], ["A", "1"]],
    "A": [["2"], ["3", "3"]],
}


def empirical_bos_ngrams(grammar, max_n, num_examples, seed, weights=None):
    """Monte Carlo reference: sample examples through the real generator,
    lay them out exactly like build_char_dataset (bos + chars + eos,
    concatenated), and count bos-rooted k-grams with a sliding window.

    Returns float64 arrays normalized per level — directly comparable to
    exact_bos_ngrams output.
    """
    tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    bos_id = tokenizer.encode_vocab[tokenizer.bos_string]
    eos_id = tokenizer.encode_vocab[tokenizer.eos_string]
    vocab_size = len(tokenizer)

    # Build the flat id stream the way the dataset builder does.
    random.seed(seed)
    stream = []
    for _ in range(num_examples):
        text = grammar.generate(weights=weights)
        stream.append(bos_id)
        stream.extend(tokenizer.encode_vocab[c] for c in text)
        stream.append(eos_id)

    # Count suffixes of each length after every bos occurrence. Skip bos
    # positions whose window would run off the end of the stream so that
    # every counted bos contributes to every level equally.
    counts = {
        f"n{k}": np.zeros((vocab_size,) * k, dtype=np.float64)
        for k in range(1, max_n + 1)
    }
    n_bos = 0
    arr = np.array(stream)
    for pos in np.nonzero(arr == bos_id)[0]:
        if pos + max_n >= len(arr):
            continue
        n_bos += 1
        window = tuple(int(t) for t in arr[pos + 1 : pos + 1 + max_n])
        for k in range(1, max_n + 1):
            counts[f"n{k}"][window[:k]] += 1

    # Normalize counts into probabilities so levels are comparable.
    for k in range(1, max_n + 1):
        counts[f"n{k}"] /= n_bos
    return counts


# ── Structural invariants ──────────────────────────────────────────────


def test_levels_sum_to_one_and_nest():
    grammar = CFGrammar.from_name("cfg3b")
    arrays = exact_bos_ngrams(grammar, max_n=4)
    for k in range(1, 5):
        # Every level is a full probability distribution over suffixes.
        assert abs(arrays[f"n{k}"].sum() - 1.0) < 1e-12
    for k in range(2, 5):
        # Marginalizing out the last token recovers the previous level —
        # the same nesting invariant the empirical count arrays satisfy.
        np.testing.assert_allclose(
            arrays[f"n{k}"].sum(axis=-1), arrays[f"n{k - 1}"], atol=1e-12
        )


def test_prefix_key_is_bos_id():
    grammar = CFGrammar.from_name("cfg3b")
    tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    arrays = exact_bos_ngrams(grammar, max_n=2)
    assert arrays["prefix"].tolist() == [
        tokenizer.encode_vocab[tokenizer.bos_string]
    ]


def test_cfg3b_no_specials_within_window():
    # cfg3b's shortest sentence is far longer than 6 tokens, so within a
    # 6-token window after bos, no eos/bos mass should ever appear: all
    # probability lives on pure-terminal suffixes (ids 0..2).
    grammar = CFGrammar.from_name("cfg3b")
    arrays = exact_bos_ngrams(grammar, max_n=6)
    terminal_block = arrays["n6"][(slice(0, 3),) * 6]
    assert abs(terminal_block.sum() - 1.0) < 1e-12


def test_symbol_prefix_distribution_is_normalized():
    grammar = CFGrammar(TINY)
    for symbol in ["S", "A", "1"]:
        dist = symbol_prefix_distribution(grammar, symbol, max_n=3)
        assert abs(sum(dist.values()) - 1.0) < 1e-12


# ── Monte Carlo cross-validation ───────────────────────────────────────


def test_tiny_grammar_matches_monte_carlo():
    # max_n=5 with 1-3 token yields guarantees windows cross at least
    # one eos/bos boundary, exercising the stream continuation path.
    grammar = CFGrammar(TINY)
    exact = exact_bos_ngrams(grammar, max_n=5)
    mc = empirical_bos_ngrams(grammar, max_n=5, num_examples=200_000, seed=0)
    for k in range(1, 6):
        # 200k samples put per-cell standard error well under 2e-3 at
        # every level; 5e-3 absolute tolerance gives ~3 sigma headroom.
        np.testing.assert_allclose(exact[f"n{k}"], mc[f"n{k}"], atol=5e-3)


def test_masked_rule_matches_monte_carlo_and_zeroes_mass():
    grammar = CFGrammar(TINY)
    # Mask S's first rule (yield "1"), mirroring build_mask_weights: the
    # masked rule gets weight 0, everything else weight 1.
    weights = {"S": [0.0, 1.0, 1.0], "A": [1.0, 1.0]}
    exact = exact_bos_ngrams(grammar, max_n=3, weights=weights)
    mc = empirical_bos_ngrams(
        grammar, max_n=3, num_examples=200_000, seed=1, weights=weights
    )
    for k in range(1, 4):
        np.testing.assert_allclose(exact[f"n{k}"], mc[f"n{k}"], atol=5e-3)

    # With S -> "1" masked, a sentence can no longer be the single token
    # '1', so P(first token = '1', second token = eos) must be zero.
    tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    one_id = tokenizer.encode_vocab["1"]
    eos_id = tokenizer.encode_vocab[tokenizer.eos_string]
    assert exact["n2"][one_id, eos_id] == 0.0


def test_cfg3b_unigram_matches_monte_carlo():
    # Sanity-check the real grammar too, at the cheapest level: the
    # distribution of the first token after bos. 20k samples keeps the
    # test fast (cfg3b sentences are ~200 tokens each).
    grammar = CFGrammar.from_name("cfg3b")
    exact = exact_bos_ngrams(grammar, max_n=1)
    mc = empirical_bos_ngrams(grammar, max_n=1, num_examples=20_000, seed=2)
    np.testing.assert_allclose(exact["n1"], mc["n1"], atol=1e-2)
