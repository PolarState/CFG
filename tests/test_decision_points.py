"""Tests for cfg.analysis.decision_points."""

import numpy as np
import pytest

from cfg.analysis.decision_points import (
    linearize_trace,
    summarize_decision_points,
    terminal_rule_yields,
)
from cfg.grammar.cfg_grammar import CFGrammar


# Hand-built Schema B tree over a toy grammar:
#   S -> A A      (rule 0 chosen)
#   A -> "1" "2"  (rule 0) | "3" (rule 1)
TREE = {
    "nt": "S", "rule": 0, "children": [
        {"nt": "A", "rule": 0, "children": [{"t": "1"}, {"t": "2"}]},
        {"nt": "A", "rule": 1, "children": [{"t": "3"}]},
    ],
}
TOY = {"S": [["A", "A"]], "A": [["1", "2"], ["3"]]}


def test_linearize_terminals_in_order():
    terminals, points = linearize_trace(TREE, "A")
    assert terminals == ["1", "2", "3"]
    # First A starts at offset 0 with rule 0; second at offset 2, rule 1.
    assert points == [(0, 0), (1, 2)]


def test_linearize_other_nt():
    _, points = linearize_trace(TREE, "S")
    assert points == [(0, 0)]


def test_terminal_rule_yields():
    grammar = CFGrammar(TOY)
    assert terminal_rule_yields(grammar, "A") == [["1", "2"], ["3"]]
    # S's rule contains nonterminals — must refuse, not silently mangle.
    with pytest.raises(ValueError, match="nonterminal"):
        terminal_rule_yields(grammar, "S")


def test_cfg3b_masked_nts_are_terminal_level():
    # The masking experiments target NTs 7/8/9; the tool's scope must
    # cover all of them.
    grammar = CFGrammar.from_name("cfg3b")
    for nt in ["7", "8", "9"]:
        yields = terminal_rule_yields(grammar, nt)
        assert len(yields) == 2


def test_summarize_groups_and_shares():
    rows = [
        {"sentence": 0, "offset": 0, "true_rule": 0, "p_rules": [0.6, 0.2]},
        {"sentence": 0, "offset": 5, "true_rule": 0, "p_rules": [0.4, 0.4]},
        {"sentence": 1, "offset": 2, "true_rule": 1, "p_rules": [0.1, 0.3]},
    ]
    s = summarize_decision_points(rows, n_rules=2)
    assert s[0]["n"] == 2 and s[1]["n"] == 1 and s["all"]["n"] == 3
    np.testing.assert_allclose(s[0]["mean_p"], [0.5, 0.3])
    # Shares normalize within each row before averaging.
    np.testing.assert_allclose(s[0]["mean_share"], [(0.75 + 0.5) / 2,
                                                    (0.25 + 0.5) / 2])
    np.testing.assert_allclose(s[1]["mean_share"], [0.25, 0.75])


def test_measure_against_bigram_toy_model():
    # Reuse the fixed-table bigram model: next-token dist depends only
    # on the last token, so every yield probability is computable in
    # closed form from the table.
    torch = pytest.importorskip("torch")
    from tests.test_model_probs import BigramTableModel, make_model_and_probs

    from cfg.analysis.decision_points import measure_decision_points
    from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer

    grammar = CFGrammar(TOY)
    tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    model, probs = make_model_and_probs(seed=2)

    rows = measure_decision_points(
        model, grammar, [TREE], "A", max_sentences=1,
        tokenizer=tokenizer, verbose=False,
    )
    assert len(rows) == 2

    ids = {c: tokenizer.encode_vocab[c] for c in "123"}
    bos = tokenizer.encode_vocab[tokenizer.bos_string]

    # Point 1: context = [bos]; last token bos.
    #   rule 0 yield "12": P(1|bos) * P(2|1)
    #   rule 1 yield "3":  P(3|bos)
    expect0 = [
        probs[bos, ids["1"]] * probs[ids["1"], ids["2"]],
        probs[bos, ids["3"]],
    ]
    np.testing.assert_allclose(rows[0]["p_rules"], expect0, rtol=1e-6)

    # Point 2: context = [bos, 1, 2]; last token "2".
    expect1 = [
        probs[ids["2"], ids["1"]] * probs[ids["1"], ids["2"]],
        probs[ids["2"], ids["3"]],
    ]
    np.testing.assert_allclose(rows[1]["p_rules"], expect1, rtol=1e-6)
