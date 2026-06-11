"""Tests for cfg.analysis.compare — per-prefix KL between n-gram sources."""

import numpy as np

from cfg.analysis.compare import conditional_table, kl_bits, kl_rows
from cfg.analysis.exact import exact_bos_ngrams
from cfg.grammar.cfg_grammar import CFGrammar


def toy_arrays(scale=1.0):
    """A hand-built 2-level source over V=2 with known conditionals.

    Stream model: P(first=0)=0.75, P(first=1)=0.25;
    P(next=0 | first=0)=2/3, P(next=1 | first=0)=1/3;
    P(next | first=1) = [0.5, 0.5].
    ``scale`` turns probabilities into pseudo-counts — KL must be
    invariant to it because conditionals normalize the scale away.
    """
    n1 = np.array([0.75, 0.25]) * scale
    n2 = np.array([[0.5, 0.25], [0.125, 0.125]]) * scale
    return {"n1": n1, "n2": n2}


def test_conditional_table_rows():
    cond, mass = conditional_table(toy_arrays(), 2)
    np.testing.assert_allclose(cond[0], [2 / 3, 1 / 3])
    np.testing.assert_allclose(cond[1], [0.5, 0.5])
    np.testing.assert_allclose(mass, [0.75, 0.25])


def test_kl_self_is_zero():
    rows, summary = kl_rows(toy_arrays(), toy_arrays())
    assert all(r["kl_bits"] == 0.0 for r in rows)
    assert all(s["expected_kl_bits"] == 0.0 for s in summary.values())


def test_kl_invariant_to_count_scale():
    # The same distribution expressed as probabilities and as counts
    # (scaled by 1e6) must compare identically against a third source.
    q = {"n1": np.array([0.5, 0.5]), "n2": np.full((2, 2), 0.25)}
    _, s_prob = kl_rows(toy_arrays(scale=1.0), q)
    _, s_count = kl_rows(toy_arrays(scale=1e6), q)
    for k in s_prob:
        np.testing.assert_allclose(
            s_prob[k]["expected_kl_bits"], s_count[k]["expected_kl_bits"]
        )


def test_kl_hand_computed():
    # P = [0.75, 0.25] vs Q = [0.5, 0.5] at level 1:
    # KL = 0.75*log2(1.5) + 0.25*log2(0.5) = 0.18872 bits.
    q = {"n1": np.array([0.5, 0.5]), "n2": np.full((2, 2), 0.25)}
    rows, summary = kl_rows(toy_arrays(), q, max_n=1)
    expected = 0.75 * np.log2(1.5) + 0.25 * np.log2(0.5)
    np.testing.assert_allclose(rows[0]["kl_bits"], expected, atol=1e-12)
    np.testing.assert_allclose(
        summary[1]["expected_kl_bits"], expected, atol=1e-12
    )


def test_q_zero_on_p_support_is_inf_and_epsilon_floors_it():
    p = {"n1": np.array([0.5, 0.5])}
    q = {"n1": np.array([1.0, 0.0])}
    rows, summary = kl_rows(p, q)
    assert np.isinf(rows[0]["kl_bits"])
    assert summary[1]["n_inf"] == 1
    assert np.isinf(summary[1]["expected_kl_bits"])

    rows_eps, summary_eps = kl_rows(p, q, epsilon=1e-9)
    assert np.isfinite(rows_eps[0]["kl_bits"])
    assert summary_eps[1]["n_inf"] == 0


def test_exact_vs_masked_exact_diverges_on_affected_prefixes_only():
    # The unmasked and mask-7:0 exact distributions should agree on
    # most prefixes and diverge sharply where NT 7's first rule fed
    # probability mass — concentrated, not uniform, divergence.
    grammar = CFGrammar.from_name("cfg3b")
    full = exact_bos_ngrams(grammar, max_n=4)
    weights = {
        nt: [0.0 if (nt == "7" and i == 0) else 1.0
             for i in range(len(prods))]
        for nt, prods in grammar.rules.items()
    }
    masked = exact_bos_ngrams(grammar, max_n=4, weights=weights)

    rows, summary = kl_rows(full, masked, epsilon=1e-12)
    # Divergence must exist (the mask changes the distribution)...
    assert summary[4]["finite_expected_kl_bits"] > 0.01
    # ...but be concentrated: some prefixes essentially unaffected.
    finite = [r["kl_bits"] for r in rows if r["level"] == 4]
    assert min(finite) < 1e-3
