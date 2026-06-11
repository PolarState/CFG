"""Compare two BOS-rooted n-gram distributions via per-prefix KL.

Works on any pair of npz array dicts in the shared schema — empirical
counts (compute_ngrams.py), exact grammar probabilities
(compute_exact_ngrams.py), or model probabilities
(extract_model_ngrams.py) — because conditionals derive the same way
from both: row-normalizing n{k} by n{k-1}. Counts and probabilities are
deliberately interchangeable; normalization cancels the scale.

Conventions:
  - KL is reported in bits (log base 2).
  - KL(P || Q) is computed per prefix over the next-token distribution,
    restricted to prefixes where P has support.
  - A prefix where Q assigns 0 to some next-token that P supports gets
    KL = inf. No silent smoothing — infs are real findings (e.g. a
    masked-trained model assigning literal zero is impossible after a
    softmax, but an empirical Q can have true zeros). Callers can pass
    ``epsilon`` to floor Q if they want finite numbers.
  - Per-level summary = sum over prefixes of P(prefix) * KL(prefix),
    i.e. the expected next-token KL under P's prefix distribution —
    equivalently the difference in per-token cross-entropy at depth k.
"""

import numpy as np


def conditional_table(arrays, k):
    """Derive P(next | prefix) rows from nested level arrays.

    Returns ``(cond, prefix_mass)`` where ``cond`` has shape
    (V^(k-1), V) — one row per prefix of length k-1, in C order — and
    ``prefix_mass`` has shape (V^(k-1),) holding each prefix's
    count/probability mass (unnormalized; useful for weighting).
    Rows whose prefix has zero mass are left as all-zero rather than
    NaN so support masks stay clean.
    """
    n_full = np.asarray(arrays[f"n{k}"], dtype=np.float64)
    V = n_full.shape[0]

    # The prefix mass for level k is level k-1; for k=1 the "empty
    # prefix" mass is the total of level 1 (count of bos occurrences in
    # the empirical case, 1.0 in the probability case).
    if k == 1:
        prefix_mass = np.array([n_full.sum()], dtype=np.float64)
    else:
        prefix_mass = np.asarray(arrays[f"n{k - 1}"], dtype=np.float64).ravel()

    flat_full = n_full.reshape(-1, V)
    cond = np.zeros_like(flat_full)
    has_mass = prefix_mass > 0
    cond[has_mass] = flat_full[has_mass] / prefix_mass[has_mass, None]
    return cond, prefix_mass


def kl_bits(p_row, q_row):
    """KL(P || Q) in bits for one next-token distribution pair.

    Restricted to P's support; returns inf if Q is 0 anywhere P is
    positive.
    """
    support = p_row > 0
    p = p_row[support]
    q = q_row[support]
    if np.any(q == 0):
        return np.inf
    return float(np.sum(p * np.log2(p / q)))


def kl_rows(p_arrays, q_arrays, max_n=None, epsilon=0.0):
    """Per-prefix KL(P || Q) for every level both sources cover.

    Args:
        p_arrays / q_arrays: npz-style dicts (n1..n{max}, optional
            prefix key). P is the reference: rows are restricted to
            prefixes with P-mass, and the per-level expectation weights
            by P's prefix distribution.
        max_n: deepest level to compare; defaults to the deepest level
            present in BOTH inputs.
        epsilon: optional floor added to Q's conditionals (then
            renormalized) to make infs finite. 0 = no smoothing.

    Returns:
        ``(rows, summary)`` where ``rows`` is a list of dicts
        (level, prefix: tuple of ids, p_mass: P's normalized prefix
        probability, kl_bits) and ``summary`` maps level -> dict with
        expected_kl_bits, max_kl_bits, n_prefixes, n_inf.
    """

    def levels_in(arrays):
        return sorted(
            int(k[1:]) for k in arrays.keys()
            if k.startswith("n") and k[1:].isdigit()
        )

    common = sorted(set(levels_in(p_arrays)) & set(levels_in(q_arrays)))
    if max_n is not None:
        common = [k for k in common if k <= max_n]
    if not common:
        raise ValueError("no common n{k} levels between the two inputs")

    rows = []
    summary = {}
    for k in common:
        p_cond, p_mass = conditional_table(p_arrays, k)
        q_cond, _ = conditional_table(q_arrays, k)
        V = p_cond.shape[1]

        # Optional smoothing of Q only: floor every cell then
        # renormalize each row, so Q stays a distribution.
        if epsilon > 0:
            q_cond = q_cond + epsilon
            q_cond /= q_cond.sum(axis=1, keepdims=True)

        # Normalize P's prefix masses into a probability distribution
        # over prefixes so per-level expectations are comparable across
        # count-based and probability-based references.
        total_mass = p_mass.sum()
        p_prefix_prob = p_mass / total_mass if total_mass > 0 else p_mass

        expected = 0.0
        max_kl = 0.0
        n_inf = 0
        n_prefixes = 0
        for flat_idx in np.nonzero(p_mass > 0)[0]:
            kl = kl_bits(p_cond[flat_idx], q_cond[flat_idx])
            prefix = (
                tuple(int(i) for i in np.unravel_index(flat_idx, (V,) * (k - 1)))
                if k > 1 else ()
            )
            rows.append({
                "level": k,
                "prefix": prefix,
                "p_mass": float(p_prefix_prob[flat_idx]),
                "kl_bits": kl,
            })
            n_prefixes += 1
            if np.isinf(kl):
                n_inf += 1
            else:
                expected += p_prefix_prob[flat_idx] * kl
                max_kl = max(max_kl, kl)

        summary[k] = {
            # Expected KL over finite rows; if any row is inf the true
            # expectation is inf too — n_inf flags that case explicitly.
            "expected_kl_bits": expected if n_inf == 0 else np.inf,
            "finite_expected_kl_bits": expected,
            "max_kl_bits": max_kl,
            "n_prefixes": n_prefixes,
            "n_inf": n_inf,
        }

    return rows, summary
