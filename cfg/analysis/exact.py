"""Exact BOS-rooted n-gram probabilities under a (weighted) CFG.

While ``cfg.analysis.ngrams`` measures the *empirical* distribution of a
sampled corpus via a suffix-array index, this module computes the *true*
distribution implied by the grammar and its sampling weights — no corpus
involved. The gap between the two is pure finite-sample noise, and the
gap between either and a trained model's distribution is what the
project actually studies.

The token stream modeled here matches build_char_dataset.py exactly:
each example is laid out as ``bos + terminals + eos`` and examples are
concatenated back to back, so the stream after any bos token looks like

    <sentence yield> eos bos <sentence yield> eos bos ...

``exact_bos_ngrams(grammar, max_n)`` returns the same dict-of-dense-
arrays schema as ``enumerate_ngrams`` (keys ``n1``..``n{max_n}`` of
shape ``(V,)*k`` plus a ``prefix`` key), except the arrays hold float64
*probabilities* P(next k tokens after a bos = suffix) instead of int64
counts. Conditionals derive identically in both cases — by dividing
``n{k}`` rows by ``n{k-1}`` — so downstream tooling can treat counts
and probabilities uniformly.

Sampling semantics mirror CFGrammar.generate():
  - the start symbol is chosen uniformly among grammar.start_symbols;
  - within a nonterminal, rule i is chosen with probability
    weights[nt][i] / sum(weights[nt]) (uniform when no weights given),
    which is exactly what random.choice / random.choices do.

Only acyclic grammars are supported (same restriction as
count_generations); a recursive grammar raises ValueError.
"""

import numpy as np

from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer


def symbol_prefix_distribution(grammar, symbol, max_n, weights=None, _memo=None):
    """Distribution over the first ``max_n`` terminals of ``symbol``'s yield.

    Returns a dict mapping ``(prefix_tuple, is_complete) -> probability``
    where ``prefix_tuple`` is a tuple of terminal *strings* (characters).

      - ``is_complete=True``: the tuple is the symbol's ENTIRE yield
        (its length is <= max_n and nothing follows it).
      - ``is_complete=False``: the yield was truncated — the tuple has
        length exactly max_n and the true yield continues past it.

    Truncating at max_n keeps the state space bounded (at most
    |terminals|^max_n truncated tuples per symbol) while remaining exact
    for every question about the first max_n tokens.
    """
    # Memo table shared across the recursion. Keyed by symbol only —
    # max_n and weights are constant within one call tree.
    if _memo is None:
        _memo = {}
    if symbol in _memo:
        return _memo[symbol]

    # Base case: a terminal yields exactly itself, always complete.
    if symbol not in grammar.rules:
        dist = {((symbol,), True): 1.0}
        _memo[symbol] = dist
        return dist

    productions = grammar.rules[symbol]

    # Normalize the rule weights into selection probabilities, matching
    # random.choices semantics (raw weights, normalized by their sum).
    # No weights for this NT means uniform over its rules, matching
    # random.choice in CFGrammar._expand.
    if weights is not None and symbol in weights:
        raw = list(weights[symbol])
    else:
        raw = [1.0] * len(productions)
    total_w = sum(raw)
    if total_w <= 0:
        raise ValueError(
            f"NT {symbol!r}: rule weights sum to {total_w}; at least one "
            f"rule must have positive weight"
        )
    rule_probs = [w / total_w for w in raw]

    dist = {}
    for rule_prob, production in zip(rule_probs, productions):
        # Rules with zero probability (e.g. masked rules) contribute
        # nothing; skip them so their subtrees aren't even computed.
        if rule_prob == 0.0:
            continue

        # Compose the children of this production left to right. The
        # accumulator maps (sequence_so_far, still_complete) -> prob.
        acc = {((), True): rule_prob}
        for child in production:
            child_dist = symbol_prefix_distribution(
                grammar, child, max_n, weights, _memo
            )
            new_acc = {}
            for (seq, complete), p in acc.items():
                # Once a branch is truncated (incomplete), it already
                # holds max_n tokens — later children can't affect the
                # first max_n tokens, so carry it through unchanged.
                if not complete:
                    new_acc[(seq, False)] = new_acc.get((seq, False), 0.0) + p
                    continue
                # Otherwise extend with every possible child prefix,
                # truncating the concatenation back down to max_n.
                for (cseq, ccomplete), cp in child_dist.items():
                    joined = seq + cseq
                    truncated = len(joined) > max_n
                    new_seq = joined[:max_n] if truncated else joined
                    # The combined branch stays complete only if the
                    # child's yield was complete AND nothing got cut.
                    new_complete = ccomplete and not truncated
                    key = (new_seq, new_complete)
                    new_acc[key] = new_acc.get(key, 0.0) + p * cp
            acc = new_acc

        # Fold this rule's branches into the symbol's distribution.
        for key, p in acc.items():
            dist[key] = dist.get(key, 0.0) + p

    _memo[symbol] = dist
    return dist


def _after_bos_distribution(grammar, remaining, weights, symbol_memo, bos_char,
                            eos_char, _memo=None):
    """Distribution over the next ``remaining`` stream tokens after a bos.

    The stream after a bos is ``yield(S) + eos + bos + yield(S) + ...``
    (start symbol S drawn uniformly per sentence), so every returned
    tuple has length exactly ``remaining`` — short sentences continue
    through their eos into the next sentence's bos and beyond.

    Tokens here are terminal characters plus the bos/eos characters;
    conversion to ids happens in exact_bos_ngrams.
    """
    # Memoized on `remaining` only — the recursion re-enters with
    # strictly smaller values (each sentence consumes >= 3 tokens:
    # one-terminal minimum yield + eos + bos).
    if _memo is None:
        _memo = {}
    if remaining in _memo:
        return _memo[remaining]

    dist = {}
    n_starts = len(grammar.start_symbols)
    for start in grammar.start_symbols:
        # Start symbols are equiprobable, matching random.choice in
        # CFGrammar.generate when no symbol is passed.
        start_p = 1.0 / n_starts
        sym_dist = symbol_prefix_distribution(
            grammar, start, remaining, weights, symbol_memo
        )
        for (seq, complete), p in sym_dist.items():
            p *= start_p
            if len(seq) >= remaining:
                # The sentence yield alone fills the window. (Truncated
                # branches always land here: incomplete means
                # len(seq) == remaining by construction.)
                key = seq[:remaining]
                dist[key] = dist.get(key, 0.0) + p
                continue
            # The full yield fits with room to spare, so the stream
            # continues: eos, then bos, then the next sentence.
            out = seq + (eos_char,)
            if len(out) >= remaining:
                dist[out[:remaining]] = dist.get(out[:remaining], 0.0) + p
                continue
            out = out + (bos_char,)
            if len(out) >= remaining:
                dist[out[:remaining]] = dist.get(out[:remaining], 0.0) + p
                continue
            # Recurse for whatever the next sentence contributes to the
            # tail of the window.
            tail_dist = _after_bos_distribution(
                grammar, remaining - len(out), weights, symbol_memo,
                bos_char, eos_char, _memo,
            )
            for tail, q in tail_dist.items():
                key = out + tail
                dist[key] = dist.get(key, 0.0) + p * q

    _memo[remaining] = dist
    return dist


def exact_bos_ngrams(grammar, max_n, weights=None, tokenizer=None):
    """Exact P(next k tokens | bos) for k = 1..max_n, as dense arrays.

    Returns the enumerate_ngrams npz schema — ``{"n1": arr, ...,
    "n{max_n}": arr, "prefix": [bos_id]}`` — with float64 probability
    arrays of shape ``(V,)*k``. Each ``n{k}`` sums to 1, and
    ``n{k}.sum(axis=-1) == n{k-1}`` exactly (marginalization), mirroring
    how count arrays nest in the empirical case.

    ``weights`` uses the same format as CFGrammar.generate (dict of
    per-NT rule weight lists, e.g. from build_mask_weights). ``tokenizer``
    defaults to ``CFGCharacterTokenizer(vocab=grammar.terminal_symbols)``,
    the same construction the dataset builders use, so token ids line up
    with the corpora on disk.
    """
    # Refuse recursive grammars up front — the truncated-prefix DP below
    # assumes the per-symbol recursion bottoms out.
    if grammar.count_generations() is None:
        raise ValueError("exact_bos_ngrams requires an acyclic grammar")

    if tokenizer is None:
        tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    vocab_size = len(tokenizer)
    bos_id = tokenizer.encode_vocab[tokenizer.bos_string]

    # Compute the full-depth distribution once; shallower levels come
    # from exact marginalization rather than separate passes.
    symbol_memo = {}
    stream_dist = _after_bos_distribution(
        grammar, max_n, weights, symbol_memo,
        bos_char=tokenizer.bos_string, eos_char=tokenizer.eos_string,
    )

    # Scatter the tuple->prob dict into the dense (V,)*max_n array,
    # mapping characters to ids through the tokenizer's encode table.
    deepest = np.zeros((vocab_size,) * max_n, dtype=np.float64)
    for seq, p in stream_dist.items():
        idx = tuple(tokenizer.encode_vocab[c] for c in seq)
        deepest[idx] += p

    # Marginalize down: summing out the last axis of n{k} yields n{k-1}
    # because conditionals sum to 1. This guarantees the same nesting
    # invariant the empirical count arrays satisfy.
    arrays = {f"n{max_n}": deepest}
    for k in range(max_n - 1, 0, -1):
        arrays[f"n{k}"] = arrays[f"n{k + 1}"].sum(axis=-1)

    arrays["prefix"] = np.array([bos_id], dtype=np.int64)
    return arrays
