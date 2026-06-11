# Utility functions for analyzing CFG properties.
# These operate on the same dict[str, list[list[str]]] grammar format
# used by cfg_defines and cfg_generator.


def count_generations(start_symbol, cfg_rules):
    """Count the total number of sentences (derivation paths) a CFG can produce.

    Args:
        start_symbol: the nonterminal to start from.
        cfg_rules: dictionary of nonterminal -> list of production sequences.

    Returns:
        The count for the start symbol, or None if the grammar is recursive.
    """
    # Sentinel to detect cycles. If we encounter a symbol that is currently
    # being computed, we know we've hit a recursive rule which means the
    # grammar produces infinite sentences.
    COMPUTING = object()
    counts = {}

    def compute(symbol):
        # If the symbol isn't a key in the grammar, it's a terminal.
        # Terminals contribute exactly 1 to the count of any production
        # they appear in.
        if symbol not in cfg_rules:
            return 1

        # If we've already computed or started computing this symbol:
        if symbol in counts:
            # If we see the sentinel, we've found a cycle.
            if counts[symbol] is COMPUTING:
                return None
            # Otherwise return the cached count.
            return counts[symbol]

        # Mark this symbol as in-progress before recursing.
        counts[symbol] = COMPUTING

        # For each production rule of this nonterminal, the number of
        # sentences it generates is the product of the counts of each
        # symbol in the rule (each choice multiplies combinatorially).
        # The total for the nonterminal is the sum across all its rules.
        total = 0
        for production in cfg_rules[symbol]:
            prod_count = 1
            for sym in production:
                sym_count = compute(sym)
                # Propagate cycle detection upward.
                if sym_count is None:
                    counts[symbol] = None
                    return None
                prod_count *= sym_count
            total += prod_count

        counts[symbol] = total
        return total

    return compute(start_symbol)


def count_generations_verbose(start_symbol, cfg_rules):
    """Same as count_generations but prints per-nonterminal breakdown."""
    result = count_generations(start_symbol, cfg_rules)

    if result is not None:
        # Recompute per-nonterminal counts for display.
        counts = _count_per_nonterminal(cfg_rules)
        print(f"\nPer-nonterminal counts:")
        for nt in sorted(counts.keys()):
            print(f"  {nt} -> {counts[nt]:,}")
        print(f"\nTotal generations from {start_symbol}: {result:,}")

    return result


def _count_per_nonterminal(cfg_rules):
    """Return a dict mapping each nonterminal to its generation count.

    Unlike count_generations, this computes counts for ALL nonterminals
    in the grammar, not just those reachable from a given start symbol.
    It also does not handle recursive grammars (no cycle detection).
    """
    counts = {}

    def compute(symbol):
        # Terminals always count as 1.
        if symbol not in cfg_rules:
            return 1
        # Return cached result if we've already computed this nonterminal.
        if symbol in counts:
            return counts[symbol]

        # Sum over production rules, product within each rule.
        # See count_generations for the explanation of why this works.
        total = 0
        for production in cfg_rules[symbol]:
            prod_count = 1
            for sym in production:
                prod_count *= compute(sym)
            total += prod_count

        counts[symbol] = total
        return total

    # Compute for every nonterminal in the grammar.
    for nt in cfg_rules:
        compute(nt)

    return counts


def _count_per_production(cfg_rules):
    """Return a dict mapping each nonterminal to a list of per-production counts.

    Where _count_per_nonterminal gives a single total per nonterminal,
    this breaks it down: how many sentences does each individual production
    rule generate? This is the building block for weighted sampling.
    """
    # First get the total count for each nonterminal so we can look up
    # child counts without recomputing.
    nt_counts = _count_per_nonterminal(cfg_rules)
    prod_counts = {}

    for nt, productions in cfg_rules.items():
        counts = []
        for production in productions:
            # Each production's count is the product of its children's counts.
            # Terminals aren't in nt_counts, so default to 1.
            prod_count = 1
            for sym in production:
                prod_count *= nt_counts.get(sym, 1)
            counts.append(prod_count)
        prod_counts[nt] = counts

    return prod_counts


def parse_mask_rule(spec, cfg_rules):
    """Parse '<NT>:<INDEX>' (e.g. '7:0') and validate against the grammar.

    Shared by the dataset/trace builders and the exact-distribution
    script so the masking syntax can never drift between them.
    """
    # The spec must contain a separator; split on the first ':' only so
    # the index part can be validated as an integer below.
    if ":" not in spec:
        raise ValueError(f"--mask-rule must be 'NT:INDEX', got {spec!r}")
    nt, idx_str = spec.split(":", 1)
    # The nonterminal must exist in the grammar.
    if nt not in cfg_rules:
        raise ValueError(f"NT {nt!r} not in grammar; valid: {sorted(cfg_rules.keys())}")
    # The rule index must be in range for that nonterminal.
    idx = int(idx_str)
    if not (0 <= idx < len(cfg_rules[nt])):
        raise ValueError(
            f"NT {nt!r} has {len(cfg_rules[nt])} rules, index {idx} out of range"
        )
    return nt, idx


def build_mask_weights(cfg_rules, mask_nt, mask_idx):
    """Weights dict: 1.0 for every rule except the masked one (0.0).

    Passed to CFGrammar.generate(weights=...). random.choices treats a
    weight of 0 as never-selected, so the masked production is fully
    suppressed without modifying the grammar object.
    """
    return {
        nt: [
            0.0 if (nt == mask_nt and i == mask_idx) else 1.0
            for i in range(len(prods))
        ]
        for nt, prods in cfg_rules.items()
    }


def uniform_sentence_weights(cfg_rules):
    """Compute production weights that sample uniformly over all possible sentences.

    Without weights, generate_from_cfg picks each production rule with equal
    probability. This means nonterminals with fewer downstream sentences get
    overrepresented (see test_count_generations.py for a concrete example).

    By weighting each production rule in proportion to how many sentences it
    leads to, every sentence in the language becomes equally likely to be
    sampled. The weights returned here can be passed directly to
    generate_from_cfg's weights parameter.

    Returns:
        A dict mapping each nonterminal to a list of weights (one per production
        rule). random.choices normalizes these internally, so raw counts work
        as weights — no need to convert to probabilities.
    """
    return _count_per_production(cfg_rules)
