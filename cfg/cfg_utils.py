def count_generations(start_symbol, cfg_rules):
    """Count the total number of sentences (derivation paths) a CFG can produce.

    Args:
        start_symbol: the nonterminal to start from.
        cfg_rules: dictionary of nonterminal -> list of production sequences.

    Returns:
        The count for the start symbol, or None if the grammar is recursive.
    """
    COMPUTING = object()
    counts = {}

    def compute(symbol):
        if symbol not in cfg_rules:  # terminal
            return 1

        if symbol in counts:
            if counts[symbol] is COMPUTING:
                return None  # recursive grammar -> infinite
            return counts[symbol]

        counts[symbol] = COMPUTING

        total = 0
        for production in cfg_rules[symbol]:
            prod_count = 1
            for sym in production:
                sym_count = compute(sym)
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
    """Return a dict mapping each nonterminal to its generation count."""
    counts = {}

    def compute(symbol):
        if symbol not in cfg_rules:
            return 1
        if symbol in counts:
            return counts[symbol]

        total = 0
        for production in cfg_rules[symbol]:
            prod_count = 1
            for sym in production:
                prod_count *= compute(sym)
            total += prod_count

        counts[symbol] = total
        return total

    for nt in cfg_rules:
        compute(nt)

    return counts


def _count_per_production(cfg_rules):
    """Return a dict mapping each nonterminal to a list of per-production counts."""
    nt_counts = _count_per_nonterminal(cfg_rules)
    prod_counts = {}

    for nt, productions in cfg_rules.items():
        counts = []
        for production in productions:
            prod_count = 1
            for sym in production:
                prod_count *= nt_counts.get(sym, 1)
            counts.append(prod_count)
        prod_counts[nt] = counts

    return prod_counts


def uniform_sentence_weights(cfg_rules):
    """Compute production weights that sample uniformly over all possible sentences.

    Returns:
        A dict mapping each nonterminal to a list of floats (one per production
        rule), suitable for passing to generate_from_cfg as the weights argument.
    """
    return _count_per_production(cfg_rules)
