"""Side-by-side tests: native count_generations vs NLTK-based count."""

from nltk import CFG as NLTK_CFG
from nltk.grammar import Nonterminal

from cfg.cfg_defines import cfg_by_name
from cfg.cfg_utils import count_generations
from cfg.cfg_generator import get_start_symbols


# ── NLTK implementation (for comparison) ──────────────────────────────────

def nltk_count_generations(grammar):
    """NLTK-based count of derivation paths."""
    COMPUTING = object()
    counts = {}

    def compute(symbol):
        if not isinstance(symbol, Nonterminal):
            return 1
        if symbol in counts:
            if counts[symbol] is COMPUTING:
                return None
            return counts[symbol]
        counts[symbol] = COMPUTING
        total = 0
        for prod in grammar.productions(lhs=symbol):
            prod_count = 1
            for sym in prod.rhs():
                sym_count = compute(sym)
                if sym_count is None:
                    counts[symbol] = None
                    return None
                prod_count *= sym_count
            total += prod_count
        counts[symbol] = total
        return total

    return compute(grammar.start())


def dict_to_nltk_string(cfg_rules, start_symbol):
    """Convert a cfg_defines dict grammar to an NLTK grammar string."""
    lines = []
    terminal_symbols = set()
    for values in cfg_rules.values():
        for production in values:
            for sym in production:
                if sym not in cfg_rules:
                    terminal_symbols.add(sym)

    # Start symbol must come first for NLTK.
    if start_symbol in cfg_rules:
        prods = cfg_rules[start_symbol]
        rhs = " | ".join(" ".join(s if s in cfg_rules else f"'{s}'" for s in p) for p in prods)
        lines.append(f"    {start_symbol} -> {rhs}")

    for nt, prods in cfg_rules.items():
        if nt == start_symbol:
            continue
        rhs = " | ".join(" ".join(s if s in cfg_rules else f"'{s}'" for s in p) for p in prods)
        lines.append(f"    {nt} -> {rhs}")

    return "\n".join(lines)


# ── Tests ─────────────────────────────────────────────────────────────────

def test_grammar(name, cfg_rules):
    start_symbols = get_start_symbols(cfg_rules)
    assert len(start_symbols) == 1, f"{name}: expected 1 start symbol, got {start_symbols}"
    start = start_symbols[0]

    # Native
    native_result = count_generations(start, cfg_rules)

    # NLTK
    nltk_str = dict_to_nltk_string(cfg_rules, start)
    nltk_grammar = NLTK_CFG.fromstring(nltk_str)
    nltk_result = nltk_count_generations(nltk_grammar)

    match = "PASS" if native_result == nltk_result else "FAIL"
    print(f"  {name:8s}  native={native_result:<45,}  nltk={nltk_result:<45,}  [{match}]")
    return native_result == nltk_result


def test_small_grammar_against_exhaustive():
    """Cross-check against actual NLTK exhaustive enumeration on a small grammar."""
    from nltk.parse.generate import generate

    # Use just the bottom layer of cfg3b — small enough to enumerate.
    small_cfg = {
        "12": [["8", "9", "7"], ["9", "7", "8"]],
        "11": [["8", "7", "9"], ["7", "8", "9"]],
        "10": [["7", "9", "8"], ["9", "8", "7"]],
        "9": [["3", "2", "1"], ["2", "1"]],
        "8": [["3", "2"], ["3", "1", "2"]],
        "7": [["3", "1"], ["1", "2", "3"]],
    }

    for start in ["10", "11", "12"]:
        native_count = count_generations(start, small_cfg)

        nltk_str = dict_to_nltk_string(small_cfg, start)
        nltk_grammar = NLTK_CFG.fromstring(nltk_str)
        exhaustive_count = sum(1 for _ in generate(nltk_grammar))

        match = "PASS" if native_count == exhaustive_count else "FAIL"
        print(f"  {start:8s}  native={native_count:<10,}  exhaustive={exhaustive_count:<10,}  [{match}]")


def test_weighted_sampling_uniformity():
    """Verify that uniform_sentence_weights produces a flat distribution over sentences."""
    from collections import Counter
    from cfg.cfg_utils import uniform_sentence_weights
    from cfg.cfg_generator import generate_from_cfg

    # Small grammar: S has 2 rules, one leads to 4 sentences, the other to 1.
    # Without weights: P(branch A) = 0.5, P(branch B) = 0.5 -> non-uniform over sentences.
    # With weights: P(branch A) = 4/5, P(branch B) = 1/5 -> uniform over sentences.
    cfg = {
        "S": [["A", "B"], ["C"]],
        "A": [["1"], ["2"]],
        "B": [["3"], ["4"]],
        "C": [["5"]],
    }

    n_samples = 50_000
    weights = uniform_sentence_weights(cfg)

    # Sample with weights (should be ~uniform across 5 sentences).
    weighted_counts = Counter()
    for _ in range(n_samples):
        weighted_counts[generate_from_cfg("S", cfg, weights)] += 1

    # Sample without weights (should be biased toward "5").
    unweighted_counts = Counter()
    for _ in range(n_samples):
        unweighted_counts[generate_from_cfg("S", cfg)] += 1

    expected_per_sentence = n_samples / 5

    print(f"  Expected per sentence: ~{expected_per_sentence:.0f}")
    print(f"\n  {'sentence':<10} {'weighted':>10} {'unweighted':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    for s in sorted(set(weighted_counts) | set(unweighted_counts)):
        print(f"  {s:<10} {weighted_counts[s]:>10} {unweighted_counts[s]:>10}")

    # Check that weighted sampling is roughly uniform (within 20% of expected).
    all_close = True
    for s, count in weighted_counts.items():
        if abs(count - expected_per_sentence) / expected_per_sentence > 0.20:
            all_close = False

    # Check that unweighted "5" is significantly overrepresented.
    unweighted_biased = unweighted_counts["5"] > expected_per_sentence * 1.5

    w_pass = "PASS" if all_close else "FAIL"
    u_pass = "PASS" if unweighted_biased else "FAIL"
    print(f"\n  Weighted uniform:   [{w_pass}]")
    print(f"  Unweighted biased:  [{u_pass}] ('5' should be overrepresented)")
    return all_close and unweighted_biased


if __name__ == "__main__":
    print("=== Side-by-side: native vs NLTK (all grammars) ===\n")
    all_passed = True
    for name, cfg_rules in cfg_by_name.items():
        if not test_grammar(name, cfg_rules):
            all_passed = False

    print("\n=== Cross-check: native vs NLTK exhaustive enumeration (small grammar) ===\n")
    test_small_grammar_against_exhaustive()

    print("\n=== Weighted vs unweighted sampling distribution ===\n")
    if not test_weighted_sampling_uniformity():
        all_passed = False

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
