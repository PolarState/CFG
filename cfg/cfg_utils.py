from nltk import CFG
from nltk.grammar import Nonterminal


def count_generations(grammar):
    """
    Count the total number of sentences (derivation paths) a CFG can produce.
    Returns the count for the start symbol, or None if the grammar is recursive.
    """
    COMPUTING = object()
    counts = {}

    def compute(symbol):
        if not isinstance(symbol, Nonterminal):
            return 1

        if symbol in counts:
            if counts[symbol] is COMPUTING:
                return None  # recursive grammar -> infinite
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

    result = compute(grammar.start())

    if result is not None:
        print(f"\nPer-nonterminal counts:")
        for nt, count in sorted(counts.items(), key=lambda x: str(x[0])):
            print(f"  {nt} -> {count:,}")
        print(f"\nTotal generations from {grammar.start()}: {result:,}")

    return result


if __name__ == "__main__":
    # Test against the grammars from nltk_exhaustive.py
    cfg3b_short_str = """
        16 -> 15 13
        15 -> 12 11 10 | 11 12 10
        14 -> 11 10 12 | 10 11 12
        13 -> 11 12 | 12 11
        12 -> 8 9 7 | 9 7 8
        11 -> 8 7 9 | 7 8 9
        10 -> 7 9 8 | 9 8 7
        9 -> '321' | '21'
        8 -> '32' | '312'
        7 -> '31' | '123'
    """

    cfg3b_str = """
        22 -> 21 20 | 20 19

        21 -> 18 16 | 16 18 17
        20 -> 17 16 18 | 16 17
        19 -> 16 17 18 | 17 18 16

        18 -> 15 14 13 | 14 13
        17 -> 14 13 15 | 15 13 14
        16 -> 15 13 | 13 15 14

        15 -> 12 11 10 | 11 12 10
        14 -> 11 10 12 | 10 11 12
        13 -> 11 12 | 12 11

        12 -> 8 9 7 | 9 7 8
        11 -> 8 7 9 | 7 8 9
        10 -> 7 9 8 | 9 8 7

        9 -> '321' | '21'
        8 -> '32' | '312'
        7 -> '31' | '123'
    """

    print("=== Short grammar (start: 16) ===")
    cfg_short = CFG.fromstring(cfg3b_short_str)
    count_generations(cfg_short)

    print("\n=== Full grammar (start: 22) ===")
    cfg_full = CFG.fromstring(cfg3b_str)
    count_generations(cfg_full)
