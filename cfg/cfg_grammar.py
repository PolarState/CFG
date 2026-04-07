# This module defines the CFGrammar class, which consolidates the grammar
# functions from cfg_generator.py and cfg_utils.py into a single object.
# The class caches derived state (terminal symbols, start symbols, generation
# counts, sampling weights) so they are computed once rather than re-derived
# on every call.

import random


class CFGrammar:
    """Encapsulates a context-free grammar and caches derived state.

    The grammar is stored as a dict mapping nonterminal symbols to lists of
    production rules, where each rule is a list of symbols (terminals or
    nonterminals). This is the same dict[str, list[list[str]]] format used
    throughout the cfg package.
    """

    def __init__(
        self,
        cfg_rules: dict[str, list[list[str]]],
        name: str | None = None,
    ) -> None:
        # Store the raw grammar dict and optional name for display.
        self.rules = cfg_rules
        self.name = name

        # Nonterminal symbols are exactly the keys of the grammar dict.
        # Every key has at least one production rule.
        self.nonterminal_symbols = set(cfg_rules.keys())

        # Terminal symbols are symbols that appear on the right-hand side of
        # production rules but are never defined as a nonterminal (i.e. they
        # have no production rules of their own). We collect them by scanning
        # every symbol in every production and keeping those not in the keys.
        terminal_set = set()
        for productions in cfg_rules.values():
            for production in productions:
                for symbol in production:
                    if symbol not in self.nonterminal_symbols:
                        terminal_set.add(symbol)
        self.terminal_symbols = sorted(terminal_set)

        # Start symbols are nonterminals that never appear on the right-hand
        # side of any production rule. They can only be reached as the root
        # of a derivation. We find them by collecting all RHS symbols and
        # taking the set difference with all nonterminals.
        rhs_symbols = set()
        for productions in cfg_rules.values():
            for production in productions:
                for symbol in production:
                    rhs_symbols.add(symbol)
        self.start_symbols = sorted(
            self.nonterminal_symbols - rhs_symbols
        )

        # Lazy caches. These are computed on first access because they
        # require a full traversal of the grammar and are not always needed.
        self._uniform_weights = None
        self._generation_counts = {}
        self._longest_sequences = {}
        self._reverse_rules = None

    @classmethod
    def from_name(cls, name: str) -> "CFGrammar":
        """Create a CFGrammar by looking up a grammar name from cfg_defines."""
        # Import deferred to the method body to avoid circular imports,
        # since cfg_defines is a sibling module in the same package.
        from . import cfg_defines
        return cls(cfg_defines.get_cfg(name), name=name)

    # ── Generation ──────────────────────────────────────────────────────

    def generate(
        self,
        symbol: str | None = None,
        weights: dict[str, list[float]] | None = None,
    ) -> str:
        """Generate a random string from the grammar by recursive expansion.

        Args:
            symbol: the nonterminal to start expanding from. If None, a
                random start symbol is chosen.
            weights: optional dict mapping each nonterminal to a list of
                floats (one per production rule) used to bias selection.
                If None, all productions are equally likely.

        Returns:
            A string composed entirely of terminal symbols.
        """
        # If no start symbol was given, pick one at random from the
        # grammar's start symbols.
        if symbol is None:
            symbol = random.choice(self.start_symbols)

        # Recursively expand the symbol into a string of terminals.
        return self._expand(symbol, weights)

    def _expand(
        self,
        symbol: str,
        weights: dict[str, list[float]] | None,
    ) -> str:
        """Recursive helper for generate. Expands a single symbol."""
        # If the symbol is a terminal, return it directly. Terminals are
        # the base case of the recursion.
        if symbol not in self.rules:
            return symbol

        productions = self.rules[symbol]

        # If weights are provided for this nonterminal, use them to bias
        # the selection. random.choices handles normalization internally,
        # so raw counts (e.g. from uniform_weights) work directly.
        if weights is not None and symbol in weights:
            production = random.choices(
                productions, weights=weights[symbol]
            )[0]
        # Otherwise, pick uniformly among the production rules.
        # Note: this is uniform over *rules*, not over *sentences* —
        # branches with fewer downstream sentences will be overrepresented.
        else:
            production = random.choice(productions)

        # Recursively expand each symbol in the chosen production and
        # concatenate the results into a single string.
        return "".join(self._expand(sym, weights) for sym in production)

    def generate_uniform(self, symbol: str | None = None) -> str:
        """Generate a string with uniform probability over all sentences.

        This uses weights proportional to the number of sentences each
        production rule can generate, so every possible sentence in the
        language is equally likely to be sampled.
        """
        return self.generate(symbol, weights=self.uniform_weights)

    # ── Validation ──────────────────────────────────────────────────────

    def validate(
        self,
        input_string: str,
        start_symbol: str | None = None,
    ) -> bool:
        """Check whether a string belongs to the language of this grammar.

        Args:
            input_string: the string to validate.
            start_symbol: the nonterminal the string must derive from.
                If None, tries each start symbol and returns True if any
                matches.

        Returns:
            True if the string is a valid derivation, False otherwise.
        """
        # If no start symbol specified, try each one. The string is valid
        # if it can be derived from any start symbol.
        if start_symbol is None:
            return any(
                self.validate(input_string, s) for s in self.start_symbols
            )

        # Lazily build the reverse lookup table on first validation call.
        # This maps tuples of symbols back to the nonterminal they came
        # from, enabling bottom-up parsing.
        if self._reverse_rules is None:
            self._reverse_rules = {}
            for nonterminal, productions in self.rules.items():
                for production in productions:
                    self._reverse_rules[tuple(production)] = nonterminal

        # Split the input string into a list of terminal symbols.
        # WARNING: this won't work if terminal symbols overlap ('1' and '11').
        #   A DP approach like parse_cfg below could be adopted if needed.
        max_t = max(len(t) for t in self.terminal_symbols)
        start = 0
        stop = 1
        tape = []
        while stop <= len(input_string):
            found_t = False
            while stop - start <= max_t and not found_t:
                if input_string[start:stop] in self.terminal_symbols:
                    tape.append(input_string[start:stop])
                    start = stop
                    found_t = True
                stop += 1
            # If we can't find a terminal symbol, the string contains
            # characters outside the grammar's alphabet.
            if not found_t:
                return False

        # Maximum number of symbols in any reverse rule key. This bounds
        # the sliding window in the bottom-up parser below.
        max_symbol_len = max(
            len(key) for key in self._reverse_rules.keys()
        )

        def parse_cfg(input_tape, output_tape):
            """Bottom-up parser that reduces terminal sequences to nonterminals.

            Reads symbols from input_tape, tries to match windows against
            reverse_rules, and pushes reductions onto output_tape. When the
            input tape is empty, the output tape becomes the new input and
            we repeat. Success is reaching a single start symbol.
            """
            # If our input tape is empty and we still have an output tape:
            if not input_tape and output_tape:
                # Check if the output tape is the start symbol.
                if output_tape == [start_symbol]:
                    return True
                # Otherwise parse the output tokens.
                else:
                    return parse_cfg(output_tape, [])
            # If there is no valid input or output tape exit.
            elif not input_tape and not output_tape:
                return False

            # Try matching every window length from the input to the rules.
            for window in range(1, max_symbol_len + 1):
                next_tuple = tuple(input_tape[:window])
                # If we get a window match, continue parsing.
                if next_tuple in self._reverse_rules and parse_cfg(
                    input_tape[window:],
                    output_tape + [self._reverse_rules[next_tuple]],
                ):
                    return True

            # If there are no true matches, return false.
            return False

        return parse_cfg(tape, [])

    # ── Analysis ────────────────────────────────────────────────────────

    @property
    def uniform_weights(self) -> dict[str, list[float]]:
        """Production weights that sample uniformly over all possible sentences.

        Without weights, generate() picks each production rule with equal
        probability. This overrepresents nonterminals with fewer downstream
        sentences. By weighting each rule in proportion to how many sentences
        it leads to, every sentence becomes equally likely.

        Computed lazily on first access and cached for subsequent calls.
        """
        if self._uniform_weights is None:
            # First compute the total generation count for each nonterminal.
            nt_counts = self._count_per_nonterminal()

            # Then for each nonterminal, compute how many sentences each
            # individual production rule generates. This is the product of
            # the counts of each symbol in the rule.
            prod_counts = {}
            for nt, productions in self.rules.items():
                counts = []
                for production in productions:
                    prod_count = 1
                    for sym in production:
                        # Terminals aren't in nt_counts, so default to 1.
                        prod_count *= nt_counts.get(sym, 1)
                    counts.append(prod_count)
                prod_counts[nt] = counts

            self._uniform_weights = prod_counts

        return self._uniform_weights

    def count_generations(self, start_symbol: str | None = None) -> int | None:
        """Count the total number of sentences the grammar can produce.

        Args:
            start_symbol: the nonterminal to count from. If None, uses the
                first start symbol.

        Returns:
            The count, or None if the grammar is recursive (infinite).
        """
        # Default to the first start symbol if none specified.
        if start_symbol is None:
            start_symbol = self.start_symbols[0]

        # Return cached result if we've already computed this start symbol.
        if start_symbol in self._generation_counts:
            return self._generation_counts[start_symbol]

        # Sentinel to detect cycles. If we encounter a symbol that is
        # currently being computed, we've found a recursive rule which
        # means the grammar produces infinite sentences.
        COMPUTING = object()
        counts = {}

        def compute(symbol):
            # If the symbol isn't a key in the grammar, it's a terminal.
            # Terminals contribute exactly 1 to the count.
            if symbol not in self.rules:
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

            # For each production rule, the number of sentences it generates
            # is the product of the counts of each symbol in the rule.
            # The total for the nonterminal is the sum across all its rules.
            total = 0
            for production in self.rules[symbol]:
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

        result = compute(start_symbol)
        self._generation_counts[start_symbol] = result
        return result

    def get_longest_sequence(self, start_symbol: str | None = None) -> int:
        """Compute the length of the longest terminal string derivable.

        Only works with non-recursive (acyclic) grammars.

        Args:
            start_symbol: the nonterminal to compute from. If None, uses
                the first start symbol.

        Returns:
            The length in terminal symbols of the longest derivation.
        """
        # Default to the first start symbol if none specified.
        if start_symbol is None:
            start_symbol = self.start_symbols[0]

        # Return cached result if available.
        if start_symbol in self._longest_sequences:
            return self._longest_sequences[start_symbol]

        # Build a length table bottom-up. Start with terminals at length 1,
        # then compute each nonterminal by taking the max production length.
        cfg_lengths = {ts: 1 for ts in self.terminal_symbols}
        rules = list(self.rules.keys())

        while rules:
            next_rule = rules.pop()

            # Collect all symbols referenced by this nonterminal's productions.
            all_syms = set()
            for production in self.rules[next_rule]:
                for sym in production:
                    all_syms.add(sym)

            # Check if we have lengths for all referenced symbols.
            # If not, defer this rule and try again later.
            if not all(sym in cfg_lengths for sym in all_syms):
                rules.insert(0, next_rule)
                continue

            # Compute the length of each production (sum of child lengths)
            # and take the maximum across all productions.
            generation_lengths = []
            for production in self.rules[next_rule]:
                list_length = sum(cfg_lengths[sym] for sym in production)
                generation_lengths.append(list_length)
            cfg_lengths[next_rule] = max(generation_lengths)

        result = cfg_lengths[start_symbol]
        self._longest_sequences[start_symbol] = result
        return result

    # ── Private helpers ─────────────────────────────────────────────────

    def _count_per_nonterminal(self) -> dict[str, int]:
        """Return a dict mapping each nonterminal to its total generation count.

        This computes counts for ALL nonterminals in the grammar, not just
        those reachable from a given start symbol. It does not handle
        recursive grammars (no cycle detection).
        """
        counts = {}

        def compute(symbol):
            # Terminals always count as 1.
            if symbol not in self.rules:
                return 1
            # Return cached result if already computed.
            if symbol in counts:
                return counts[symbol]

            # Sum over production rules, product within each rule.
            total = 0
            for production in self.rules[symbol]:
                prod_count = 1
                for sym in production:
                    prod_count *= compute(sym)
                total += prod_count

            counts[symbol] = total
            return total

        # Compute for every nonterminal in the grammar.
        for nt in self.rules:
            compute(nt)

        return counts

    # ── Dunder methods ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CFGrammar("
            f"name={self.name!r}, "
            f"terminals={len(self.terminal_symbols)}, "
            f"nonterminals={len(self.nonterminal_symbols)}, "
            f"start_symbols={self.start_symbols}"
            f")"
        )
