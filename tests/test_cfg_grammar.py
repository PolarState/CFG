# Tests for the CFGrammar class in cfg_grammar.py.
# These verify that the class produces the same results as the free
# functions it consolidates, and that cached state is correct.

from cfg.cfg_grammar import CFGrammar
from cfg.cfg_defines import cfg_by_name
from cfg import cfg_generator, cfg_utils


def test_construction_from_dict():
    """CFGrammar can be constructed from a raw grammar dict."""
    grammar = CFGrammar(cfg_by_name["cfg3b"])
    assert grammar.rules is cfg_by_name["cfg3b"]
    assert grammar.name is None


def test_construction_from_name():
    """CFGrammar.from_name looks up the grammar and stores the name."""
    grammar = CFGrammar.from_name("cfg3b")
    assert grammar.rules == cfg_by_name["cfg3b"]
    assert grammar.name == "cfg3b"


def test_terminal_symbols_match():
    """Eagerly cached terminal_symbols match the free function."""
    for name, rules in cfg_by_name.items():
        grammar = CFGrammar(rules, name=name)
        expected = sorted(set(cfg_generator.get_terminal_symbols(rules)))
        assert grammar.terminal_symbols == expected, (
            f"terminal_symbols mismatch for {name}"
        )


def test_start_symbols_match():
    """Eagerly cached start_symbols match the free function."""
    for name, rules in cfg_by_name.items():
        grammar = CFGrammar(rules, name=name)
        expected = sorted(set(cfg_generator.get_start_symbols(rules)))
        assert grammar.start_symbols == expected, (
            f"start_symbols mismatch for {name}"
        )


def test_nonterminal_symbols():
    """Nonterminal symbols are exactly the keys of the grammar dict."""
    for name, rules in cfg_by_name.items():
        grammar = CFGrammar(rules, name=name)
        assert grammar.nonterminal_symbols == set(rules.keys())


def test_count_generations_match():
    """count_generations matches the free function for all grammars."""
    for name, rules in cfg_by_name.items():
        grammar = CFGrammar(rules, name=name)
        start = grammar.start_symbols[0]
        expected = cfg_utils.count_generations(start, rules)
        assert grammar.count_generations(start) == expected, (
            f"count_generations mismatch for {name}"
        )


def test_count_generations_cached():
    """Calling count_generations twice returns the cached result."""
    grammar = CFGrammar.from_name("cfg3b")
    first = grammar.count_generations()
    second = grammar.count_generations()
    assert first == second
    assert first is second  # Same object, not recomputed.


def test_longest_sequence_match():
    """get_longest_sequence matches the free function."""
    for name, rules in cfg_by_name.items():
        grammar = CFGrammar(rules, name=name)
        start = grammar.start_symbols[0]
        expected = cfg_generator.get_longest_sequence(start, rules)
        assert grammar.get_longest_sequence(start) == expected, (
            f"get_longest_sequence mismatch for {name}"
        )


def test_generate_produces_valid_strings():
    """Generated strings validate against the grammar."""
    grammar = CFGrammar.from_name("cfg3b")
    for _ in range(50):
        s = grammar.generate()
        assert grammar.validate(s), f"Generated string failed validation: {s}"


def test_generate_uniform_produces_valid_strings():
    """Uniformly sampled strings validate against the grammar."""
    grammar = CFGrammar.from_name("cfg3b")
    for _ in range(50):
        s = grammar.generate_uniform()
        assert grammar.validate(s), f"Uniform string failed validation: {s}"


def test_validate_rejects_invalid():
    """Strings not in the language are rejected."""
    grammar = CFGrammar.from_name("cfg3b")
    assert not grammar.validate("not_a_valid_cfg_string")
    assert not grammar.validate("")


def test_uniform_weights_match():
    """uniform_weights matches the free function."""
    for name, rules in cfg_by_name.items():
        grammar = CFGrammar(rules, name=name)
        expected = cfg_utils.uniform_sentence_weights(rules)
        assert grammar.uniform_weights == expected, (
            f"uniform_weights mismatch for {name}"
        )


def test_repr():
    """__repr__ includes useful info."""
    grammar = CFGrammar.from_name("cfg3b")
    r = repr(grammar)
    assert "cfg3b" in r
    assert "terminals=" in r
    assert "nonterminals=" in r
