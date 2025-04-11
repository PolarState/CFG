# Functions to generate and validate cfg strings from cfg rules.
# It would be helpful to create either a class to encapsulate this
# or a dataclass which would hold some state. Reason being, the cfg
# rules dont hold enough state. We have to derive the terminal and start
# symbols in a few places and re-deriving them is inefficient.
# Ultimately this isn't something that is hard or particularly time
# consuming but if done in a loop is wasteful.
# ^TODO if generation or validation is in the critical path.

import random


def generate_from_cfg(
    symbol: str,
    cfg_rules: dict[str, str],
) -> str:
    """Generate a string from the CFG using recursive expansion.

    Args:
        symbol: next symbol to select from.
        cfg_rules: dictionary of tokens to recursively sample from.

    Returns:
        cfg string of length max_depth or exclusively terminal symbols.
    """
    if symbol not in cfg_rules:  # Terminal symbol reached.
        return symbol
    production = random.choice(cfg_rules[symbol])  # Randomly pick a production rule
    return "".join(generate_from_cfg(sym, cfg_rules) for sym in production)


def get_terminal_symbols(cfg_rules):
    # Find all terminal symbols in cfg.
    terminal_symbols = []
    for values in cfg_rules.values():
        for value in values:
            for v in value:
                if v not in cfg_rules:
                    terminal_symbols.append(v)

    return terminal_symbols


def _flatten_symbols(symbols):
    flat_symbols = []
    for symbol in symbols:
        if isinstance(symbol, list):
            flat_symbols.extend(_flatten_symbols(symbol))
        else:
            flat_symbols.append(symbol)
    return flat_symbols


def get_start_symbols(cfg_rules) -> list[str]:
    rules = list(cfg_rules.keys())

    # Create a set of all symbols created FROM a production rule.
    output_symbols = set()
    for rule in rules:
        output_symbols = output_symbols.union(set(_flatten_symbols(cfg_rules[rule])))

    # Find production rule that is not in the set created.
    start_symbols = set()
    for rule in rules:
        if rule not in output_symbols:
            start_symbols.add(rule)

    return start_symbols


# Only works with no cyclic dependencies. I forget if this is a pretense of CFGs.
def get_longest_sequence(start_symbol, cfg_rules):
    terminal_symbols = set(get_terminal_symbols(cfg_rules))
    cfg_lengths = {ts: 1 for ts in terminal_symbols}
    rules = list(cfg_rules.keys())

    while rules:
        next_rule = rules.pop()
        nts_set = set(_flatten_symbols(cfg_rules[next_rule]))
        calculate_length = True
        for nts in nts_set:
            # if we don't find all the symbols we need in our lengths, skip for now.
            if nts not in cfg_lengths:
                calculate_length = False
                break

        if calculate_length:
            generation_lengths = []
            for generation_list in cfg_rules[next_rule]:
                list_length = 0
                for generation_rule in generation_list:
                    list_length += cfg_lengths[generation_rule]
                generation_lengths.append(list_length)
            cfg_lengths[next_rule] = max(generation_lengths)
        else:
            rules.append(next_rule)

    return cfg_lengths[start_symbol]


def validate_string(input: str, start_symbol: str, cfg_rules: dict[str, str]):
    # TODO: derive the start symbol(s)

    # Reverse dictionary.
    reverse_cfg_rules = {}
    for k, values in cfg_rules.items():
        for v in values:
            reverse_cfg_rules[tuple(v)] = k

    # Find all terminal symbols in cfg.
    terminal_symbols = get_terminal_symbols(cfg_rules)

    # Split input string into array of terminal symbols.
    # WARNING: this won't work if terminal symbols overlap ('1' and '11').
    #   We can adopt a DP approach like parse_cfg if this is needed.
    max_t = max(len(t) for t in terminal_symbols)
    start = 0
    stop = 1
    tape = []
    while stop <= len(input):
        found_t = False
        while stop - start <= max_t and not found_t:
            if input[start:stop] in terminal_symbols:
                tape.append(input[start:stop])
                start = stop
                found_t = True
            stop += 1
        # If there is a non-terminal symbol then it's not a valid cfg.
        if not found_t:
            return False

    max_symbol_len = max([len(key) for key in reverse_cfg_rules.keys()])

    def parse_cfg(input_tape, output_tape):
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
            if next_tuple in reverse_cfg_rules and parse_cfg(
                input_tape[window:], output_tape + [reverse_cfg_rules[next_tuple]]
            ):
                return True

        # If there are no true matches, return false.
        return False

    # return parse_cfg(tape, [start_symbol], reverse_cfg_rules)
    return parse_cfg(tape, [])
