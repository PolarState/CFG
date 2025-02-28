import math
import random
from typing import Optional

import cfg_defines


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


def validate_string(input: str, start_symbol: str, cfg_rules: dict[str, str]):
    # TODO: derive the start symbol(s)

    # Reverse dictionary.
    reverse_cfg_rules = {}
    for k, values in cfg_rules.items():
        for v in values:
            reverse_cfg_rules[tuple(v)] = k

    # Find all terminal symbols in cfg.
    terminal_symbols = []
    for values in cfg_rules.values():
        for value in values:
            for v in value:
                if v not in cfg_rules:
                    terminal_symbols.append(v)

    # Split input into terminal symbols.
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


cfg_rules = cfg_defines.cfg3b

# Example Usage
generated_string = generate_from_cfg("22", cfg_rules)
print("Generated CFG String:", generated_string)

# Validate a given sequence
print(
    f"Is '{generated_string}' in the CFG language? {validate_string(generated_string, '22', cfg_rules)}"
)
