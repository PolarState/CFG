import csv
from typing import Any

def save_cfg_to_csv(cfg_rules: dict[Any, Any], filename: str):
    """
    Save a CFG (provided as a dictionary) to a CSV file.
    
    Each key in cfg_rules is the LHS of a production, and its value is a list of productions.
    Each production is a list of symbols (numbers or strings).
    
    Args:
        cfg_rules (dict): The CFG rules dictionary.
        filename (str): The output CSV filename.
    """
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for lhs, productions in cfg_rules.items():
            lhs_str = str(lhs)  # Convert LHS to string
            for production in productions:
                # Convert each symbol in the production to string
                row = [lhs_str] + [str(sym) for sym in production]
                writer.writerow(row)

def load_from_csv(filename: str) -> dict[Any, Any]:
    """
    Load CFG rules from a CSV file and return them as a dictionary.
    
    Each row in the CSV should have at least two columns:
        - The first column is the left-hand side (nonterminal).
        - The remaining columns form one production rule.
    
    Example row: S,A,B  (which means S -> A B)
    
    Returns:
        dict: A dictionary mapping nonterminals to a list of production rules.
    """
    rules = {}
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Skip rows that do not have at least LHS and one RHS symbol.
            if len(row) < 2:
                continue
            lhs = row[0].strip()
            # Build production rule by stripping whitespace from each symbol.
            production = [symbol.strip() for symbol in row[1:] if symbol.strip() != ""]
            if lhs in rules:
                rules[lhs].append(production)
            else:
                rules[lhs] = [production]
    return rules


if __name__ == "__main__":
    # Example CFG dictionary
    CFG_RULES = {
        "root": [[20, 21], [20, 19, 21], [21, 19, 19], [20, 20]],  # Root expansions
        19: [[18, 16, 18], [17, 18], [18, 18]],  # Recursive expansion
    }
    
    output_file = "cfg_rules.csv"
    save_cfg_to_csv(CFG_RULES, output_file)
    print(f"CFG saved to {output_file}")
