"""Aggregate JSONL parse traces and emit a Graphviz DOT of the grammar DAG.

One node per NT, one node per terminal, one "rule" midpoint per production.
Edge from NT to rule-midpoint is labeled with raw count and the fraction
that rule was chosen *among expansions of that NT*. Edges from rule-midpoint
to children are labeled with child position (1, 2, 3) so productions read
left-to-right.

The "DAG" framing means counts roll up by NT: every expansion of NT 7 is
pooled into the same two outgoing edges regardless of where in the tree
NT 7 was instantiated. Shared children appear once (e.g. terminal '3' has
incoming edges from many rule-midpoints).

Usage:

    python scripts/visualize_grammar_dag.py \\
        --traces traces_cfg3b_val_seed1.jsonl \\
        --output grammar_dag_val_seed1.dot

Then render:

    dot -Tsvg grammar_dag_val_seed1.dot \\
        -o grammar_dag_val_seed1.svg
"""

import argparse
import json
import sys
import time
from collections import Counter

from cfg.grammar.cfg_grammar import CFGrammar


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--traces", required=True, help="input JSONL traces file")
    p.add_argument("--cfg", default="cfg3b", help="grammar name from cfg_defines")
    p.add_argument("--output", required=True, help="output .dot path")
    p.add_argument(
        "--progress-seconds",
        type=float,
        default=10.0,
        help="print progress every N seconds while streaming traces",
    )
    return p.parse_args()


def walk(node, rule_counts, nt_counts):
    """Recursive tree walk: increment (nt, rule) and nt counts."""
    if "t" in node:
        return
    nt = node["nt"]
    rule_counts[(nt, node["rule"])] += 1
    nt_counts[nt] += 1
    for child in node["children"]:
        walk(child, rule_counts, nt_counts)


def aggregate(traces_path, progress_seconds):
    rule_counts = Counter()
    nt_counts = Counter()
    n_trees = 0
    t0 = time.time()
    last = t0
    with open(traces_path) as f:
        for line in f:
            walk(json.loads(line), rule_counts, nt_counts)
            n_trees += 1
            now = time.time()
            if now - last >= progress_seconds:
                rate = n_trees / (now - t0)
                print(
                    f"  {n_trees:>10,} trees  {rate:.0f}/s  elapsed={now - t0:.0f}s",
                    flush=True,
                )
                last = now
    return n_trees, rule_counts, nt_counts


def emit_dot(grammar, rule_counts, nt_counts, n_trees, source_label):
    nt_order = sorted(grammar.rules.keys(), key=int)
    terminals = list(grammar.terminal_symbols)
    lines = [
        "digraph grammar_dag {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica", labelloc="t", '
        f'label=<<b>cfg3b grammar DAG</b><br/>'
        f"<i>aggregated over {n_trees:,} trees from {source_label}</i>>];",
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=9];',
        "",
    ]

    # NT nodes
    lines.append("  // nonterminals")
    for nt in nt_order:
        n = nt_counts.get(nt, 0)
        lines.append(
            f'  "{nt}" [shape=ellipse, style=filled, fillcolor="#dbe9ff", '
            f'label=<<b>NT {nt}</b><br/><font point-size="9">{n:,} expansions</font>>];'
        )

    # Terminal nodes (rank=same so they sit on one row at the bottom)
    lines.append("")
    lines.append("  // terminals")
    lines.append("  { rank=sink;")
    for t in terminals:
        lines.append(
            f'    "T_{t}" [shape=box, style="filled,rounded", '
            f'fillcolor="#ffe9c2", label=<<b>\'{t}\'</b>>];'
        )
    lines.append("  }")

    # Productions
    lines.append("")
    lines.append("  // productions")
    for nt in nt_order:
        total = nt_counts.get(nt, 0)
        for rule_idx, prod in enumerate(grammar.rules[nt]):
            count = rule_counts.get((nt, rule_idx), 0)
            frac = (count / total * 100.0) if total > 0 else 0.0
            rule_id = f"r_{nt}_{rule_idx}"
            prod_str = " ".join(prod)
            lines.append(
                f'  "{rule_id}" [shape=box, style="filled,rounded", '
                f'fillcolor="#f3f3f3", fontsize=10, '
                f"label=<r{rule_idx}: <b>{count:,}</b> "
                f'({frac:.1f}%)<br/><font point-size="8">→ {prod_str}</font>>];'
            )
            lines.append(
                f'  "{nt}" -> "{rule_id}" [penwidth={max(0.5, min(6.0, frac / 15.0)):.2f}];'
            )
            for pos, sym in enumerate(prod, start=1):
                target = f"T_{sym}" if sym in terminals else sym
                lines.append(
                    f'  "{rule_id}" -> "{target}" '
                    f'[label="{pos}", fontsize=8, arrowsize=0.7, color="#888888"];'
                )

    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    grammar = CFGrammar.from_name(args.cfg)
    print(f"streaming traces from {args.traces}...", flush=True)
    n_trees, rule_counts, nt_counts = aggregate(args.traces, args.progress_seconds)
    print(f"aggregated {n_trees:,} trees, {sum(nt_counts.values()):,} NT expansions")
    dot = emit_dot(grammar, rule_counts, nt_counts, n_trees, args.traces)
    with open(args.output, "w") as f:
        f.write(dot)
    print(f"wrote {args.output} ({len(dot):,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
