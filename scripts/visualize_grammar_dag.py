"""CLI: aggregate JSONL parse traces and render the grammar DAG.

Thin wrapper around ``cfg.analysis.visualization``. Streams a JSONL
trace file (from scripts/dump_traces.py), aggregates per-NT and per-rule
counts, and renders the production graph as a Graphviz DOT + SVG.

Usage:

    python scripts/visualize_grammar_dag.py \\
        --traces traces_cfg3b_val_seed1.jsonl \\
        --output-prefix grammar_dag_val_seed1
"""

import argparse

from cfg.analysis.visualization import aggregate_traces, render_grammar_dag
from cfg.grammar.cfg_grammar import CFGrammar


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--traces", required=True, help="input JSONL traces file")
    p.add_argument("--cfg", default="cfg3b", help="grammar name from cfg_defines")
    p.add_argument("--output-prefix", required=True,
                   help="output path prefix (no extension; .dot and .svg appended)")
    p.add_argument("--progress-seconds", type=float, default=10.0,
                   help="print progress every N seconds while streaming traces")
    return p.parse_args()


def main():
    args = parse_args()
    grammar = CFGrammar.from_name(args.cfg)
    print(f"streaming traces from {args.traces}...", flush=True)
    n_trees, rule_counts, nt_counts = aggregate_traces(args.traces, args.progress_seconds)
    print(f"aggregated {n_trees:,} trees, {sum(nt_counts.values()):,} NT expansions")

    dot_path, svg_path = render_grammar_dag(
        grammar, rule_counts, nt_counts, n_trees,
        output_prefix=args.output_prefix,
        source_label=args.traces,
        grammar_name=args.cfg,
    )
    print(f"wrote {dot_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
