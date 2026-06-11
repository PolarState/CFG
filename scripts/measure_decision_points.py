"""CLI: probe a model's rule probabilities at real parse decision points.

Thin wrapper around ``cfg.analysis.decision_points``. Streams parse
traces (Schema B JSONL from dump_traces.py), teacher-forces each
sentence through a checkpoint, and reports P(rule yield | context) for
every rule of a terminal-level nonterminal at each position where the
trace says an instance begins.

The probe traces should come from the FULL grammar even when the model
was trained masked — that is the point: rows where true_rule is the
masked rule show exactly how much mass the model retains for decisions
it never saw in training.

Outputs (under <output-prefix>):

  <output-prefix>.csv   sentence, offset, true_rule, p_rule_0..k, share_0..k

Requires torch — run in the gpt2 env.
"""

import argparse
import csv
import json

import numpy as np

from cfg.analysis.decision_points import (
    measure_decision_points,
    summarize_decision_points,
)
from cfg.grammar.cfg_grammar import CFGrammar


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="path to a HF checkpoint directory")
    p.add_argument("--traces", required=True,
                   help="Schema B JSONL parse traces (one sentence per line)")
    p.add_argument("--cfg", default="cfg3b",
                   help="FULL grammar name — supplies the rule inventory")
    p.add_argument("--nt", required=True,
                   help="terminal-level nonterminal to probe (e.g. '7')")
    p.add_argument("--num-sentences", type=int, default=50,
                   help="sentences (containing the NT) to probe (default 50)")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--output-prefix", required=True,
                   help="output path prefix (.csv appended)")
    return p.parse_args()


def iter_traces(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    grammar = CFGrammar.from_name(args.cfg)
    n_rules = len(grammar.rules[args.nt])

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint} on {device}")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)

    print(f"probing NT {args.nt!r} ({n_rules} rules: "
          f"{grammar.rules[args.nt]}) over {args.num_sentences} sentences")
    rows = measure_decision_points(
        model, grammar, iter_traces(args.traces), args.nt,
        max_sentences=args.num_sentences, device=device,
    )

    csv_path = args.output_prefix + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["sentence", "offset", "true_rule"]
                  + [f"p_rule_{i}" for i in range(n_rules)]
                  + [f"share_{i}" for i in range(n_rules)])
        w.writerow(header)
        for r in rows:
            total = sum(r["p_rules"])
            shares = [p / total if total > 0 else 0.0 for p in r["p_rules"]]
            w.writerow([r["sentence"], r["offset"], r["true_rule"]]
                       + [f"{p:.6g}" for p in r["p_rules"]]
                       + [f"{s:.6g}" for s in shares])
    print(f"wrote {csv_path} ({len(rows)} decision points)")

    summary = summarize_decision_points(rows, n_rules)
    print(f"\nNT {args.nt}: mean P(yield_r | context), grouped by TRUE rule")
    for key in list(range(n_rules)) + ["all"]:
        s = summary[key]
        if s["n"] == 0:
            print(f"  true={key}: (no points)")
            continue
        p_str = "  ".join(f"P(r{i})={v:.4f}" for i, v in enumerate(s["mean_p"]))
        sh_str = "  ".join(f"{v:.3f}" for v in s["mean_share"])
        print(f"  true={key}: n={s['n']:>5}  {p_str}  shares=[{sh_str}]")


if __name__ == "__main__":
    main()
