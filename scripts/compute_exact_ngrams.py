"""CLI: compute exact BOS-rooted n-gram probabilities from a grammar.

Thin wrapper around ``cfg.analysis.exact``. No corpus or index needed —
this is the true distribution the sampler draws from, against which both
the empirical dataset n-grams (compute_ngrams.py) and a trained model's
distributions (extract_model_ngrams.py) can be compared.

Outputs (under <output-prefix>):

  <output-prefix>.npz   one float64 array per n, key 'n{k}', shape (V,)*k,
                        holding P(suffix | bos). Includes 'prefix' = [bos_id]
                        so the file is interchangeable with a
                        compute_ngrams.py --prefix 4 run.

Supports --mask-rule with the same 'NT:INDEX' syntax as
build_char_dataset.py so the exact distribution of a masked dataset is
one command away.
"""

import argparse
import os

import numpy as np

from cfg.analysis.exact import exact_bos_ngrams
from cfg.grammar.cfg_grammar import CFGrammar
from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer
from cfg.grammar.cfg_utils import build_mask_weights, parse_mask_rule


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cfg", default="cfg3b", help="grammar name from cfg_defines")
    p.add_argument("--max-n", type=int, default=6,
                   help="largest suffix length after bos to compute (default 6)")
    p.add_argument("--mask-rule", default=None,
                   help="zero one production rule's sampling weight, 'NT:INDEX' "
                        "(same syntax as build_char_dataset.py)")
    p.add_argument("--output-prefix", required=True,
                   help="output path prefix (no extension; .npz appended)")
    return p.parse_args()


def main():
    args = parse_args()
    grammar = CFGrammar.from_name(args.cfg)
    tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)

    weights = None
    if args.mask_rule is not None:
        nt, idx = parse_mask_rule(args.mask_rule, grammar.rules)
        print(f"masking NT {nt!r} rule index {idx}: {grammar.rules[nt][idx]}")
        weights = build_mask_weights(grammar.rules, nt, idx)

    print(f"grammar={args.cfg}  vocab_size={len(tokenizer)}  max_n={args.max_n}")
    arrays = exact_bos_ngrams(grammar, args.max_n, weights=weights,
                              tokenizer=tokenizer)

    # Report support sizes per level so a masked run's shrunken support
    # is visible at a glance next to the unmasked equivalent.
    for k in range(1, args.max_n + 1):
        arr = arrays[f"n{k}"]
        nonzero = int((arr > 0).sum())
        print(f"  n{k}: nonzero={nonzero:,}/{arr.size:,}  sum={arr.sum():.12f}")

    npz_path = args.output_prefix + ".npz"
    np.savez(npz_path, **arrays)
    print(f"wrote {npz_path} ({os.path.getsize(npz_path):,} bytes)")


if __name__ == "__main__":
    main()
