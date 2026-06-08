"""CLI: enumerate exact n-gram counts from a suffix-array index.

Thin wrapper around ``cfg.analysis.ngrams``. Loads an SA via
``cfg.indexing.sa.load_index``, calls ``enumerate_ngrams``, saves the
resulting dense count tables as a .npz, and emits a CSV of non-zero
rows for eyeball inspection.

Outputs (under <output-prefix>):

  <output-prefix>.npz   one int64 array per n, key 'n{k}', shape (V,)*k.
                        Optional 'prefix' key when --prefix is set.
  <output-prefix>.csv   flat (length, ngram_ids, ngram_sym, count) for
                        non-zero rows.

The npz is the source of truth — marginals, conditionals, and
prefix-restricted views are all derivable from it.
"""

import argparse
import os

import numpy as np

from cfg.analysis.ngrams import dump_ngrams_to_csv, enumerate_ngrams
from cfg.indexing.sa import load_index

# cfg3b vocab labels — script-level concern, not the library's.
VOCAB_NAMES = {0: "1", 1: "2", 2: "3", 3: "eos", 4: "bos"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sa-dir", required=True,
                   help="suffix-array directory (tokens.bin, suffix_array.bin, meta.txt)")
    p.add_argument("--max-n", type=int, default=6,
                   help="largest *suffix* length to enumerate (default 6)")
    p.add_argument("--vocab-size", type=int, default=5,
                   help="alphabet size to enumerate over (default 5: terminals + bos + eos)")
    p.add_argument("--prefix", default="",
                   help="comma-separated token IDs to fix at the start of every pattern "
                        "(e.g. '4' = bos-rooted). Default: empty (unconstrained).")
    p.add_argument("--output-prefix", required=True,
                   help="output path prefix (no extension; .npz and .csv appended)")
    return p.parse_args()


def main():
    args = parse_args()
    tokens, sa, n_tokens = load_index(args.sa_dir)

    prefix_ids = [int(x) for x in args.prefix.split(",") if x.strip()]
    prefix_syms = " ".join(VOCAB_NAMES[i] for i in prefix_ids) if prefix_ids else "(none)"
    print(f"loaded SA: n_tokens={n_tokens:,}  vocab_size={args.vocab_size}  max_n={args.max_n}")
    print(f"prefix: ids={prefix_ids}  syms={prefix_syms}  "
          f"(full pattern length = {len(prefix_ids)} + n)")

    arrays = enumerate_ngrams(
        tokens, sa,
        max_n=args.max_n,
        vocab_size=args.vocab_size,
        prefix=prefix_ids,
    )

    npz_path = args.output_prefix + ".npz"
    csv_path = args.output_prefix + ".csv"

    np.savez(npz_path, **arrays)
    print(f"\nwrote {npz_path} ({os.path.getsize(npz_path):,} bytes)")

    n_rows, csv_size = dump_ngrams_to_csv(arrays, csv_path, vocab_names=VOCAB_NAMES)
    print(f"wrote {csv_path} ({n_rows:,} non-zero rows, {csv_size:,} bytes)")


if __name__ == "__main__":
    main()
