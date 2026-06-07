"""Enumerate exact n-gram counts from a suffix-array index.

For each n in 1..max_n, exhaustively queries all V^n possible token
sequences over the vocab via SA binary search and records the count.
Designed for small vocabularies (cfg3b: V=5 → 5^6 = 15,625 queries for
n=6) where full enumeration is cheaper than the bookkeeping to enumerate
only the non-zero n-grams.

Writes two artifacts under <output_prefix>:

  <output_prefix>.npz   one int64 array per n, key 'n{k}', shape (V,)*k.
                        Dense — zeros mean "0 occurrences".
  <output_prefix>.csv   flat table: length, ngram_ids, ngram_sym, count.
                        Only non-zero rows.

The npz is the source of truth (everything else is derivable from it —
marginals, conditionals, prefix-restricted views). The CSV is for eyeball
inspection and sorting in a data wrangler.
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_infinigram import count

VOCAB_NAMES = {0: "1", 1: "2", 2: "3", 3: "eos", 4: "bos"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sa-dir", required=True, help="suffix-array directory (contains tokens.bin, suffix_array.bin, meta.txt)")
    p.add_argument("--max-n", type=int, default=6,
                   help="largest *suffix* length to enumerate (default 6); full pattern length = len(prefix) + n")
    p.add_argument("--vocab-size", type=int, default=5, help="alphabet size to enumerate over (default 5: terminals + bos + eos)")
    p.add_argument("--prefix", default="",
                   help="comma-separated token IDs to fix at the start of every pattern "
                        "(e.g. '4' = bos-rooted). Default: empty (unconstrained).")
    p.add_argument("--output-prefix", required=True, help="output path prefix (no extension; .npz and .csv appended)")
    return p.parse_args()


def load_meta(sa_dir):
    meta = {}
    with open(os.path.join(sa_dir, "meta.txt")) as f:
        for line in f:
            if not line.strip():
                continue
            k, v = line.strip().split(None, 1)
            meta[k] = v
    return meta


def load_sa(sa_dir):
    meta = load_meta(sa_dir)
    n_tokens = int(meta["n_tokens"])
    tokens_dtype = np.dtype(meta["tokens_dtype"])
    sa_dtype = np.dtype(meta["sa_dtype"])
    tokens = np.memmap(os.path.join(sa_dir, "tokens.bin"), dtype=tokens_dtype, mode="c", shape=(n_tokens,))
    sa = np.memmap(os.path.join(sa_dir, "suffix_array.bin"), dtype=sa_dtype, mode="c", shape=(n_tokens,))
    return tokens, sa, n_tokens


def main():
    args = parse_args()
    tokens, sa, n_tokens = load_sa(args.sa_dir)

    prefix_ids = [int(x) for x in args.prefix.split(",") if x.strip()]
    prefix_syms = " ".join(VOCAB_NAMES[i] for i in prefix_ids) if prefix_ids else "(none)"
    print(f"loaded SA: n_tokens={n_tokens:,}  vocab_size={args.vocab_size}  max_n={args.max_n}")
    print(f"prefix: ids={prefix_ids}  syms={prefix_syms}  "
          f"(full pattern length = {len(prefix_ids)} + n)")

    arrays = {}
    csv_rows = []

    prefix_arr = np.array(prefix_ids, dtype=np.uint8) if prefix_ids else None

    for n in range(1, args.max_n + 1):
        total = args.vocab_size ** n
        full_len = len(prefix_ids) + n
        print(f"\nenumerating {total:,} suffixes of length {n} "
              f"(full {full_len}-gram = prefix + suffix) via SA count()...", flush=True)
        counts = np.zeros((args.vocab_size,) * n, dtype=np.int64)

        pat_buf = np.empty(full_len, dtype=np.uint8)
        if prefix_arr is not None:
            pat_buf[: len(prefix_ids)] = prefix_arr

        t0 = time.time()
        for idx in itertools.product(range(args.vocab_size), repeat=n):
            pat_buf[len(prefix_ids):] = idx
            counts[idx] = count(tokens, sa, pat_buf)
        elapsed = time.time() - t0

        nonzero = int((counts > 0).sum())
        total_count = int(counts.sum())
        print(f"  {elapsed:.1f}s ({total / max(elapsed, 1e-9):.0f} q/s)  "
              f"nonzero={nonzero:,}/{total:,}  total_count={total_count:,}")

        arrays[f"n{n}"] = counts
        for idx in itertools.product(range(args.vocab_size), repeat=n):
            c = int(counts[idx])
            if c == 0:
                continue
            full_idx = tuple(prefix_ids) + idx
            csv_rows.append((
                full_len,
                ",".join(str(i) for i in full_idx),
                " ".join(VOCAB_NAMES[i] for i in full_idx),
                c,
            ))

    npz_path = args.output_prefix + ".npz"
    csv_path = args.output_prefix + ".csv"

    save_kwargs = dict(arrays)
    if prefix_ids:
        save_kwargs["prefix"] = np.array(prefix_ids, dtype=np.int64)
    np.savez(npz_path, **save_kwargs)
    print(f"\nwrote {npz_path} ({os.path.getsize(npz_path):,} bytes)")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "ngram_ids", "ngram_sym", "count"])
        w.writerows(csv_rows)
    print(f"wrote {csv_path} ({len(csv_rows):,} non-zero rows, "
          f"{os.path.getsize(csv_path):,} bytes)")


if __name__ == "__main__":
    main()
