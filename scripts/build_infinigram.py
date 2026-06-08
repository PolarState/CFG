"""CLI: build a suffix-array infinigram from a CFGFileDataset .bin file.

Thin wrapper around ``cfg.indexing.sa``. The library handles convert /
build / save / count; this script just parses args, drives the pipeline,
and prints sanity-check queries.

Usage:
    python scripts/build_infinigram.py --dataset ../datasets/cfg3b_val_dataset_seed1.bin
    python scripts/build_infinigram.py --dataset ../datasets/cfg3b_train_dataset_seed0.bin
"""

import argparse
import os
import shutil
import time

import numpy as np

from cfg.indexing import sa as sa_lib


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Path to .bin dataset")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: analysis/sa_<dataset_stem>/)")
    p.add_argument("--window-length", type=int, default=512)
    p.add_argument("--max-windows", type=int, default=None,
                   help="Optionally cap number of 512-token windows read")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output_dir = os.path.join("analysis", f"sa_{stem}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading dataset: {args.dataset}")
    t0 = time.time()
    tokens = sa_lib.convert_dataset(args.dataset, args.window_length, args.max_windows)
    t_read = time.time() - t0
    unique = sorted(set(tokens.tolist())) if len(tokens) < 1_000_000 else sorted(np.unique(tokens).tolist())
    print(f"  Read {len(tokens):,} tokens in {t_read:.1f}s (unique IDs: {unique})\n")

    t0 = time.time()
    sa = sa_lib.build_sa(tokens)
    t_build = time.time() - t0
    print(f"  SA built in {t_build:.1f}s (dtype={sa.dtype}, len={len(sa):,})\n")

    print("Saving index to disk...")
    t0 = time.time()
    tokens_path, sa_path, _ = sa_lib.save_index(args.output_dir, tokens, sa)
    t_save = time.time() - t0
    tok_mb = os.path.getsize(tokens_path) / 1e6
    sa_mb = os.path.getsize(sa_path) / 1e6
    print(f"  Tokens: {tok_mb:.1f} MB  SA: {sa_mb:.1f} MB  Total: {tok_mb + sa_mb:.1f} MB")
    print(f"  Save: {t_save:.1f}s\n")

    # Sanity check — cfg3b terminals (0='1', 1='2', 2='3'):
    print("Sanity check — count queries:")
    test_seqs = [
        ([0],         "'1'"),
        ([1],         "'2'"),
        ([2],         "'3'"),
        ([0, 1, 2],   "'123'"),
        ([2, 0, 1],   "'312'"),
    ]
    for seq, label in test_seqs:
        pat = np.asarray(seq, dtype=tokens.dtype)
        c = sa_lib.count(tokens, sa, pat)
        print(f"  {label:>8s} {seq}: count = {c:,}")

    print("\nNext-token distributions (P(next | context) over terminals 0,1,2):")
    vocab = [0, 1, 2]
    test_contexts = [
        ([], "empty"),
        ([0], "'1'"),
        ([0, 1], "'12'"),
        ([0, 1, 2], "'123'"),
        ([0, 1, 2, 0, 1], "'12312'"),
        ([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], "'1231231231'"),
    ]
    for ctx, label in test_contexts:
        ctx_count, dist = sa_lib.next_token_distribution(tokens, sa, ctx, vocab)
        parts = " ".join(f"{tok}={prob:.4f}({cnt:,})" for tok, cnt, prob in dist)
        print(f"  ctx={label:<14s} count={ctx_count:>12,}  {parts}")

    total = t_read + t_build + t_save
    print(f"\nDone. Total: {total:.1f}s. Index at: {args.output_dir}/")


if __name__ == "__main__":
    main()
