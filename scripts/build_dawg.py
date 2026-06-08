"""CLI: build a disk-backed CDAWG from a pre-tokenized CFG .bin dataset.

Thin wrapper around ``cfg.indexing.dawg``. The library handles convert /
build / fill-counts with progress monitoring; this script just parses
args, cleans stale artifacts, drives the pipeline, and prints
sanity-check queries.

Usage:
    # Validation set (~5M tokens, fast):
    python scripts/build_dawg.py --dataset ../datasets/cfg3b_val_dataset_seed1.bin

    # Full training set (~4.9B tokens):
    python scripts/build_dawg.py --dataset ../datasets/cfg3b_train_dataset_seed0.bin

    # Subset:
    python scripts/build_dawg.py --dataset ../datasets/cfg3b_train_dataset_seed0.bin --max-windows 1000000
"""

import argparse
import os
import shutil
import time

from cfg.indexing import dawg as dawg_lib


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Path to the .bin dataset file.")
    p.add_argument("--output-dir", default=None,
                   help="CDAWG output directory (default: analysis/cdawg_<dataset_stem>/).")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Limit number of 512-token windows to read.")
    p.add_argument("--window-length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output_dir = os.path.join("analysis", f"cdawg_{stem}")

    os.makedirs(args.output_dir, exist_ok=True)
    tokens_path = os.path.join(args.output_dir, "tokens.diskvec")
    graph_dir = os.path.join(args.output_dir, "graph")

    # DiskVec/graph_dir creation in Rust fails if files already exist.
    if os.path.exists(tokens_path):
        os.remove(tokens_path)
    if os.path.exists(graph_dir):
        shutil.rmtree(graph_dir)

    print(f"Reading dataset: {args.dataset}")
    t0 = time.time()
    n_tokens = dawg_lib.convert_dataset(
        args.dataset, tokens_path,
        window_length=args.window_length, max_windows=args.max_windows,
    )
    t_read = time.time() - t0
    print(f"Conversion complete in {t_read:.1f}s\n")

    t0 = time.time()
    cdawg = dawg_lib.build_cdawg(tokens_path, graph_dir, n_tokens)
    t_build = time.time() - t0
    print(f"Build complete in {t_build:.1f}s")
    print(f"  Nodes: {cdawg.node_count():,}")
    print(f"  Edges: {cdawg.edge_count():,}")

    print("\nFilling counts...")
    t0 = time.time()
    dawg_lib.fill_counts(cdawg, graph_dir)
    t_counts = time.time() - t0
    print(f"Counts filled in {t_counts:.1f}s")

    # Sanity check — cfg3b terminals (0='1', 1='2', 2='3'):
    print("\nSanity check — querying short sequences:")
    test_sequences = [
        ([0],         "'1'"),
        ([1],         "'2'"),
        ([2],         "'3'"),
        ([0, 1, 2],   "'123'"),
        ([2, 0, 1],   "'312'"),
    ]
    for seq, label in test_sequences:
        state = cdawg.get_initial()
        for token in seq:
            state = cdawg.transition_and_count(state, token)
        count = cdawg.get_suffix_count(state)
        print(f"  {label:>8s} {seq}: count = {count:,}")

    total_time = t_read + t_build + t_counts
    print(f"\nDone. Total time: {total_time:.1f}s")
    print(f"CDAWG saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
