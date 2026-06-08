"""CLI: visualize n-gram counts produced by compute_ngrams.py.

Thin wrapper around ``cfg.analysis.visualization``. Loads a .npz produced
by compute_ngrams.py and emits three views from it:

  <output-prefix>_heatmap.{png,svg}   P(next | prefix-n-gram)
  <output-prefix>_trie.{dot,svg}      Graphviz trie of non-zero n-grams
  <output-prefix>_sunburst.html       interactive Plotly sunburst

The trie depth 6 has ~675 nodes on cfg3b train — large, but Graphviz
handles it and the SVG scrolls cleanly in VS Code preview. The sunburst
is the easiest of the three to grok at high depth.
"""

import argparse
import os

import numpy as np

from cfg.analysis.visualization import (
    render_heatmap, render_sunburst, render_trie,
)

VOCAB_NAMES = {0: "1", 1: "2", 2: "3", 3: "E", 4: "B"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--npz", required=True, help="input .npz from compute_ngrams.py")
    p.add_argument("--output-prefix", required=True, help="output path prefix (no extension)")
    p.add_argument("--prefix-n", type=int, default=5,
                   help="prefix length for heatmap rows; columns are next-token (default 5)")
    p.add_argument("--trie-depth", type=int, default=6,
                   help="max depth for trie (default 6)")
    p.add_argument("--sunburst-depth", type=int, default=6,
                   help="max depth for sunburst (default 6)")
    p.add_argument("--sunburst-color", choices=["prob", "logcount", "token"], default="prob",
                   help="sunburst color encoding (default 'prob' = P(child|prefix) viridis)")
    p.add_argument("--sunburst-size", choices=["count", "equal"], default="count",
                   help="sunburst wedge angular-size encoding (default 'count')")
    p.add_argument("--annot-threshold", type=float, default=0.05,
                   help="annotate heatmap cells where P(next|prefix) > this (default 0.05)")
    return p.parse_args()


def main():
    args = parse_args()
    arrays = np.load(args.npz)
    print(f"loaded {args.npz}  keys={list(arrays.keys())}")

    if "prefix" in arrays.files:
        prefix_ids = [int(x) for x in arrays["prefix"].tolist()]
        prefix_syms = " ".join(VOCAB_NAMES[i] for i in prefix_ids)
        print(f"prefix: ids={prefix_ids}  syms=[{prefix_syms}]")

    print(f"\nrendering heatmap (prefix_n={args.prefix_n})...")
    png, svg = render_heatmap(
        arrays, args.output_prefix,
        prefix_n=args.prefix_n,
        annot_threshold=args.annot_threshold,
        vocab_names=VOCAB_NAMES,
        source_label=os.path.basename(args.npz),
    )
    print(f"  wrote {png}  ({os.path.getsize(png):,} bytes)")
    print(f"  wrote {svg}  ({os.path.getsize(svg):,} bytes)")

    print(f"\nrendering trie (max_depth={args.trie_depth})...")
    dot, svg = render_trie(
        arrays, args.output_prefix,
        max_depth=args.trie_depth,
        vocab_names=VOCAB_NAMES,
    )
    print(f"  wrote {dot}  ({os.path.getsize(dot):,} bytes)")
    print(f"  wrote {svg}  ({os.path.getsize(svg):,} bytes)")

    print(f"\nrendering sunburst (max_depth={args.sunburst_depth}, "
          f"color={args.sunburst_color}, size={args.sunburst_size})...")
    html = render_sunburst(
        arrays, args.output_prefix,
        max_depth=args.sunburst_depth,
        vocab_names=VOCAB_NAMES,
        color_mode=args.sunburst_color,
        size_mode=args.sunburst_size,
    )
    print(f"  wrote {html}  ({os.path.getsize(html):,} bytes)")


if __name__ == "__main__":
    main()
