"""Visualize n-gram counts produced by compute_ngrams.py.

Emits three views from a single .npz of n-gram counts:

  <output-prefix>_heatmap.{png,svg}  conditional P(next | prefix-n-gram),
                                     one row per non-zero prefix sorted
                                     descending by prefix count. Cells are
                                     annotated with the conditional prob
                                     above a threshold for readability.

  <output-prefix>_trie.{dot,svg}     trie of all non-zero n-grams from
                                     length 1 to max-depth. Edges labeled
                                     with the next-token added and its
                                     conditional probability; edge width
                                     scaled by log(count). Nodes show the
                                     prefix's total count.

  <output-prefix>_sunburst.html      interactive Plotly sunburst: ring k
                                     = k-grams, arc ∝ count, hover reveals
                                     full n-gram + count + P(child|prefix),
                                     click drills into a sub-tree. Much
                                     easier to grok at high depth than the
                                     trie.

Designed to pair with compute_ngrams.py output for small vocabularies
(cfg3b: V=5). Trie depth 6 has ~675 nodes — large, but Graphviz handles
it and the SVG scrolls cleanly in VS Code preview.
"""

import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

VOCAB_NAMES = {0: "1", 1: "2", 2: "3", 3: "E", 4: "B"}
VOCAB_LONG = {0: "'1'", 1: "'2'", 2: "'3'", 3: "eos", 4: "bos"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
                   help="sunburst color encoding: 'prob' = P(child|prefix) viridis (default, max dynamic range); "
                        "'logcount' = log10(count) magma; 'token' = saturated token-identity palette")
    p.add_argument("--sunburst-size", choices=["count", "equal"], default="count",
                   help="sunburst wedge angular-size encoding: 'count' (default) = corpus count of the n-gram; "
                        "'equal' = each parent splits its arc equally among its children (structure-only view, "
                        "all weight goes into the color channel)")
    p.add_argument("--annot-threshold", type=float, default=0.05,
                   help="annotate heatmap cells where P(next|prefix) > this (default 0.05)")
    return p.parse_args()


def render_heatmap(data, prefix_n, output_prefix, annot_threshold, fixed_prefix_syms=""):
    n_prefix = data[f"n{prefix_n}"]
    n_full = data[f"n{prefix_n + 1}"]
    V = n_prefix.shape[0]

    flat_prefix = n_prefix.ravel()
    flat_full = n_full.reshape(-1, V)

    nonzero = np.nonzero(flat_prefix)[0]
    counts = flat_prefix[nonzero]
    order = np.argsort(-counts)
    nonzero = nonzero[order]
    counts = counts[order]
    cond = flat_full[nonzero] / counts[:, None]

    row_prefix = f"{fixed_prefix_syms} " if fixed_prefix_syms else ""
    row_labels = []
    for fi, c in zip(nonzero, counts):
        idx = np.unravel_index(int(fi), (V,) * prefix_n)
        sym = " ".join(VOCAB_NAMES[i] for i in idx)
        row_labels.append(f"{row_prefix}{sym}  [{int(c):>11,}]")
    col_labels = [VOCAB_NAMES[i] for i in range(V)]

    n_rows = len(row_labels)
    fig_h = max(6.0, n_rows * 0.16)
    fig, ax = plt.subplots(figsize=(4.5, fig_h))
    im = ax.imshow(cond, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(V))
    ax.set_xticklabels(col_labels, fontfamily="monospace", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontfamily="monospace", fontsize=5.5)
    ax.set_xlabel("next token")
    full_prefix_len = (len(fixed_prefix_syms.split()) if fixed_prefix_syms else 0) + prefix_n
    ax.set_ylabel(f"{full_prefix_len}-gram prefix  (sorted desc by prefix count, count in brackets)")
    title_root = (
        f"P(next | [{fixed_prefix_syms}] + {prefix_n}-suffix)"
        if fixed_prefix_syms
        else f"P(next | {prefix_n}-gram prefix)"
    )
    ax.set_title(
        f"{title_root}\n"
        f"{n_rows:,} non-zero prefixes  ·  source: {os.path.basename(args.npz)}",
        fontsize=10,
    )

    for i in range(cond.shape[0]):
        for j in range(V):
            p = cond[i, j]
            if p > annot_threshold:
                color = "white" if p < 0.55 else "black"
                ax.text(j, i, f"{p:.2f}", ha="center", va="center",
                        fontsize=4.5, color=color)

    fig.colorbar(im, ax=ax, label="P(next | prefix)", shrink=0.4)
    fig.tight_layout()

    png_path = output_prefix + "_heatmap.png"
    svg_path = output_prefix + "_heatmap.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def render_trie(data, max_depth, output_prefix, fixed_prefix_syms=""):
    V = data["n1"].shape[0]
    total_tokens = int(data["n1"].sum())

    max_count = max(int(data[f"n{n}"].max()) for n in range(1, max_depth + 1))
    log_max = max(1.0, np.log10(max_count + 1))

    root_label = (
        f"ROOT\\n[{fixed_prefix_syms}]\\n{total_tokens:,} occurrences"
        if fixed_prefix_syms
        else f"ROOT\\n{total_tokens:,} tokens"
    )
    title_root = (
        f"n-gram trie rooted at [{fixed_prefix_syms}]  (suffix depth ≤ {max_depth})"
        if fixed_prefix_syms
        else f"n-gram trie (depth ≤ {max_depth})"
    )

    lines = [
        "digraph ngram_trie {",
        "  rankdir=LR;",
        '  graph [labelloc=t, fontname="Helvetica", ranksep=0.5, nodesep=0.05];',
        '  node [fontname="Helvetica", shape=box, style="rounded,filled", margin="0.04,0.02", fontsize=9];',
        '  edge [fontname="Helvetica", fontsize=7];',
        "",
        f'  graph [label=<<b>{title_root}</b>'
        f"<br/><i>nodes = prefixes; edge label = P(child | prefix); "
        f'edge width ∝ log(count)</i>>];',
        "",
        f'  "" [label="{root_label}", fillcolor="#dddddd", fontsize=11];',
    ]

    def color_for(tok):
        if tok < 3:
            return "#fff4d6"
        if tok == 3:
            return "#ffd6d6"
        return "#d6e6ff"

    for n in range(1, max_depth + 1):
        arr = data[f"n{n}"]
        flat = arr.ravel()
        parent_arr = data[f"n{n - 1}"] if n > 1 else None
        for fi in np.nonzero(flat)[0]:
            idx = tuple(int(x) for x in np.unravel_index(int(fi), arr.shape))
            count = int(flat[fi])
            node_id = ",".join(str(i) for i in idx)
            parent_id = ",".join(str(i) for i in idx[:-1])
            last_tok = idx[-1]
            sym = VOCAB_NAMES[last_tok]

            if n == 1:
                parent_count = total_tokens
            else:
                parent_count = int(parent_arr[idx[:-1]])
            prob = count / parent_count if parent_count > 0 else 0.0

            lines.append(
                f'  "{node_id}" [label="{sym}\\n{count:,}", fillcolor="{color_for(last_tok)}"];'
            )
            penwidth = max(0.4, min(5.0, np.log10(count + 1) / log_max * 5.0))
            lines.append(
                f'  "{parent_id}" -> "{node_id}" '
                f'[label="{prob:.3f}", penwidth={penwidth:.2f}];'
            )

    lines.append("}")
    dot_text = "\n".join(lines) + "\n"
    dot_path = output_prefix + "_trie.dot"
    svg_path = output_prefix + "_trie.svg"
    with open(dot_path, "w") as f:
        f.write(dot_text)
    subprocess.run(["dot", "-Tsvg", dot_path, "-o", svg_path], check=True)
    return dot_path, svg_path


def render_sunburst(data, max_depth, output_prefix, fixed_prefix_ids=(),
                    color_mode="prob", size_mode="count"):
    import plotly.graph_objects as go

    V = data["n1"].shape[0]
    total_tokens = int(data["n1"].sum())

    # Saturated token palette (used in color_mode="token"). The previous
    # pastel palette had ~0 dynamic range; these are the same hues but with
    # the saturation/value cranked up so adjacent wedges actually pop.
    def token_color(tok):
        if tok < 3:
            return "#e8a83a"  # terminals: deep amber
        if tok == 3:
            return "#c83838"  # eos: deep red
        return "#3a66c8"      # bos: deep blue

    fixed_syms = " ".join(VOCAB_NAMES[i] for i in fixed_prefix_ids)
    root_label = f"<b>{fixed_syms}</b>" if fixed_prefix_ids else "<b>ROOT</b>"
    root_hover = (
        f"<b>{fixed_syms}</b><br>{total_tokens:,} occurrences"
        if fixed_prefix_ids
        else f"<b>ROOT</b><br>{total_tokens:,} tokens"
    )

    ids = ["ROOT"]
    labels = [root_label]
    parents = [""]
    values = [total_tokens]
    hovers = [root_hover]
    # Collected once; color/colorscale dispatch happens after the walk.
    node_meta = []  # list of (last_tok, count, prob) per non-root node

    for n in range(1, max_depth + 1):
        arr = data[f"n{n}"]
        flat = arr.ravel()
        parent_arr = data[f"n{n - 1}"] if n > 1 else None
        for fi in np.nonzero(flat)[0]:
            idx = tuple(int(x) for x in np.unravel_index(int(fi), arr.shape))
            cnt = int(flat[fi])
            node_id = ",".join(str(i) for i in idx)
            parent_id = ",".join(str(i) for i in idx[:-1]) if n > 1 else "ROOT"
            last_tok = idx[-1]
            sym = VOCAB_NAMES[last_tok]
            full_idx = tuple(fixed_prefix_ids) + idx
            sym_full = " ".join(VOCAB_NAMES[i] for i in full_idx)

            if n == 1:
                parent_count = total_tokens
            else:
                parent_count = int(parent_arr[idx[:-1]])
            prob = cnt / parent_count if parent_count > 0 else 0.0
            frac_global = cnt / total_tokens if total_tokens > 0 else 0.0

            ids.append(node_id)
            labels.append(sym)
            parents.append(parent_id)
            values.append(cnt)
            hovers.append(
                f"<b>{sym_full}</b><br>"
                f"depth: {n}<br>"
                f"count: {cnt:,}<br>"
                f"P({sym} | prefix): {prob:.4f}<br>"
                f"P(n-gram): {frac_global:.4f}"
            )
            node_meta.append((last_tok, cnt, prob))

    if color_mode == "token":
        root_color = token_color(fixed_prefix_ids[-1]) if fixed_prefix_ids else "#888888"
        colors = [root_color] + [token_color(m[0]) for m in node_meta]
        marker = dict(colors=colors, line=dict(width=0.8, color="white"))
        legend_note = "amber=terminals · red=eos · blue=bos"
    elif color_mode == "prob":
        # Numeric `marker.colors` + colorscale → plotly auto-renders the colorbar.
        # Root assigned 1.0 (the conditioning event is always satisfied within
        # the chart's scope) so it sits at the bright end.
        cvals = [1.0] + [m[2] for m in node_meta]
        marker = dict(
            colors=cvals,
            colorscale="Viridis",
            cmin=0.0, cmax=1.0,
            showscale=True,
            colorbar=dict(title=dict(text="P(child<br>| prefix)", side="right"),
                          thickness=15, len=0.7, x=1.0),
            line=dict(width=0.5, color="#222"),
        )
        legend_note = "color = P(child | prefix)  (dark purple = rare, bright yellow = near-deterministic)"
    elif color_mode == "logcount":
        max_count = max((m[1] for m in node_meta), default=1)
        log_max = float(np.log10(max_count + 1))
        cvals = [log_max] + [float(np.log10(m[1] + 1)) for m in node_meta]
        marker = dict(
            colors=cvals,
            colorscale="Magma",
            cmin=0.0, cmax=log_max,
            showscale=True,
            colorbar=dict(title=dict(text="log10(count)", side="right"),
                          thickness=15, len=0.7, x=1.0),
            line=dict(width=0.5, color="#222"),
        )
        legend_note = f"color = log10(count)  (max={max_count:,})"
    else:
        raise ValueError(f"unknown sunburst color_mode={color_mode!r}")

    if size_mode == "equal":
        # Equal-leaf allocation: every leaf in the displayed tree gets the
        # same arc; each internal node's arc = number of leaves in its
        # subtree. branchvalues="total" is satisfied because for any node
        # leaf_count(node) = sum(leaf_count(child)) by definition.
        # A "leaf" here is a node with no children in the displayed tree —
        # i.e., either at max_depth or with no observed extensions in-corpus.
        children_by_parent = {}
        for i, par in enumerate(parents):
            if i == 0:
                continue
            children_by_parent.setdefault(par, []).append(i)

        leaf_count = [0] * len(ids)
        # Iterative post-order DFS so children are settled before parents.
        stack = [(0, False)]
        while stack:
            idx, done = stack.pop()
            kids = children_by_parent.get(ids[idx], [])
            if done:
                leaf_count[idx] = sum(leaf_count[k] for k in kids) if kids else 1
            else:
                stack.append((idx, True))
                for k in kids:
                    stack.append((k, False))
        values = leaf_count
        size_legend = "arc ∝ leaf count in subtree (every leaf wedge same width)"
    else:
        size_legend = "arc ∝ count"

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=marker,
        hovertext=hovers,
        hovertemplate="%{hovertext}<extra></extra>",
        insidetextorientation="radial",
        textfont=dict(size=13, family="monospace"),
        sort=False,
    ))
    title_root = (
        f"n-gram sunburst rooted at [{fixed_syms}]  (suffix depth ≤ {max_depth})"
        if fixed_prefix_ids
        else f"n-gram sunburst (depth ≤ {max_depth})"
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title_root}</b><br>"
                f"<sub>ring k = k-grams · {size_legend} · {legend_note} · "
                f"hover for details · click to zoom</sub>"
            ),
            x=0.5, xanchor="center",
        ),
        margin=dict(t=80, l=10, r=80, b=10),
        width=1100, height=1000,
    )

    html_path = output_prefix + "_sunburst.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


if __name__ == "__main__":
    args = parse_args()
    data = np.load(args.npz)
    print(f"loaded {args.npz}  keys={list(data.keys())}")

    if "prefix" in data.files:
        fixed_prefix_ids = tuple(int(x) for x in data["prefix"].tolist())
        fixed_prefix_syms = " ".join(VOCAB_NAMES[i] for i in fixed_prefix_ids)
        print(f"prefix: ids={list(fixed_prefix_ids)}  syms=[{fixed_prefix_syms}]")
    else:
        fixed_prefix_ids = ()
        fixed_prefix_syms = ""

    print(f"\nrendering heatmap (prefix_n={args.prefix_n})...")
    png, svg = render_heatmap(data, args.prefix_n, args.output_prefix, args.annot_threshold,
                              fixed_prefix_syms=fixed_prefix_syms)
    print(f"  wrote {png}  ({os.path.getsize(png):,} bytes)")
    print(f"  wrote {svg}  ({os.path.getsize(svg):,} bytes)")

    print(f"\nrendering trie (max_depth={args.trie_depth})...")
    dot, svg = render_trie(data, args.trie_depth, args.output_prefix,
                           fixed_prefix_syms=fixed_prefix_syms)
    print(f"  wrote {dot}  ({os.path.getsize(dot):,} bytes)")
    print(f"  wrote {svg}  ({os.path.getsize(svg):,} bytes)")

    print(f"\nrendering sunburst (max_depth={args.sunburst_depth}, "
          f"color={args.sunburst_color}, size={args.sunburst_size})...")
    html = render_sunburst(data, args.sunburst_depth, args.output_prefix,
                           fixed_prefix_ids=fixed_prefix_ids,
                           color_mode=args.sunburst_color,
                           size_mode=args.sunburst_size)
    print(f"  wrote {html}  ({os.path.getsize(html):,} bytes)")
