"""CFG visualizations: n-gram views and grammar DAG.

Two families of renderers:

  n-gram views (heatmap, trie, sunburst) take a dict of n-gram count
  arrays (the output of ``cfg.analysis.ngrams.enumerate_ngrams`` or a
  ``np.load()`` of its .npz output) and produce on-disk artifacts.

  grammar DAG takes aggregated (NT, rule)-choice counts from parse-tree
  traces and renders the production graph rolled up by NT.

None of the functions mutate their inputs.

n-gram arrays dict structure:

  n1, n2, ..., n{max_n}     dense int64 arrays, shape (V,)*k
  prefix (optional)         int64 1-d array of fixed prefix IDs

A ``vocab_names`` mapping (id → display string) is passed per-render so
the library is vocab-agnostic for label display. The wedge-color
heuristics in the sunburst/trie currently assume the cfg3b layout
(ids 0..n_terminals-1 = terminals, n_terminals = eos, n_terminals+1 =
bos) — pass ``n_terminals`` to adapt.

API:

  render_heatmap(arrays, output_prefix, prefix_n=5, annot_threshold=0.05,
                 vocab_names=None, source_label="")     -> (png_path, svg_path)
  render_trie(arrays, output_prefix, max_depth=6,
              vocab_names=None, n_terminals=3)          -> (dot_path, svg_path)
  render_sunburst(arrays, output_prefix, max_depth=6,
                  vocab_names=None, n_terminals=3,
                  color_mode="prob", size_mode="count") -> html_path
  aggregate_traces(traces_path, progress_seconds=10.0)
                                                       -> (n_trees, rule_counts, nt_counts)
  render_grammar_dag(grammar, rule_counts, nt_counts, n_trees,
                     output_prefix, source_label="", grammar_name="")
                                                       -> (dot_path, svg_path)
"""

import json
import os
import subprocess
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def _sym_of(vocab_names, i):
    return vocab_names[i] if vocab_names is not None else str(i)


def _prefix_info(arrays, vocab_names):
    """Return ``(prefix_ids_tuple, prefix_syms_str)`` from arrays['prefix']."""
    if "prefix" in arrays:
        ids = tuple(int(x) for x in arrays["prefix"].tolist())
        syms = " ".join(_sym_of(vocab_names, i) for i in ids)
        return ids, syms
    return (), ""


# ── Heatmap ───────────────────────────────────────────────────────────────


def render_heatmap(arrays, output_prefix, prefix_n=5, annot_threshold=0.05,
                   vocab_names=None, source_label=""):
    """Render P(next | prefix-n-gram) as a heatmap.

    Rows: non-zero prefix n-grams, sorted descending by prefix count.
    Cols: next-token. Cells above ``annot_threshold`` are annotated
    with the conditional probability for readability.
    """
    n_prefix = arrays[f"n{prefix_n}"]
    n_full = arrays[f"n{prefix_n + 1}"]
    V = n_prefix.shape[0]

    fixed_prefix_ids, fixed_prefix_syms = _prefix_info(arrays, vocab_names)

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
        sym = " ".join(_sym_of(vocab_names, i) for i in idx)
        row_labels.append(f"{row_prefix}{sym}  [{int(c):>11,}]")
    col_labels = [_sym_of(vocab_names, i) for i in range(V)]

    n_rows = len(row_labels)
    fig_h = max(6.0, n_rows * 0.16)
    fig, ax = plt.subplots(figsize=(4.5, fig_h))
    im = ax.imshow(cond, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(V))
    ax.set_xticklabels(col_labels, fontfamily="monospace", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontfamily="monospace", fontsize=5.5)
    ax.set_xlabel("next token")
    full_prefix_len = len(fixed_prefix_ids) + prefix_n
    ax.set_ylabel(f"{full_prefix_len}-gram prefix  (sorted desc by prefix count, count in brackets)")
    title_root = (
        f"P(next | [{fixed_prefix_syms}] + {prefix_n}-suffix)"
        if fixed_prefix_syms
        else f"P(next | {prefix_n}-gram prefix)"
    )
    title_extra = f"  ·  source: {source_label}" if source_label else ""
    ax.set_title(
        f"{title_root}\n{n_rows:,} non-zero prefixes{title_extra}",
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


# ── Trie ──────────────────────────────────────────────────────────────────


def _pastel_color(tok, n_terminals):
    """Pastel palette for trie nodes (cfg3b layout assumed)."""
    if tok < n_terminals:
        return "#fff4d6"  # terminals
    if tok == n_terminals:
        return "#ffd6d6"  # eos
    return "#d6e6ff"      # bos


def render_trie(arrays, output_prefix, max_depth=6,
                vocab_names=None, n_terminals=3):
    """Render the n-gram trie as a Graphviz DOT file + rendered SVG.

    Each node = a prefix. Edge label = P(child | prefix). Edge width
    scales with log(count). Node color is by terminal/eos/bos.
    """
    V = arrays["n1"].shape[0]
    total_tokens = int(arrays["n1"].sum())
    fixed_prefix_ids, fixed_prefix_syms = _prefix_info(arrays, vocab_names)

    max_count = max(int(arrays[f"n{n}"].max()) for n in range(1, max_depth + 1))
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

    for n in range(1, max_depth + 1):
        arr = arrays[f"n{n}"]
        flat = arr.ravel()
        parent_arr = arrays[f"n{n - 1}"] if n > 1 else None
        for fi in np.nonzero(flat)[0]:
            idx = tuple(int(x) for x in np.unravel_index(int(fi), arr.shape))
            cnt = int(flat[fi])
            node_id = ",".join(str(i) for i in idx)
            parent_id = ",".join(str(i) for i in idx[:-1])
            last_tok = idx[-1]
            sym = _sym_of(vocab_names, last_tok)

            if n == 1:
                parent_count = total_tokens
            else:
                parent_count = int(parent_arr[idx[:-1]])
            prob = cnt / parent_count if parent_count > 0 else 0.0

            lines.append(
                f'  "{node_id}" [label="{sym}\\n{cnt:,}", '
                f'fillcolor="{_pastel_color(last_tok, n_terminals)}"];'
            )
            penwidth = max(0.4, min(5.0, np.log10(cnt + 1) / log_max * 5.0))
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


# ── Sunburst ──────────────────────────────────────────────────────────────


def _saturated_color(tok, n_terminals):
    """Saturated palette for sunburst nodes when color_mode='token'."""
    if tok < n_terminals:
        return "#e8a83a"  # terminals: deep amber
    if tok == n_terminals:
        return "#c83838"  # eos: deep red
    return "#3a66c8"      # bos: deep blue


def render_sunburst(arrays, output_prefix, max_depth=6,
                    vocab_names=None, n_terminals=3,
                    color_mode="prob", size_mode="count"):
    """Render an interactive Plotly sunburst of the n-gram tree.

    color_mode:
      'prob'     — viridis on P(child | prefix). High dynamic range.
      'logcount' — magma on log10(count).
      'token'    — saturated by token identity (terminal/eos/bos).
    size_mode:
      'count'    — arc ∝ corpus count of the n-gram.
      'equal'    — every leaf gets equal arc; inner = sum of descendant leaves.
    """
    import plotly.graph_objects as go

    V = arrays["n1"].shape[0]
    total_tokens = int(arrays["n1"].sum())
    fixed_prefix_ids, fixed_syms = _prefix_info(arrays, vocab_names)

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
    node_meta = []  # (last_tok, count, prob) per non-root node

    for n in range(1, max_depth + 1):
        arr = arrays[f"n{n}"]
        flat = arr.ravel()
        parent_arr = arrays[f"n{n - 1}"] if n > 1 else None
        for fi in np.nonzero(flat)[0]:
            idx = tuple(int(x) for x in np.unravel_index(int(fi), arr.shape))
            cnt = int(flat[fi])
            node_id = ",".join(str(i) for i in idx)
            parent_id = ",".join(str(i) for i in idx[:-1]) if n > 1 else "ROOT"
            last_tok = idx[-1]
            sym = _sym_of(vocab_names, last_tok)
            full_idx = tuple(fixed_prefix_ids) + idx
            sym_full = " ".join(_sym_of(vocab_names, i) for i in full_idx)

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
        root_color = (
            _saturated_color(fixed_prefix_ids[-1], n_terminals)
            if fixed_prefix_ids else "#888888"
        )
        colors = [root_color] + [_saturated_color(m[0], n_terminals) for m in node_meta]
        marker = dict(colors=colors, line=dict(width=0.8, color="white"))
        legend_note = "amber=terminals · red=eos · blue=bos"
    elif color_mode == "prob":
        cvals = [1.0] + [m[2] for m in node_meta]
        marker = dict(
            colors=cvals, colorscale="Viridis", cmin=0.0, cmax=1.0, showscale=True,
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
            colors=cvals, colorscale="Magma", cmin=0.0, cmax=log_max, showscale=True,
            colorbar=dict(title=dict(text="log10(count)", side="right"),
                          thickness=15, len=0.7, x=1.0),
            line=dict(width=0.5, color="#222"),
        )
        legend_note = f"color = log10(count)  (max={max_count:,})"
    else:
        raise ValueError(f"unknown sunburst color_mode={color_mode!r}")

    if size_mode == "equal":
        children_by_parent = {}
        for i, par in enumerate(parents):
            if i == 0:
                continue
            children_by_parent.setdefault(par, []).append(i)

        leaf_count = [0] * len(ids)
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
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total", marker=marker,
        hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
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


# ── Grammar DAG ───────────────────────────────────────────────────────────


def _walk_tree(node, rule_counts, nt_counts):
    """Recursive tree walk: increment (nt, rule) and nt counts."""
    if "t" in node:
        return
    nt = node["nt"]
    rule_counts[(nt, node["rule"])] += 1
    nt_counts[nt] += 1
    for child in node["children"]:
        _walk_tree(child, rule_counts, nt_counts)


def aggregate_traces(traces_path, progress_seconds=10.0):
    """Stream a JSONL parse-trace file and aggregate counts.

    Each line of ``traces_path`` is one parse tree (Schema B); see
    ``scripts/dump_traces.py``. Returns ``(n_trees, rule_counts, nt_counts)``
    where ``rule_counts`` is a ``Counter`` keyed by ``(nt, rule_idx)``
    and ``nt_counts`` is keyed by ``nt``.

    Prints a one-line progress update every ``progress_seconds`` seconds.
    """
    rule_counts = Counter()
    nt_counts = Counter()
    n_trees = 0
    t0 = time.time()
    last = t0
    with open(traces_path) as f:
        for line in f:
            _walk_tree(json.loads(line), rule_counts, nt_counts)
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


def render_grammar_dag(grammar, rule_counts, nt_counts, n_trees,
                       output_prefix, source_label="", grammar_name=""):
    """Render the grammar DAG as DOT + SVG.

    One node per NT, one node per terminal, one "rule" midpoint per
    production. NT → rule edges show count + frequency-among-this-NT;
    rule → child edges show child position. The "DAG" framing means
    counts roll up by NT: every expansion of (say) NT 7 is pooled into
    the same outgoing edges regardless of where in the tree it appeared.

    Returns ``(dot_path, svg_path)``.
    """
    nt_order = sorted(grammar.rules.keys(), key=int)
    terminals = list(grammar.terminal_symbols)

    title_subj = f"{grammar_name} grammar DAG" if grammar_name else "grammar DAG"
    title_extra = f" from {source_label}" if source_label else ""

    lines = [
        "digraph grammar_dag {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica", labelloc="t", '
        f'label=<<b>{title_subj}</b><br/>'
        f"<i>aggregated over {n_trees:,} trees{title_extra}</i>>];",
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=9];',
        "",
        "  // nonterminals",
    ]

    for nt in nt_order:
        n = nt_counts.get(nt, 0)
        lines.append(
            f'  "{nt}" [shape=ellipse, style=filled, fillcolor="#dbe9ff", '
            f'label=<<b>NT {nt}</b><br/><font point-size="9">{n:,} expansions</font>>];'
        )

    lines.append("")
    lines.append("  // terminals")
    lines.append("  { rank=sink;")
    for t in terminals:
        lines.append(
            f'    "T_{t}" [shape=box, style="filled,rounded", '
            f'fillcolor="#ffe9c2", label=<<b>\'{t}\'</b>>];'
        )
    lines.append("  }")

    lines.append("")
    lines.append("  // productions")
    for nt in nt_order:
        total = nt_counts.get(nt, 0)
        for rule_idx, prod in enumerate(grammar.rules[nt]):
            cnt = rule_counts.get((nt, rule_idx), 0)
            frac = (cnt / total * 100.0) if total > 0 else 0.0
            rule_id = f"r_{nt}_{rule_idx}"
            prod_str = " ".join(prod)
            lines.append(
                f'  "{rule_id}" [shape=box, style="filled,rounded", '
                f'fillcolor="#f3f3f3", fontsize=10, '
                f"label=<r{rule_idx}: <b>{cnt:,}</b> "
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
    dot_text = "\n".join(lines) + "\n"

    dot_path = output_prefix + ".dot"
    svg_path = output_prefix + ".svg"
    with open(dot_path, "w") as f:
        f.write(dot_text)
    subprocess.run(["dot", "-Tsvg", dot_path, "-o", svg_path], check=True)
    return dot_path, svg_path
