"""Disk-backed CDAWG index over a tokenized corpus.

A Compacted Directed Acyclic Word Graph (CDAWG) is a space-efficient
index over every substring of a corpus. Given a query string, the CDAWG
reports in O(|query|) time:

  - whether the query appears in the corpus,
  - how many times it appears (after fill_counts), and
  - the longest suffix of any prefix that matches the corpus
    (the "non-novel suffix length" from the Rusty-DAWG paper).

We use the DiskCdawg variant from rusty-dawg, which memory-maps the
graph node/edge arrays to disk. The OS page cache decides what stays
resident, so RAM usage stays bounded regardless of corpus size.

On-disk layout under <output_dir>/:

  tokens.diskvec       flat DiskVec<u16> file (bincode-serialized u16s)
  graph/nodes.vec      memory-mapped node array
  graph/edges.vec      memory-mapped edge array

API:

  convert_dataset(dataset_path, output_path, window_length=512, max_windows=None) -> n_tokens
  build_cdawg(tokens_path, graph_dir, n_tokens, show_progress=True)               -> DiskCdawg
  fill_counts(cdawg, graph_dir, show_progress=True)
"""

import multiprocessing
import os
import sys
import time

import numpy as np
from rusty_dawg import DiskCdawg


# ── Progress monitoring helpers ───────────────────────────────────────────


def _fmt_time(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def _monitor_sparse_progress(graph_dir, stop_event, label="Building"):
    """Monitor sparse-file block growth in a child process to show build progress.

    DiskCdawg pre-allocates large files (nodes.vec, edges.vec) via mmap.
    These start as sparse files — apparent size = full capacity, but
    actual disk blocks are only allocated as the Rust build loop writes.
    Comparing st_blocks (real disk blocks) to st_size (apparent size)
    gives a rough progress fraction.

    Runs in a separate process because cdawg.build() / fill_counts_ram()
    hold the Python GIL for their entire duration — no in-process polling
    would get scheduled.
    """
    nodes_path = os.path.join(graph_dir, "nodes.vec")
    edges_path = os.path.join(graph_dir, "edges.vec")

    while not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
        if stop_event.is_set():
            return
        time.sleep(0.2)

    nodes_cap = os.stat(nodes_path).st_size // 512
    edges_cap = os.stat(edges_path).st_size // 512
    total_cap = nodes_cap + edges_cap
    if total_cap == 0:
        return

    t0 = time.time()
    bar_width = 40

    while not stop_event.is_set():
        try:
            nodes_blocks = os.stat(nodes_path).st_blocks
            edges_blocks = os.stat(edges_path).st_blocks
        except OSError:
            break
        cur_blocks = nodes_blocks + edges_blocks

        frac = cur_blocks / total_cap
        pct = 100.0 * frac
        elapsed = time.time() - t0
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)

        if cur_blocks > 0 and elapsed > 0:
            rate = cur_blocks / elapsed
            remaining = total_cap - cur_blocks
            eta = remaining / rate if rate > 0 else 0
            eta_str = _fmt_time(eta)
        else:
            eta_str = "??:??"

        sys.stderr.write(
            f"\r  {label}: [{bar}] {pct:5.1f}%  "
            f"elapsed {_fmt_time(elapsed)}  eta {eta_str}   "
        )
        sys.stderr.flush()
        stop_event.wait(5)

    sys.stderr.write("\n")
    sys.stderr.flush()


def _run_with_progress(work_fn, graph_dir, label):
    """Run a GIL-holding work_fn while a child-process progress bar updates."""
    stop_evt = multiprocessing.Event()
    monitor = multiprocessing.Process(
        target=_monitor_sparse_progress,
        args=(graph_dir, stop_evt, label),
        daemon=True,
    )
    monitor.start()
    try:
        work_fn()
    finally:
        stop_evt.set()
        monitor.join(timeout=5)


# ── Library API ───────────────────────────────────────────────────────────


def convert_dataset(dataset_path, output_path, window_length=512, max_windows=None):
    """Convert a CFG .bin dataset to a DiskVec<u16> file for rusty-dawg.

    The CFG .bin files store token IDs as big-endian int32 in fixed-size
    windows (see cfg.grammar.cfg_datasets.CFGFileDataset). Rusty-dawg's
    DiskVec<u16> expects flat little-endian uint16 values. All cfg3b
    token IDs are ≤ 91, so the int32 → uint16 downcast is lossless.

    Appends ``DiskCdawg.EOS`` (u16::MAX = 65535) at the very end as the
    corpus-boundary sentinel.

    Returns the total number of tokens written (including the EOS).
    """
    file_size = os.path.getsize(dataset_path)
    total_ints = file_size // 4
    total_windows = total_ints // window_length

    if max_windows is not None:
        n_windows = min(max_windows, total_windows)
    else:
        n_windows = total_windows

    n_tokens = n_windows * window_length
    print(f"  File has {total_windows:,} windows; reading {n_windows:,} ({n_tokens:,} tokens)")

    chunk_size = 50_000_000
    if n_tokens <= chunk_size:
        raw = np.fromfile(dataset_path, dtype=">i4", count=n_tokens)
        tokens = raw.astype(np.dtype("<u2"))
        del raw
    else:
        tokens = np.empty(n_tokens, dtype="<u2")
        n_chunks = (n_tokens + chunk_size - 1) // chunk_size
        bar_width = 40
        with open(dataset_path, "rb") as f:
            for i in range(n_chunks):
                offset = i * chunk_size
                count_n = min(chunk_size, n_tokens - offset)
                chunk = np.frombuffer(f.read(count_n * 4), dtype=">i4")
                tokens[offset:offset + count_n] = chunk.astype(np.dtype("<u2"))

                frac = (offset + count_n) / n_tokens
                filled = int(bar_width * frac)
                bar = "█" * filled + "░" * (bar_width - filled)
                sys.stderr.write(
                    f"\r  Reading: [{bar}] {100 * frac:5.1f}%  "
                    f"{offset + count_n:,}/{n_tokens:,} tokens   "
                )
                sys.stderr.flush()
        sys.stderr.write("\n")

    tokens = np.append(tokens, np.array([DiskCdawg.EOS], dtype="<u2"))
    tokens.tofile(output_path)

    n_total = len(tokens)
    unique = sorted(set(tokens.tolist()))
    del tokens

    print(f"  Wrote {n_total:,} tokens to {output_path}")
    print(f"  Unique token IDs ({len(unique)}): {unique}")
    return n_total


def build_cdawg(tokens_path, graph_dir, n_tokens, show_progress=True):
    """Construct and build a DiskCdawg from a converted tokens file.

    Capacity for the memory-mapped node and edge files is estimated from
    ``n_tokens`` via empirically measured ratios (0.40 nodes/tok,
    0.85 edges/tok — measured on cfg3b corpora from 4.9M to 512M tokens).

    Returns the built (but not-yet-fill-counted) DiskCdawg.
    """
    est_nodes = int(n_tokens * 0.40)
    est_edges = int(n_tokens * 0.85)
    print(f"Estimated capacity: {est_nodes:,} nodes, {est_edges:,} edges")

    print("Building DiskCdawg...")
    cdawg = DiskCdawg(tokens_path, graph_dir, est_nodes, est_edges)

    if show_progress:
        _run_with_progress(cdawg.build, graph_dir, label="Build")
    else:
        cdawg.build()

    return cdawg


def fill_counts(cdawg, graph_dir, show_progress=True):
    """Propagate substring frequencies through the graph via topological traversal.

    After ``build_cdawg``, each node represents an equivalence class of
    substrings but doesn't yet know how many times those substrings appear.
    ``fill_counts_ram`` propagates counts from leaves to root.
    """
    if show_progress:
        _run_with_progress(cdawg.fill_counts_ram, graph_dir, label="Counts")
    else:
        cdawg.fill_counts_ram()
