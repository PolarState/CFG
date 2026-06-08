"""Suffix-array index over a tokenized corpus.

Pure-data utilities — no torch, no CFG-specific knowledge. The "infinigram"
framing (arbitrary-length n-gram count queries via SA binary search) comes
from Liu et al. 2024 (https://arxiv.org/abs/2406.13069).

The on-disk layout that ``save_index`` writes and ``load_index`` reads is:

  tokens.bin        raw little-endian uint8 of the token stream
  suffix_array.bin  raw little-endian int32 (or int64) suffix array
  meta.txt          small text file recording dtype + lengths

Why uint8: for small alphabets (e.g., cfg3b has V=5) the int32 → uint8
downcast is lossless and lets pydivsufsort use its fast bytes path.
``convert_dataset`` asserts the downcast holds before writing.

API:

  convert_dataset(dataset_path, window_length=512, max_windows=None) -> tokens
  build_sa(tokens)                                                   -> sa
  save_index(output_dir, tokens, sa)                                 -> paths
  load_meta(sa_dir)                                                  -> dict
  load_index(sa_dir)                                                 -> (tokens, sa, n_tokens)
  count(tokens, sa, pattern)                                         -> int
  next_token_distribution(tokens, sa, context, vocab)                -> (ctx_count, [(tok, ext_count, prob), ...])
"""

import os
import sys

import numpy as np
import pydivsufsort


def convert_dataset(dataset_path, window_length=512, max_windows=None):
    """Read a big-endian int32 .bin file into a uint8 numpy array.

    The CFG .bin files store token IDs as big-endian int32 in fixed-size
    windows (see cfg.grammar.cfg_datasets.CFGFileDataset). For small
    alphabets the int32 → uint8 downcast is lossless; we assert it holds
    chunk-by-chunk while reading.

    Reads in 50M-token chunks for inputs larger than 50M tokens so peak
    memory stays bounded. Prints a progress bar to stderr in that case.
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
        max_id = int(raw.max())
        if max_id > 255:
            raise ValueError(f"token id {max_id} exceeds uint8 range — use a wider dtype")
        tokens = raw.astype(np.uint8)
        del raw
    else:
        tokens = np.empty(n_tokens, dtype=np.uint8)
        n_chunks = (n_tokens + chunk_size - 1) // chunk_size
        bar_width = 40
        with open(dataset_path, "rb") as f:
            for i in range(n_chunks):
                offset = i * chunk_size
                count_n = min(chunk_size, n_tokens - offset)
                chunk = np.frombuffer(f.read(count_n * 4), dtype=">i4")
                cmax = int(chunk.max())
                if cmax > 255:
                    raise ValueError(f"token id {cmax} exceeds uint8 range")
                tokens[offset:offset + count_n] = chunk.astype(np.uint8)

                frac = (offset + count_n) / n_tokens
                filled = int(bar_width * frac)
                bar = "█" * filled + "░" * (bar_width - filled)
                sys.stderr.write(
                    f"\r  Reading: [{bar}] {100 * frac:5.1f}%  "
                    f"({offset + count_n:,}/{n_tokens:,} tokens)"
                )
                sys.stderr.flush()
        sys.stderr.write("\n")

    return tokens


def build_sa(tokens):
    """Build a suffix array over ``tokens`` using libdivsufsort.

    Uses int32 SA for inputs ≤ 2^31 tokens, int64 (``force64=True``)
    above. The SA has the same length as ``tokens`` (no sentinel —
    libdivsufsort handles the implicit end-of-text).
    """
    force64 = len(tokens) >= 2**31
    print(f"  Building SA over {len(tokens):,} tokens "
          f"({'int64' if force64 else 'int32'} SA)...")
    return pydivsufsort.divsufsort(tokens, force64=force64)


def save_index(output_dir, tokens, sa):
    """Write tokens + SA as raw little-endian binary blobs plus meta.txt."""
    tokens_path = os.path.join(output_dir, "tokens.bin")
    sa_path = os.path.join(output_dir, "suffix_array.bin")
    meta_path = os.path.join(output_dir, "meta.txt")

    tokens.tofile(tokens_path)
    sa.tofile(sa_path)

    with open(meta_path, "w") as f:
        f.write(f"n_tokens {len(tokens)}\n")
        f.write(f"tokens_dtype {tokens.dtype.str}\n")
        f.write(f"sa_dtype {sa.dtype.str}\n")

    return tokens_path, sa_path, meta_path


def load_meta(sa_dir):
    """Parse meta.txt into a dict of strings."""
    meta = {}
    with open(os.path.join(sa_dir, "meta.txt")) as f:
        for line in f:
            if not line.strip():
                continue
            k, v = line.strip().split(None, 1)
            meta[k] = v
    return meta


def load_index(sa_dir):
    """Memory-map a previously-saved index.

    Returns ``(tokens, sa, n_tokens)``. Both arrays are mmap'd in
    copy-on-write mode (``mode='c'``) so pydivsufsort can operate on
    them without complaint about read-only buffers.
    """
    meta = load_meta(sa_dir)
    n_tokens = int(meta["n_tokens"])
    tokens_dtype = np.dtype(meta["tokens_dtype"])
    sa_dtype = np.dtype(meta["sa_dtype"])
    tokens = np.memmap(
        os.path.join(sa_dir, "tokens.bin"),
        dtype=tokens_dtype, mode="c", shape=(n_tokens,),
    )
    sa = np.memmap(
        os.path.join(sa_dir, "suffix_array.bin"),
        dtype=sa_dtype, mode="c", shape=(n_tokens,),
    )
    return tokens, sa, n_tokens


def count(tokens, sa, pattern):
    """Count occurrences of ``pattern`` in ``tokens`` via SA binary search.

    Wraps ``pydivsufsort.sa_search``, which returns ``(count, first_pos)``.
    ``pattern`` must be a numpy array of the same dtype as ``tokens``.
    """
    n, _ = pydivsufsort.sa_search(tokens, sa, pattern)
    return int(n)


def next_token_distribution(tokens, sa, context, vocab):
    """Return ``(ctx_count, [(tok, ext_count, prob), ...])`` for candidate next tokens.

    Probabilities are conditioned on ``context`` appearing in the corpus.
    If ``context`` is empty, ``ctx_count`` is ``len(tokens)``. If
    ``context`` never appears (``ctx_count == 0``), probs are 0.0.
    """
    ctx = np.asarray(context, dtype=tokens.dtype)
    ctx_count = count(tokens, sa, ctx) if len(ctx) > 0 else len(tokens)

    extended = np.empty(len(ctx) + 1, dtype=tokens.dtype)
    extended[:len(ctx)] = ctx

    dist = []
    for tok in vocab:
        extended[-1] = tok
        ext_count = count(tokens, sa, extended)
        prob = ext_count / ctx_count if ctx_count > 0 else 0.0
        dist.append((int(tok), ext_count, prob))
    return ctx_count, dist
