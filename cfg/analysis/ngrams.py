"""Exhaustive n-gram enumeration via a suffix-array index.

For each n in 1..max_n, queries all V^n possible token sequences over a
vocab via SA binary search and records the count. Designed for small
vocabularies (cfg3b: V=5 → 5^6 = 15,625 queries for n=6) where full
enumeration is cheaper than the bookkeeping to enumerate only the
non-zero n-grams.

API:

  enumerate_ngrams(tokens, sa, max_n, vocab_size, prefix=None)
      -> {"n1": arr, "n2": arr, ..., "n{max_n}": arr[, "prefix": arr]}

  dump_ngrams_to_csv(arrays, csv_path, vocab_names=None)
      -> writes (length, ngram_ids, ngram_sym, count) for non-zero rows
"""

import csv
import itertools
import os
import time

import numpy as np

from cfg.indexing.sa import count


def enumerate_ngrams(tokens, sa, max_n, vocab_size, prefix=None, verbose=True):
    """Enumerate all V^n n-grams of each length 1..max_n in the corpus.

    With ``prefix`` set to a list of token IDs, each pattern becomes
    ``prefix + suffix`` and the returned ``n{k}`` array of shape (V,)*k
    counts the *suffixes* of length k. (Total pattern length = len(prefix) + k.)

    Returns a dict with keys ``n1`` through ``n{max_n}`` (dense int64 arrays
    of shape ``(vocab_size,) * k``). If ``prefix`` is non-empty, also
    includes ``prefix`` (an int64 array of the prefix IDs) so the result
    is self-describing on round-trip through ``np.savez``.
    """
    prefix_ids = list(prefix) if prefix else []
    prefix_arr = np.array(prefix_ids, dtype=np.uint8) if prefix_ids else None

    arrays = {}
    for n in range(1, max_n + 1):
        total = vocab_size ** n
        full_len = len(prefix_ids) + n
        if verbose:
            print(
                f"\nenumerating {total:,} suffixes of length {n} "
                f"(full {full_len}-gram = prefix + suffix) via SA count()...",
                flush=True,
            )
        counts = np.zeros((vocab_size,) * n, dtype=np.int64)

        pat_buf = np.empty(full_len, dtype=np.uint8)
        if prefix_arr is not None:
            pat_buf[:len(prefix_ids)] = prefix_arr

        t0 = time.time()
        for idx in itertools.product(range(vocab_size), repeat=n):
            pat_buf[len(prefix_ids):] = idx
            counts[idx] = count(tokens, sa, pat_buf)
        elapsed = time.time() - t0

        if verbose:
            nonzero = int((counts > 0).sum())
            total_count = int(counts.sum())
            print(
                f"  {elapsed:.1f}s ({total / max(elapsed, 1e-9):.0f} q/s)  "
                f"nonzero={nonzero:,}/{total:,}  total_count={total_count:,}"
            )

        arrays[f"n{n}"] = counts

    if prefix_ids:
        arrays["prefix"] = np.array(prefix_ids, dtype=np.int64)

    return arrays


def dump_ngrams_to_csv(arrays, csv_path, vocab_names=None):
    """Write the non-zero n-grams from ``arrays`` as a flat CSV.

    Columns: length, ngram_ids (comma-separated), ngram_sym
    (space-separated), count. If ``arrays`` contains a ``prefix`` key,
    its IDs are prepended to every emitted ngram so the row reflects
    the full pattern.

    ``vocab_names`` maps token-id → display string. If None, str(id) is
    used. The mapping must cover every id that appears in the arrays
    plus those in the prefix.
    """
    prefix_ids = tuple(int(x) for x in arrays["prefix"].tolist()) if "prefix" in arrays else ()

    def sym(i):
        return vocab_names[i] if vocab_names is not None else str(i)

    n_keys = sorted(
        (k for k in arrays.keys() if k.startswith("n") and k != "prefix"),
        key=lambda s: int(s[1:]),
    )

    n_rows = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "ngram_ids", "ngram_sym", "count"])
        for key in n_keys:
            arr = arrays[key]
            vocab_size = arr.shape[0]
            n = arr.ndim
            full_len = len(prefix_ids) + n
            for idx in itertools.product(range(vocab_size), repeat=n):
                c = int(arr[idx])
                if c == 0:
                    continue
                full_idx = prefix_ids + idx
                w.writerow([
                    full_len,
                    ",".join(str(i) for i in full_idx),
                    " ".join(sym(i) for i in full_idx),
                    c,
                ])
                n_rows += 1

    return n_rows, os.path.getsize(csv_path)
