"""CLI: per-prefix KL divergence between two BOS-rooted n-gram sources.

Thin wrapper around ``cfg.analysis.compare``. The two inputs are npz
files in the shared schema, from any of:

  compute_ngrams.py --prefix 4      (empirical corpus counts)
  compute_exact_ngrams.py           (exact grammar probabilities)
  extract_model_ngrams.py           (trained-model probabilities)

P is the reference: rows cover P-supported prefixes and the per-level
summary is the expectation of KL(P||Q) under P's prefix distribution.

Outputs (under <output-prefix>):

  <output-prefix>.csv   level, prefix_ids, prefix_syms, p_mass, kl_bits
  <output-prefix>.png   per-level expected KL bars + the top-20
                        worst prefixes at the deepest level
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cfg.analysis.compare import kl_rows

# cfg3b vocab labels — script-level concern, mirroring compute_ngrams.py.
VOCAB_NAMES = {0: "1", 1: "2", 2: "3", 3: "eos", 4: "bos"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--p-npz", required=True, help="reference source P (.npz)")
    p.add_argument("--q-npz", required=True, help="comparison source Q (.npz)")
    p.add_argument("--p-label", default="P", help="display label for P")
    p.add_argument("--q-label", default="Q", help="display label for Q")
    p.add_argument("--max-n", type=int, default=None,
                   help="deepest level to compare (default: all common levels)")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="optional floor added to Q's conditionals to make "
                        "infinite KL finite (default 0: report inf)")
    p.add_argument("--output-prefix", required=True,
                   help="output path prefix (.csv and .png appended)")
    return p.parse_args()


def main():
    args = parse_args()
    p_arrays = dict(np.load(args.p_npz))
    q_arrays = dict(np.load(args.q_npz))

    # Both sources must be rooted at the same fixed prefix (e.g. bos),
    # otherwise the conditionals aren't about the same quantity at all.
    p_prefix = p_arrays.get("prefix", np.array([])).tolist()
    q_prefix = q_arrays.get("prefix", np.array([])).tolist()
    if p_prefix != q_prefix:
        raise SystemExit(
            f"prefix mismatch: {args.p_npz} has {p_prefix}, "
            f"{args.q_npz} has {q_prefix}"
        )

    rows, summary = kl_rows(
        p_arrays, q_arrays, max_n=args.max_n, epsilon=args.epsilon
    )

    # ── CSV: one row per P-supported prefix ────────────────────────────
    csv_path = args.output_prefix + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "prefix_ids", "prefix_syms", "p_mass", "kl_bits"])
        for r in rows:
            w.writerow([
                r["level"],
                ",".join(str(i) for i in r["prefix"]),
                " ".join(VOCAB_NAMES.get(i, str(i)) for i in r["prefix"]),
                f"{r['p_mass']:.10g}",
                f"{r['kl_bits']:.10g}",
            ])

    # ── Printed per-level summary ──────────────────────────────────────
    print(f"KL({args.p_label} || {args.q_label}) per level (bits):")
    for k, s in summary.items():
        inf_note = f"  [{s['n_inf']} inf prefixes]" if s["n_inf"] else ""
        print(
            f"  n{k}: E[KL]={s['finite_expected_kl_bits']:.6f}  "
            f"max={s['max_kl_bits']:.6f}  "
            f"prefixes={s['n_prefixes']:,}{inf_note}"
        )
    print(f"wrote {csv_path}")

    # ── PNG: level summary + worst offenders at the deepest level ─────
    deepest = max(summary.keys())
    deep_rows = sorted(
        (r for r in rows if r["level"] == deepest and np.isfinite(r["kl_bits"])),
        key=lambda r: -r["kl_bits"],
    )[:20]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, max(4.0, 0.3 * len(deep_rows))),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    levels = list(summary.keys())
    ax1.bar([f"n{k}" for k in levels],
            [summary[k]["finite_expected_kl_bits"] for k in levels],
            color="steelblue")
    ax1.set_ylabel("E[KL] (bits)")
    ax1.set_title(f"KL({args.p_label} || {args.q_label}) by level")

    labels = [
        " ".join(VOCAB_NAMES.get(i, str(i)) for i in r["prefix"]) or "(root)"
        for r in deep_rows
    ]
    ax2.barh(range(len(deep_rows)), [r["kl_bits"] for r in deep_rows],
             color="indianred")
    ax2.set_yticks(range(len(deep_rows)))
    ax2.set_yticklabels(labels, fontfamily="monospace", fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("KL (bits)")
    ax2.set_title(f"worst prefixes at n{deepest}")

    fig.tight_layout()
    png_path = args.output_prefix + ".png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
