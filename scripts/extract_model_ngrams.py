"""CLI: extract a trained model's BOS-rooted n-gram probabilities.

Thin wrapper around ``cfg.analysis.model_probs``. Loads a checkpoint
via AutoModelForCausalLM (works for the GPTNeoX and GPT2 checkpoints
the gpt2 repo trains), roots every prefix at the CFG tokenizer's bos
id, and writes the same npz schema as compute_ngrams.py (--prefix 4)
and compute_exact_ngrams.py, so all three sources feed the same
comparison tooling.

Requires torch + transformers — run inside the gpt2 conda env:

    conda run -n gpt2 python scripts/extract_model_ngrams.py \
        --checkpoint ~/Source/gpt2/gptneox-cfg3b/<run>/checkpoint-80000 \
        --output-prefix analysis/model_seed0_80k
"""

import argparse
import os

import numpy as np

from cfg.analysis.model_probs import extract_model_bos_ngrams
from cfg.grammar.cfg_grammar import CFGrammar
from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="path to a HF checkpoint directory (config.json + weights)")
    p.add_argument("--cfg", default="cfg3b",
                   help="grammar name from cfg_defines — supplies the tokenizer "
                        "vocab/ids the model was trained with")
    p.add_argument("--max-n", type=int, default=6,
                   help="deepest suffix length after bos to extract (default 6)")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="prefixes per forward pass (default 1024)")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--output-prefix", required=True,
                   help="output path prefix (no extension; .npz appended)")
    return p.parse_args()


def main():
    args = parse_args()

    # Heavy imports stay inside main so --help is instant.
    import torch
    from transformers import AutoModelForCausalLM

    # Rebuild the exact tokenizer construction the training run used
    # (main.py in the gpt2 repo) so token ids line up with the corpora.
    grammar = CFGrammar.from_name(args.cfg)
    tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    vocab_size = len(tokenizer)
    bos_id = tokenizer.encode_vocab[tokenizer.bos_string]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint} on {device}")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model: {model.config.model_type}  params={n_params:,}  "
        f"vocab_size={model.config.vocab_size}"
    )
    print(f"grammar={args.cfg}  bos_id={bos_id}  max_n={args.max_n}")

    arrays = extract_model_bos_ngrams(
        model,
        max_n=args.max_n,
        vocab_size=vocab_size,
        bos_id=bos_id,
        device=device,
        batch_size=args.batch_size,
    )

    npz_path = args.output_prefix + ".npz"
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(npz_path, **arrays)
    print(f"wrote {npz_path} ({os.path.getsize(npz_path):,} bytes)")


if __name__ == "__main__":
    main()
