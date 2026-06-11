"""Extract a trained LM's BOS-rooted n-gram probabilities.

Third leg of the measurement loop: ``cfg.analysis.ngrams`` gives the
empirical corpus distribution, ``cfg.analysis.exact`` gives the true
grammar distribution, and this module gives the distribution a trained
model has actually learned — all three in the same npz schema
(``n1``..``n{max_n}`` dense arrays plus a ``prefix`` key) so they can be
compared cell-for-cell.

For each k in 1..max_n, every possible suffix prefix of length k-1 is
fed to the model as ``[bos] + suffix`` and the last-position logits are
softmaxed into a conditional P(next | bos, suffix). Joints then chain
exactly: n{k} = n{k-1}[..., None] * conditional. Each level therefore
sums to 1 and marginalizes onto the previous level, the same invariants
the exact arrays satisfy.

Caveat: prefixes are evaluated with bos at position 0, while training
windows contain bos at arbitrary positions (examples are packed
back-to-back and sliced into 512-token windows). With rotary position
embeddings the difference should be small, but it is a real
train/probe mismatch to keep in mind when reading absolute KL numbers.

torch is imported inside the function (mirroring how hf_adapter
confines the transformers dependency) so the rest of cfg.analysis stays
importable in torch-free environments.
"""

import itertools

import numpy as np


def extract_model_bos_ngrams(model, max_n, vocab_size, bos_id,
                             device=None, batch_size=1024, verbose=True):
    """Compute P(next k tokens | bos) under ``model`` for k = 1..max_n.

    Args:
        model: a causal LM (e.g. GPTNeoXForCausalLM) whose config
            vocab_size equals ``vocab_size`` — the CFG vocabulary
            including bos/eos.
        max_n: deepest suffix length to compute.
        vocab_size: alphabet size V; arrays have shape (V,)*k.
        bos_id: token id to root every prefix at.
        device: torch device; defaults to the model's own device.
        batch_size: prefixes per forward pass.

    Returns:
        ``{"n1": arr, ..., "n{max_n}": arr, "prefix": [bos_id]}`` with
        float64 probability arrays — the same schema as
        ``exact_bos_ngrams`` and (counts aside) ``enumerate_ngrams``.
    """
    import torch

    # The whole construction assumes the model's softmax runs over
    # exactly the CFG vocabulary — otherwise rows wouldn't sum to 1
    # over our V and the joints would leak mass.
    model_vocab = model.config.vocab_size
    if model_vocab != vocab_size:
        raise ValueError(
            f"model vocab_size={model_vocab} != requested vocab_size={vocab_size}"
        )

    model.eval()
    # Default to wherever the model's weights already live so callers
    # don't have to thread a device string through.
    if device is None:
        device = next(model.parameters()).device

    arrays = {}
    # Joint probability of the empty suffix is 1 — the chain rule's
    # base case. Kept as a numpy scalar so the reshape below is uniform.
    prev_joint = np.ones((), dtype=np.float64)

    for k in range(1, max_n + 1):
        n_prefixes = vocab_size ** (k - 1)

        # Enumerate all suffix prefixes of length k-1 in C order —
        # itertools.product matches numpy's row-major reshape, so the
        # flat row index below addresses the same cell as the tuple
        # index in the dense array.
        prefixes = np.fromiter(
            itertools.chain.from_iterable(
                itertools.product(range(vocab_size), repeat=k - 1)
            ),
            dtype=np.int64,
            count=n_prefixes * (k - 1),
        ).reshape(n_prefixes, k - 1)

        # Every input is bos followed by the suffix prefix; no padding,
        # so no attention mask is needed.
        inputs = np.concatenate(
            [np.full((n_prefixes, 1), bos_id, dtype=np.int64), prefixes],
            axis=1,
        )

        # Batch the forward passes and collect each prefix's next-token
        # conditional from the final position's logits.
        conds = np.empty((n_prefixes, vocab_size), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n_prefixes, batch_size):
                batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
                logits = model(input_ids=batch).logits[:, -1, :]
                # Softmax in float64 so each row sums to 1 at the
                # precision the downstream sum/marginalization checks
                # expect.
                probs = torch.softmax(logits.to(torch.float64), dim=-1)
                conds[start : start + batch.shape[0]] = probs.cpu().numpy()

        # Chain rule: joint over k tokens = joint over the first k-1
        # times the conditional on the k-th. Row i of `conds` lines up
        # with flat cell i of prev_joint by the C-order argument above.
        joint = (prev_joint.reshape(-1, 1) * conds).reshape((vocab_size,) * k)
        arrays[f"n{k}"] = joint
        prev_joint = joint

        if verbose:
            print(
                f"  n{k}: {n_prefixes:,} forward(s)  "
                f"sum={joint.sum():.12f}  max={joint.max():.6f}",
                flush=True,
            )

    arrays["prefix"] = np.array([bos_id], dtype=np.int64)
    return arrays
