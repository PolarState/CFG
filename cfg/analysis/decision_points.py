"""Per-decision-point probing: model rule probabilities inside real parses.

Where ``cfg.analysis.model_probs`` measures a model's distribution over
the first few tokens after bos, this module measures it at specific
*grammatical decisions* deep inside real sentences: positions where a
parse trace says "an instance of nonterminal X starts here, and rule r
was chosen". At each such point we ask the model: how much probability
do you give each of X's rules' yields, given everything before?

Scope: terminal-level nonterminals only (every rule expands directly to
terminals — NTs 7/8/9 in the cfg3 family, exactly the ones the masking
experiments knock out). For these, a rule's yield is a literal token
string, so P(rule | context) = the product of per-token conditionals
along the yield — no marginalization over subtrees needed.

Caveat worth keeping in mind when reading the numbers: the model never
*knows* an NT-X starts at a position; its next-token mass also covers
other parse continuations. P(yield_r | context) is therefore a lower
bound on "preference for rule r", and the normalized share among X's
rules is the cleaner relative measure (both are reported).

Cost note: the TRUE rule's yield probability is read off a single
teacher-forced forward of the sentence (the yield lies on the real
token path). ALTERNATIVE rules need extra forwards — after the first
substituted token, the context diverges from the real sentence — so
all (point, rule, step) variants are batched into one padded forward
per sentence.
"""

import numpy as np

from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer


def linearize_trace(tree, target_nt):
    """Flatten a Schema B parse tree into its terminal sequence, recording
    where each instance of ``target_nt`` begins.

    Returns ``(terminals, points)`` where ``terminals`` is the list of
    terminal characters in order and ``points`` is a list of
    ``(rule_idx, start_offset)`` — start_offset indexes into
    ``terminals`` at the first character the instance produces.
    """
    terminals = []
    points = []

    # Iterative DFS (explicit stack) — tree depth is small for cfg3
    # grammars but recursion limits shouldn't be this module's problem.
    stack = [tree]
    while stack:
        node = stack.pop()
        if "t" in node:
            terminals.append(node["t"])
            continue
        if node["nt"] == target_nt:
            points.append((node["rule"], len(terminals)))
        # Push children reversed so they pop in left-to-right order.
        stack.extend(reversed(node["children"]))

    return terminals, points


def terminal_rule_yields(grammar, nt):
    """The yield string of each of ``nt``'s rules, as lists of terminal
    characters. Raises if any rule contains a nonterminal — only
    terminal-level NTs have literal yields."""
    yields = []
    for rule in grammar.rules[nt]:
        for sym in rule:
            if sym in grammar.rules:
                raise ValueError(
                    f"NT {nt!r} rule {rule} contains nonterminal {sym!r}; "
                    f"only terminal-level NTs are supported"
                )
        yields.append(list(rule))
    return yields


def measure_decision_points(model, grammar, traces, target_nt,
                            max_sentences=50, device=None, tokenizer=None,
                            verbose=True):
    """Measure P(each rule's yield | context) at every ``target_nt``
    decision point across ``traces``.

    Args:
        model: causal LM with the CFG vocab.
        grammar: the FULL grammar (rule inventory + yields); traces may
            come from a masked variant, the rule list must not.
        traces: iterable of Schema B tree dicts (one sentence each).
        target_nt: nonterminal to probe (terminal-level).
        max_sentences: stop after this many sentences that contain at
            least one decision point.
        tokenizer: defaults to the dataset builders' construction.

    Returns:
        list of row dicts: sentence (int), offset (int), true_rule
        (int), p_rules (list of float, one per rule of target_nt).
    """
    import torch

    if tokenizer is None:
        tokenizer = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
    bos_id = tokenizer.encode_vocab[tokenizer.bos_string]

    yields = terminal_rule_yields(grammar, target_nt)
    yield_ids = [
        [tokenizer.encode_vocab[c] for c in y] for y in yields
    ]

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    rows = []
    n_done = 0
    for si, tree in enumerate(traces):
        if n_done >= max_sentences:
            break
        terminals, points = linearize_trace(tree, target_nt)
        if not points:
            continue
        n_done += 1

        ids = [bos_id] + [tokenizer.encode_vocab[c] for c in terminals]

        with torch.no_grad():
            # One full forward gives every on-path conditional:
            # logprobs[j] is the distribution for the token at ids[j+1]
            # given ids[:j+1].
            full = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids=full).logits[0]
            logprobs = torch.log_softmax(logits.to(torch.float64), dim=-1)

            # A decision at terminal offset k sits at sequence position
            # k+1 (bos shifts everything by one); its first-token
            # conditional is logprobs[k].
            #
            # Build the off-path variant prefixes for every alternative
            # rule: context + yield_r[:i] predicts yield_r[i], for
            # i >= 1 (i = 0 is on the shared context, free from `full`).
            variants = []   # (point_idx, rule_idx, step_i, prefix_ids)
            for pi, (true_rule, k) in enumerate(points):
                base = ids[: k + 1]
                for ri, y in enumerate(yield_ids):
                    if ri == true_rule:
                        continue
                    for i in range(1, len(y)):
                        variants.append((pi, ri, i, base + y[:i]))

            # Pad-batch the variants and read each prefix's last-token
            # distribution. Right padding works because we gather at
            # each true length - 1.
            var_logp = {}
            if variants:
                maxlen = max(len(v[3]) for v in variants)
                batch = torch.zeros(
                    (len(variants), maxlen), dtype=torch.long, device=device
                )
                mask = torch.zeros_like(batch)
                for vi, (_, _, _, prefix) in enumerate(variants):
                    batch[vi, : len(prefix)] = torch.tensor(prefix)
                    mask[vi, : len(prefix)] = 1
                vlogits = model(input_ids=batch, attention_mask=mask).logits
                for vi, (pi, ri, i, prefix) in enumerate(variants):
                    lp = torch.log_softmax(
                        vlogits[vi, len(prefix) - 1].to(torch.float64), dim=-1
                    )
                    var_logp[(pi, ri, i)] = lp

            for pi, (true_rule, k) in enumerate(points):
                p_rules = []
                for ri, y in enumerate(yield_ids):
                    # First token of every rule is scored from the
                    # shared context distribution.
                    lp = logprobs[k, y[0]].item()
                    for i in range(1, len(y)):
                        if ri == true_rule:
                            # On-path: later conditionals are also in
                            # the full forward.
                            lp += logprobs[k + i, y[i]].item()
                        else:
                            lp += var_logp[(pi, ri, i)][y[i]].item()
                    p_rules.append(float(np.exp(lp)))
                rows.append({
                    "sentence": si,
                    "offset": k,
                    "true_rule": true_rule,
                    "p_rules": p_rules,
                })

        if verbose and n_done % 10 == 0:
            print(f"  {n_done}/{max_sentences} sentences, "
                  f"{len(rows)} decision points", flush=True)

    return rows


def summarize_decision_points(rows, n_rules):
    """Aggregate rows into a per-true-rule summary.

    Returns dict: true_rule -> {n, mean_p (list, per probed rule),
    mean_share (list, per probed rule — p normalized within the NT's
    rules)}. Also key "all" aggregating over every point regardless of
    the true rule (useful when the probe set never contains the masked
    rule as truth).
    """
    groups = {r: [] for r in range(n_rules)}
    for row in rows:
        groups[row["true_rule"]].append(row["p_rules"])

    out = {}
    everything = []
    for r, plist in groups.items():
        if not plist:
            out[r] = {"n": 0, "mean_p": None, "mean_share": None}
            continue
        arr = np.array(plist)
        share = arr / arr.sum(axis=1, keepdims=True)
        out[r] = {
            "n": len(plist),
            "mean_p": arr.mean(axis=0).tolist(),
            "mean_share": share.mean(axis=0).tolist(),
        }
        everything.extend(plist)
    arr = np.array(everything)
    share = arr / arr.sum(axis=1, keepdims=True)
    out["all"] = {
        "n": len(everything),
        "mean_p": arr.mean(axis=0).tolist(),
        "mean_share": share.mean(axis=0).tolist(),
    }
    return out
