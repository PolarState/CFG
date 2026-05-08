# CFG Datasets

This directory holds binary datasets generated from CFG grammars and consumed
by `cfg.cfg_datasets.CFGFileDataset`. Each `.bin` filename ends with the seed
used to generate it (e.g. `cfg3b_train_dataset_seed0.bin`) so that two runs
drawn from the same grammar are easy to distinguish.

## File format

All `.bin` files share the same on-disk layout (see
`cfg.cfg_datasets.CFGFileDataset`):

- Token IDs stored as **big-endian signed int32**.
- Tokens packed into **fixed-size windows** of 512 tokens (2,048 bytes/window).
- Total file size is always a multiple of `window_length × 4` bytes.
- No per-window header. Example boundaries do not align to window boundaries —
  example bytes are `bos_id + char_ids + eos_id` concatenated end-to-end across
  the whole stream, then sliced into windows. The final window is padded with
  `pad_token_id` (= `eos_token_id`) up to 512 tokens.

`CFGFileDataset.__len__` returns `num_windows - 1` (the last window is
reserved/dropped from the visible range, regardless of `reverse=` mode).

## Common assumptions for `cfg3b` datasets

- **Grammar**: `cfg3b` from `cfg.cfg_defines` (terminals `'1'`, `'2'`, `'3'`).
- **Tokenizer**: `CFGCharacterTokenizer(vocab=grammar.terminal_symbols)` wrapped
  in `HFTokenizerAdapter`. Token IDs:
  - `0 = '1'`
  - `1 = '2'`
  - `2 = '3'`
  - `3 = 'E'` (eos, also used as pad)
  - `4 = 'B'` (bos)
- **Window length**: 512.
- **Generation tool**: `gpt2/build_char_dataset.py --num-windows N --seed S` —
  streams examples from `grammar.generate()`, encodes via
  `CFGCharacterTokenizer.encode_vocab` (bypassing the HF tokenizer for speed),
  and flushes 512-token windows directly to disk in batches.

## Files

### Standard length (9,600,001 windows ≈ 4.92B tokens, ~19 GB each)

| File | Seed | Notes |
|---|---|---|
| `cfg3b_train_dataset_seed0.bin` | 0 | Reference training set. |
| `cfg3b_train_dataset_seed1.bin` | 1 | Independent draw, same distribution. |
| `cfg3b_train_dataset_seed2.bin` | 2 | Independent draw, same distribution. |
| `cfg3b_val_dataset_seed1.bin`   | 1 | 96,001 windows (~49M tokens, 188 MB). |

Window count `9,600,001` is sized for 100,000 optimizer steps × batch 96 plus
one window of headroom for the `__len__ - 1` quirk. Validation `96,001` is
the same scaled to 1,000 eval steps.

> **Note**: train seed=1 and val seed=1 share a generator seed. Grammar-space
> overlap between samples is statistically negligible at this scale, but if
> you need strict independence, do not pair these two for evaluating
> generalization — pair val (seed=1) with train (seed=0) or train (seed=2).

### Extended length

| File | Seed | Windows | Notes |
|---|---|---|---|
| `cfg3b_train_dataset_seed0_long.bin` | 0 | 19,200,001 | 9.83B tokens, ~37 GB. **Byte-exact prefix-extension** of `cfg3b_train_dataset_seed0.bin`: the first 19,660,802,048 bytes (= 9,600,001 windows) are identical, with another 9,600,001 fresh windows appended. Verified via `cmp -n`. |

### Production-rule masks (seed=0, standard 9,600,001 windows)

Each masked file suppresses one of the two production rules of an NT in the
deepest layer of `cfg3b` (NTs `7`, `8`, `9` — these productions yield only
terminals). Implementation: pass a `weights` dict to
`CFGrammar.generate(weights=...)` with `0.0` for the masked rule and `1.0`
elsewhere. The grammar object is *not* modified; suppression is purely at
sampling time. Verified by walking 1,000 sampled trees per spec — none
contained the masked `(nt, rule)` node, vs 1,000/1,000 in the unmasked
baseline.

| File | Mask | Suppressed production | Kept production |
|---|---|---|---|
| `cfg3b_train_dataset_seed0_mask7a.bin` | `7:0` | `['3', '1']`     | `['1', '2', '3']` |
| `cfg3b_train_dataset_seed0_mask7b.bin` | `7:1` | `['1', '2', '3']` | `['3', '1']`      |
| `cfg3b_train_dataset_seed0_mask8a.bin` | `8:0` | `['3', '2']`     | `['3', '1', '2']` |
| `cfg3b_train_dataset_seed0_mask8b.bin` | `8:1` | `['3', '1', '2']` | `['3', '2']`      |
| `cfg3b_train_dataset_seed0_mask9a.bin` | `9:0` | `['3', '2', '1']` | `['2', '1']`      |
| `cfg3b_train_dataset_seed0_mask9b.bin` | `9:1` | `['2', '1']`     | `['3', '2', '1']` |

All six are 19,660,802,048 bytes (9,600,001 windows × 512 tokens × 4 bytes).
Note: total **tokens** is the same across all six, but the number of
**examples** generated differs because masking the longer rule shortens
average example length.

> Because all six share `--seed 0`, the **first** generation call advances the
> RNG identically (same start symbol picked). They diverge after the first
> rule choice that touches the masked NT, which happens early in the very
> first derivation. None of the six is a prefix of any other.

## Generation commands

Run from the `gpt2` project directory (so `../CFG/datasets/` resolves):

```bash
# Standard training sets
for s in 0 1 2; do
    python build_char_dataset.py \
        --output ../CFG/datasets/cfg3b_train_dataset_seed${s}.bin \
        --num-windows 9600001 --seed ${s}
done

# Validation, seed=1
python build_char_dataset.py \
    --output ../CFG/datasets/cfg3b_val_dataset_seed1.bin \
    --num-windows 96001 --seed 1

# Extended length (2x), seed=0 — prefix-extends seed0
python build_char_dataset.py \
    --output ../CFG/datasets/cfg3b_train_dataset_seed0_long.bin \
    --num-windows 19200001 --seed 0

# Production-rule masks (all seed=0)
for spec in 7:0 7:1 8:0 8:1 9:0 9:1; do
    nt=${spec%:*}
    idx=${spec#*:}
    suffix=$([ $idx -eq 0 ] && echo a || echo b)
    python build_char_dataset.py \
        --output ../CFG/datasets/cfg3b_train_dataset_seed0_mask${nt}${suffix}.bin \
        --num-windows 9600001 --seed 0 --mask-rule ${spec}
done
```

A standard-length build takes ~37 min single-threaded; the long build ~75 min;
val takes ~28 s. Mask builds that suppress the longer rule of a pair (`7b`,
`8b`, `9a`) are slightly slower (~55–60 min) because more NT expansions are
needed to fill 512-token windows.
