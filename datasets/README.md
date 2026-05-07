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

| File | Seed | Windows | Tokens | Approx. size |
|---|---|---|---|---|
| `cfg3b_train_dataset_seed0.bin` | 0 | 9,600,001 | 4,915,200,512 | 19 GB |
| `cfg3b_train_dataset_seed1.bin` | 1 | 9,600,001 | 4,915,200,512 | 19 GB |
| `cfg3b_train_dataset_seed2.bin` | 2 | 9,600,001 | 4,915,200,512 | 19 GB |
| `cfg3b_val_dataset_seed1.bin`   | 1 | 96,001    | 49,152,512    | 188 MB |

Window count `9,600,001` is sized for 100,000 optimizer steps × batch 96 plus
one window of headroom for the `__len__ - 1` quirk. Validation `96,001` is the
same scaled to 1,000 eval steps.

> **Note**: train seed=1 and val seed=1 share a generator seed. Grammar-space
> overlap between samples is statistically negligible at this scale, but if
> you need strict independence, do not pair these two for evaluating
> generalization — pair val (seed=1) with train (seed=0) or train (seed=2).

## Generation commands

Run from the `gpt2` project directory (so `../CFG/datasets/` resolves):

```bash
# Training, seed=0
python build_char_dataset.py \
    --output ../CFG/datasets/cfg3b_train_dataset_seed0.bin \
    --num-windows 9600001 --seed 0

# Training, seed=1
python build_char_dataset.py \
    --output ../CFG/datasets/cfg3b_train_dataset_seed1.bin \
    --num-windows 9600001 --seed 1

# Training, seed=2
python build_char_dataset.py \
    --output ../CFG/datasets/cfg3b_train_dataset_seed2.bin \
    --num-windows 9600001 --seed 2

# Validation, seed=1
python build_char_dataset.py \
    --output ../CFG/datasets/cfg3b_val_dataset_seed1.bin \
    --num-windows 96001 --seed 1
```

Each train build takes ~37 minutes on a single core; val takes ~28 seconds.
