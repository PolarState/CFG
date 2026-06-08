# CFG

Project to recreate the context free grammar (CFG) from the [Physics of Language Models: Part 1](https://arxiv.org/abs/2305.13673) paper. Includes the grammar definitions, sampling tools, corpus indexes (suffix array / CDAWG), and n-gram analysis & visualization.

## Layout

```
cfg/                  importable Python package (the library)
  grammar/              grammar definitions, generators, tokenizers, datasets
  indexing/             SA + CDAWG indexes over tokenized corpora
  analysis/             n-gram enumeration and visualization
scripts/              CLI entry points — thin argparse wrappers over cfg.*
  build_dataset.py        — sample grammar → binary dataset
  dump_traces.py          — sample grammar → JSONL parse trees
  build_infinigram.py     — corpus .bin → suffix-array index
  build_dawg.py           — corpus .bin → CDAWG index
  compute_ngrams.py       — SA → dense n-gram count tables (.npz)
  visualize_ngrams.py     — n-gram .npz → heatmap + trie + sunburst
  visualize_grammar_dag.py — trace JSONL → grammar DAG
tests/                pytest suite for the library
datasets/             pre-built `.bin` datasets (see datasets/README.md)
```

The library (`cfg/`) is pure-Python with no torch dependency at import time. The scripts/ entry points each do one thing: parse CLI args, call into the library, print sanity info. To add a new CLI surface for an existing library function, copy one of the scripts as a template.

## Installation

This project is pip-installable as an editable package so it can be used as a library from other projects. Editable mode means any changes to source files in `cfg/` are immediately available without reinstalling.

To install into a conda environment:

```bash
conda run -n <env_name> pip install -e /path/to/CFG
```

## Usage as a library

```python
from cfg.grammar import cfg_defines, cfg_generator, cfg_datasets
from cfg.grammar.cfg_grammar import CFGrammar
from cfg.indexing.sa import build_sa, count, load_index
from cfg.analysis.ngrams import enumerate_ngrams
from cfg.analysis.visualization import render_sunburst
```

### Training with HuggingFace

An adapter wraps the character tokenizer for use with HuggingFace models:

```python
from cfg.grammar.cfg_grammar import CFGrammar
from cfg.grammar.cfg_tokenizers import CFGCharacterTokenizer
from cfg.grammar.hf_adapter import HFTokenizerAdapter

import transformers
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

grammar = CFGrammar.from_name("cfg3b")
char_tok = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
tokenizer = HFTokenizerAdapter(char_tok)

config = transformers.GPTNeoXConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)
model = transformers.GPTNeoXForCausalLM(config)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="output/", num_train_epochs=1),
    processing_class=tokenizer,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
```

## Usage as CLI

End-to-end pipeline: sample a dataset, build an index, enumerate n-grams, visualize.

```bash
# 1. Sample a tokenized binary dataset (or use one from datasets/).
python scripts/build_char_dataset.py --num-windows 100000 --seed 0 \
    --output datasets/my_dataset.bin

# 2. Build a suffix-array index over it.
python scripts/build_infinigram.py --dataset datasets/my_dataset.bin

# 3. Enumerate exact n-gram counts (1..6) via the SA.
python scripts/compute_ngrams.py \
    --sa-dir analysis/sa_my_dataset/ \
    --max-n 6 --output-prefix analysis/my_dataset_ngrams

# 4. Render heatmap + trie + interactive sunburst from the count tables.
python scripts/visualize_ngrams.py \
    --npz analysis/my_dataset_ngrams.npz \
    --output-prefix analysis/my_dataset
```

For grammar-structure inspection (production-rule choice frequencies):

```bash
# Dump parse-tree traces as JSONL.
python scripts/dump_traces.py --num-samples 1000 --seed 0 \
    --output traces.jsonl

# Render the production graph rolled up by NT.
python scripts/visualize_grammar_dag.py --traces traces.jsonl \
    --output-prefix grammar_dag
```

See each script's `--help` for the full flag surface; see the docstrings in `cfg/indexing/`, `cfg/analysis/` for the underlying library functions.

## N-gram visualizations

`scripts/visualize_ngrams.py` consumes a `.npz` of n-gram counts (from `scripts/compute_ngrams.py`) and produces three views from the same data. The sunburst is the most useful at high depth; the heatmap and trie are kept for the views they expose well.

### Sunburst (interactive HTML)

```bash
python scripts/visualize_ngrams.py \
    --npz analysis/my_dataset_ngrams.npz \
    --output-prefix analysis/my_dataset \
    --sunburst-depth 6 \
    --sunburst-color prob \
    --sunburst-size count
```

Produces `<output-prefix>_sunburst.html` — an interactive Plotly chart. Each concentric ring is a depth level: ring 1 = unigrams, ring 6 = 6-grams. Hover any wedge for the full n-gram, count, and `P(child | prefix)`. Click a wedge to zoom into its subtree; click the center to zoom back out.

**Color encodes (configurable via `--sunburst-color`):**

| Mode | Encoding | When to use |
|---|---|---|
| `prob` (default) | `P(child \| prefix)` on viridis (dark purple → bright yellow) | Spot deterministic vs branching steps. Yellow spokes = "given this prefix, the next token is almost certain." |
| `logcount` | `log10(count)` on magma | Spot where the bulk of the corpus mass lives. |
| `token` | Saturated by token identity (amber=terminals, red=eos, blue=bos) | See eos/bos boundary structure. |

**Arc size encodes (configurable via `--sunburst-size`):**

| Mode | Encoding | When to use |
|---|---|---|
| `count` (default) | arc ∝ corpus count of the n-gram | Truthful view: big wedges = high-frequency n-grams. |
| `equal` | every leaf wedge has equal arc; inner rings ∝ leaf count in subtree | Structure-only view: low-frequency paths get the same visual real estate as high-frequency ones. Pair with `--sunburst-color prob` to put all the dynamic range into color. |

**Viewing the HTML:** open directly in any browser (`file://...sunburst.html`), or — if VS Code's preview is unavailable — serve the analysis directory over localhost:

```bash
cd analysis && python3 -m http.server 8765 --bind 127.0.0.1
# then open http://localhost:8765/my_dataset_sunburst.html
```

### Heatmap (PNG/SVG)

`<output-prefix>_heatmap.{png,svg}` — a static matrix of `P(next | prefix-n-gram)`. Rows = non-zero prefixes sorted desc by count, cols = next-token. Cells above `--annot-threshold` (default 0.05) are annotated with the conditional probability. Best for *reading off exact conditional values* at a chosen prefix length (`--prefix-n`, default 5).

### Trie (Graphviz)

`<output-prefix>_trie.{dot,svg}` — a Graphviz tree of every non-zero n-gram up to `--trie-depth` (default 6). Edge labels show `P(child | prefix)`; edge width scales with `log(count)`. Big for high depth (~675 nodes at depth 6 on cfg3b train), but the SVG scrolls cleanly. Best for *enumerating every observed continuation* of a given prefix.

### Prefix-rooted analysis (bos-rooted, etc.)

To restrict the analysis to n-grams starting with a fixed prefix, pass `--prefix` to `compute_ngrams.py`:

```bash
# Enumerate the 7-grams that start with bos (token id 4 in cfg3b):
python scripts/compute_ngrams.py \
    --sa-dir analysis/sa_my_dataset/ \
    --max-n 6 --prefix 4 \
    --output-prefix analysis/my_dataset_bos_ngrams

python scripts/visualize_ngrams.py \
    --npz analysis/my_dataset_bos_ngrams.npz \
    --output-prefix analysis/my_dataset_bos
```

The `.npz` stores the prefix IDs alongside the count arrays; the visualizer reads them automatically and titles/labels every view as bos-rooted. The sunburst's root wedge becomes the prefix itself, and the rings underneath show the conditional distribution at each step *after* the prefix.
