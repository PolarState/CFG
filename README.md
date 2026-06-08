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
