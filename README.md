# CFG

Project to recreate the context free grammar (CFG) from the [Physics of Language Models: Part 1](https://arxiv.org/abs/2305.13673) paper.

## Installation

This project is pip-installable as an editable package so it can be used as a library from other projects. Since the library is under active development, editable mode is recommended so that changes are reflected immediately without reinstalling.

To install into a conda environment:

```bash
conda run -n <env_name> pip install -e /path/to/CFG
```

Editable mode means any changes to the source files in `cfg/` are immediately available in the target environment without reinstalling.

## Usage

Once installed, import the modules directly:

```python
from cfg import cfg_defines, cfg_generator, cfg_datasets
from cfg.cfg_grammar import CFGrammar
```

### Training with HuggingFace

An adapter is provided to wrap the character tokenizer for use with HuggingFace models:

```python
from cfg.cfg_grammar import CFGrammar
from cfg.cfg_tokenizers import CFGCharacterTokenizer
from cfg.hf_adapter import HFTokenizerAdapter

import transformers
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Build tokenizer from a grammar's terminal symbols.
grammar = CFGrammar.from_name("cfg3b")
char_tok = CFGCharacterTokenizer(vocab=grammar.terminal_symbols)
tokenizer = HFTokenizerAdapter(char_tok)

# Configure a GPTNeoX model with matching vocab size.
config = transformers.GPTNeoXConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)
model = transformers.GPTNeoXForCausalLM(config)

# Train with the HF Trainer.
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="output/", num_train_epochs=1),
    processing_class=tokenizer,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
```
