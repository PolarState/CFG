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
```
