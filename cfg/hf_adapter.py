# HuggingFace adapter for CFGCharacterTokenizer.
#
# This module provides a thin wrapper that makes a CFGCharacterTokenizer
# compatible with the HuggingFace PreTrainedTokenizer interface. This
# keeps the core CFG library free of any HuggingFace dependency while
# still allowing CFG tokenizers to be used with HF models and pipelines.

import json
import os

from transformers import PreTrainedTokenizer

from .cfg_tokenizers import CFGCharacterTokenizer


class HFTokenizerAdapter(PreTrainedTokenizer):
    """Wraps a CFGCharacterTokenizer for use with HuggingFace models.

    The adapter delegates all tokenization logic to the underlying
    CFGCharacterTokenizer and exposes the standard HF interface:
    ``__call__`` with ``return_tensors``, ``batch_decode``,
    string-valued ``bos_token`` / ``eos_token`` / ``pad_token``, etc.

    Usage::

        from cfg.cfg_tokenizers import CFGCharacterTokenizer
        from cfg.hf_adapter import HFTokenizerAdapter

        char_tok = CFGCharacterTokenizer(vocab=["0", "1", "2"])
        hf_tok = HFTokenizerAdapter(char_tok)

        # Now usable with any HuggingFace model.
        inputs = hf_tok("B012E", return_tensors="pt")
    """

    # HF looks for this attribute when loading from a saved directory.
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, cfg_tokenizer: CFGCharacterTokenizer, **kwargs):
        # Store the underlying tokenizer before super().__init__() because
        # the parent constructor may invoke methods that need it (e.g.
        # _convert_token_to_id when registering special tokens).
        self._cfg_tokenizer = cfg_tokenizer

        # Map the CFGCharacterTokenizer's special tokens to HF string
        # special-token arguments.  pad_token defaults to eos_token,
        # matching the convention used in the gpt2 training scripts.
        kwargs.setdefault("bos_token", cfg_tokenizer.bos_string)
        kwargs.setdefault("eos_token", cfg_tokenizer.eos_string)
        kwargs.setdefault("pad_token", cfg_tokenizer.eos_string)

        super().__init__(**kwargs)

    # -- Core vocabulary interface -------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (including special tokens)."""
        return len(self._cfg_tokenizer)

    def get_vocab(self) -> dict[str, int]:
        """Return the full token-to-id mapping as a plain dict."""
        return dict(self._cfg_tokenizer.encode_vocab)

    # -- Tokenization primitives ---------------------------------------------

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Split text into character tokens via the underlying tokenizer."""
        return self._cfg_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Map a single character token to its integer id."""
        # Return the eos id for unknown tokens as a safe fallback.
        return self._cfg_tokenizer.encode_vocab.get(
            token, self._cfg_tokenizer.encode_vocab[self._cfg_tokenizer.eos_string]
        )

    def _convert_id_to_token(self, index: int) -> str:
        """Map an integer id back to its character token."""
        return self._cfg_tokenizer.decode_vocab.get(index, "")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join character tokens without spaces (unlike the default which
        inserts a space between each token)."""
        return "".join(tokens)

    # -- Serialization -------------------------------------------------------

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        """Write the vocabulary to a JSON file in the given directory.

        This is required by the HF interface for ``save_pretrained`` /
        ``from_pretrained`` round-tripping.
        """
        # Build the output filename.
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_file = os.path.join(save_directory, f"{prefix}vocab.json")

        # Persist the token-to-id mapping.
        with open(vocab_file, "w") as f:
            json.dump(self._cfg_tokenizer.encode_vocab, f, indent=2)

        return (vocab_file,)
