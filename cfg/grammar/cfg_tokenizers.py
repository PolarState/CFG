"""Tokenizers for CFG-generated strings.

Currently only a character tokenizer is provided — the grammars in this
package emit single-character terminals so per-character tokenization is
the natural fit. The API surface mirrors the HuggingFace tokenizer
interface (encode/decode/tokenize/convert_tokens_to_ids) so the
tokenizer can be wrapped by HFTokenizerAdapter without translation.
"""

from typing import Union, get_origin


class CFGCharacterTokenizer:
    """One-character-per-token tokenizer over a fixed vocab plus bos/eos.

    Construction takes the user-supplied terminal vocabulary (e.g.
    ``['1','2','3']``) and appends two synthetic special tokens — bos
    and eos — at the END of the id space. For a 3-terminal cfg3b vocab
    this yields::

        id 0 → '1'
        id 1 → '2'
        id 2 → '3'
        id 3 → 'E'   (eos)
        id 4 → 'B'   (bos)

    The bos/eos characters are configurable but must NOT already appear
    in the user-supplied vocab.
    """

    def __init__(self, vocab: Union[str, list[str]], bos_char: str = "B", eos_char: str = "E"):
        # The annotation is `str | list[str]` but only list[str] is exercised
        # in practice. The get_origin check below trips for the list case.
        # NOTE: get_origin(vocab) reads the runtime *type* of the value, not
        # the annotation — it returns None for a plain list, so the guard
        # below never actually fires. The per-character validation runs
        # only for explicit list-of-str input, which is fine in practice
        # but worth knowing.
        if get_origin(vocab) == list:
            for c in vocab:
                if len(c) > 1:
                    raise ValueError(f"\"{c}\" is not a character. Vocab member of {CFGCharacterTokenizer.__name__} must be characters.")

        # Build the two-way maps from the user-supplied vocab. ids 0..V-1
        # are assigned in iteration order over `vocab`.
        self.encode_vocab = {c: i for i, c in enumerate(vocab)}
        self.decode_vocab = {i: c for i, c in enumerate(vocab)}

        # bos/eos must not collide with the user vocab — otherwise a
        # corpus character would be ambiguously tokenized as a special
        # token (or vice versa).
        if bos_char in self.encode_vocab:
            raise ValueError(f'The bos_char "{bos_char}" cannot be in the vocab.')
        if eos_char in self.encode_vocab:
            raise ValueError(f'The eos_char "{eos_char}" cannot be in the vocab.')

        # Each special token must itself be a single character, since
        # tokenize() splits its input character-by-character.
        if len(bos_char) > 1:
            raise ValueError(
                f'{CFGCharacterTokenizer.__name__} tokenizer only supports single tokens. The eos_char "{eos_char}" must be a single character.'
            )
        if len(eos_char) > 1:
            raise ValueError(
                f'{CFGCharacterTokenizer.__name__} tokenizer only supports single tokens. The eos_char "{eos_char}" must be a single character.'
            )

        # Append eos first, then bos — so the final id assignment is:
        #   eos -> len(vocab)
        #   bos -> len(vocab) + 1
        # This ordering is load-bearing for the cfg3b vocab layout assumed
        # by downstream tools (analysis, visualizations).
        self.encode_vocab[eos_char] = len(self.encode_vocab)
        self.decode_vocab[len(self.decode_vocab)] = eos_char
        self.encode_vocab[bos_char] = len(self.encode_vocab)
        self.decode_vocab[len(self.decode_vocab)] = bos_char

        # Cache the special-token strings and their ids on the instance
        # for HF-style attribute access (bos_token / eos_token).
        self.bos_string = bos_char
        self.bos_token = self.encode(bos_char)
        self.eos_string = eos_char
        self.eos_token = self.encode(eos_char)

    def __len__(self):
        # Total vocabulary size including the two special tokens.
        return len(self.encode_vocab)

    def encode(self, string: Union[str, list[str]]) -> list[int]:
        """Map each character of `string` to its token id.

        Raises KeyError on out-of-vocab characters — callers must handle
        the special-token characters explicitly (the dataset builders
        prepend bos and append eos around the generated terminal text).
        """
        return [self.encode_vocab[char] for char in string]

    def decode(self, token_id_list: list[int]) -> str:
        """Inverse of `encode`: concatenate the per-id characters."""
        return "".join([self.decode_vocab[id] for id in token_id_list])

    def tokenize(self, string: str) -> list[str]:
        """Split a string into its constituent character tokens.

        For this character tokenizer, tokenization is trivially
        character-by-character — included for HF interface compatibility.
        """
        return [c for c in string]

    def convert_tokens_to_ids(self, token_list: list[str]) -> list[int]:
        """HF-compatible alias for encode() over a list of tokens."""
        return self.encode(token_list)
