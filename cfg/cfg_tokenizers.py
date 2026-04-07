# This file contains implementations of a tokenizer for CFGs. Currently I only
# support character tokenizers. This may be just a starting point or all I ever
# need.
# 
# The API of the tokenizer currently follows that of the HuggingFace tokenizer.
# Translating between characters and token IDs are the only operations
# supported.

from typing import Union, get_origin

class CFGCharacterTokenizer:
    def __init__(self, vocab: Union[str, list[str]], bos_char: str = "B", eos_char: str = "E"):
        
        if get_origin(vocab) == list:
            for c in vocab:
                if len(c) > 1:
                    raise ValueError(f"\"{c}\" is not a character. Vocab member of {CFGCharacterTokenizer.__name__} must be characters.")
        
        self.encode_vocab = {c: i for i, c in enumerate(vocab)}
        self.decode_vocab = {i: c for i, c in enumerate(vocab)}

        if bos_char in self.encode_vocab:
            raise ValueError(f'The bos_char "{bos_char}" cannot be in the vocab.')
        if eos_char in self.encode_vocab:
            raise ValueError(f'The eos_char "{eos_char}" cannot be in the vocab.')

        if len(bos_char) > 1:
            raise ValueError(
                f'{CFGCharacterTokenizer.__name__} tokenizer only supports single tokens. The eos_char "{eos_char}" must be a single character.'
            )
        if len(eos_char) > 1:
            raise ValueError(
                f'{CFGCharacterTokenizer.__name__} tokenizer only supports single tokens. The eos_char "{eos_char}" must be a single character.'
            )

        self.encode_vocab[eos_char] = len(self.encode_vocab)
        self.decode_vocab[len(self.decode_vocab)] = eos_char
        self.encode_vocab[bos_char] = len(self.encode_vocab)
        self.decode_vocab[len(self.decode_vocab)] = bos_char
        
        self.bos_string = bos_char
        self.bos_token = self.encode(bos_char)
        self.eos_string = eos_char
        self.eos_token = self.encode(eos_char)

    def __len__(self):
        return len(self.encode_vocab)

    # Encode currently only returns token IDs. There are no additional features.
    def encode(self, string: Union[str, list[str]]) -> list[int]:
        return [self.encode_vocab[char] for char in string]

    def decode(self, token_id_list: list[int]) -> str:
        return "".join([self.decode_vocab[id] for id in token_id_list])

    # All tokens are characters.
    def tokenize(self, string: str) -> list[str]:
        return [c for c in string]

    # List of tokens -> list of token ids.
    def convert_tokens_to_ids(self, token_list: list[str]) -> list[int]:
        return self.encode(token_list)
