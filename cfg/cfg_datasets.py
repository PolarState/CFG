import random
import struct
from typing import Any

from . import cfg_generator
from .cfg_grammar import CFGrammar

import torch


class CFGFileDataset(torch.utils.data.Dataset):
    """Dataset to load a cfg from a file.

    The file needs to already contain token ids.

    """

    def __init__(self, filename, device, window_length: int = 512):
        super().__init__()
        self.device = device
        self.filename = filename
        self.window_length = window_length

        self.dataset = []

        with open(self.filename, "rb") as f:
            while True:
                bytes_to_read = self.window_length * struct.calcsize("i")
                binary_chunk = f.read(bytes_to_read)
                if not binary_chunk:
                    break  # End of file

                # Unpack the binary data into a tuple of integers
                format_string = "!" + "i" * self.window_length
                if len(binary_chunk) != struct.calcsize(format_string):
                    raise ValueError(
                        f"Unexpected end of file or corrupted data. Expected {struct.calcsize(format_string)} bytes, got {len(binary_chunk)}."
                    )

                token_list = list(struct.unpack(format_string, binary_chunk))
                self.dataset.append(token_list)

    def __getitem__(self, index):
        return torch.tensor(self.dataset[index], device=self.device)

    def __len__(self):
        return len(self.dataset) - 1


class CFGRandomGenerationDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        cfg_rules: dict[str, list[list[str]]] | CFGrammar,
        num_generations: int,
        tokenizer: Any,
        device: torch.device = torch.device("cpu"),
        window_length: int = 512,
    ):
        """Each CFG could be drawn from infinite times. To satisfy PyTorch Dataset, we ask for the length."""
        super().__init__()

        # Wrap raw rules in a CFGrammar to cache derived state (terminal
        # symbols, start symbols) rather than re-deriving on each call.
        if isinstance(cfg_rules, CFGrammar):
            self.grammar = cfg_rules
        else:
            self.grammar = CFGrammar(cfg_rules)

        # Keep cfg_rules as an alias for backwards compatibility with
        # code that references it directly.
        self.cfg_rules = self.grammar.rules

        self.num_generations = num_generations
        self.idx = 0
        self.window_length = window_length
        self.tokenizer = tokenizer
        self.device = device

        # Make the first token the Eos token as it's the divider token between datasets.
        self.generation_buffer = []

    def __len__(self):
        return self.num_generations

    def __iter__(self):
        # Reset our internal count when we're asked to iterate again.
        self.idx = 0
        return self

    def __next__(self):
        # Exit if we've completed all iterations.
        if self.idx >= len(self):
            raise StopIteration

        # Fill our generation buffer up to our widow length.
        while len(self.generation_buffer) < self.window_length:
            self.generation_buffer.extend(self.tokenizer.bos_token)
            self.generation_buffer.extend(
                self.tokenizer.encode(c)[0]
                for c in self.grammar.generate()
            )
            self.generation_buffer.extend(self.tokenizer.eos_token)

        # Update our fake iterator length.
        self.idx += self.window_length

        # Generate tensors from our window.
        next_item = torch.tensor(
            self.generation_buffer[: self.window_length],
            device=self.device,
        )

        # Trim outgoing tokens from our window.
        self.generation_buffer = self.generation_buffer[self.window_length :]

        return next_item
