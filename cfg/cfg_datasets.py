import torch
from . import cfg_generator

from typing import Any


class CFGDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        cfg_rules: dict[str, str],
        start_symbol: str,
        num_generations: int,
        tokenizer: Any,
        device: torch.device = torch.device("cpu"),
        window_length: int = 512,
    ):
        """Each CFG could be drawn from infinite times. To satisfy PyTorch Dataset, we ask for the length."""
        super().__init__()
        self.cfg_rules = cfg_rules
        self.start_symbol = start_symbol
        self.num_generations = num_generations
        self.idx = 0
        self.generation_buffer = []
        self.window_length = window_length
        self.tokenizer = tokenizer
        self.device = device

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
            self.generation_buffer.extend(
                self.tokenizer.encode(
                    self.tokenizer.bos_token
                    + cfg_generator.generate_from_cfg(self.start_symbol, self.cfg_rules)
                    + self.tokenizer.eos_token
                )
            )

        # Update our fake iterator length.
        self.idx += 1

        # Generate tensors from our window.
        next_item = torch.tensor(
            self.generation_buffer[: self.window_length],
            device=self.device,
        )

        # Trim outgoing tokens from our window.
        self.generation_buffer = self.generation_buffer[self.window_length :]

        return next_item
