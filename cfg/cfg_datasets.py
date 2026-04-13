import mmap
import random
import struct
from typing import Any

import numpy as np

from . import cfg_generator
from .cfg_grammar import CFGrammar

import torch


class CFGFileDataset(torch.utils.data.Dataset):
    """Dataset to load a cfg from a memory-mapped file.

    The file needs to already contain token ids stored as big-endian
    signed 32-bit integers, grouped into fixed-size windows.

    """

    def __init__(self, filename, device, window_length: int = 512):
        super().__init__()
        self.device = device
        self.filename = filename
        self.window_length = window_length
        self.bytes_per_window = window_length * 4  # 4 bytes per int32

        self._file = open(self.filename, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        file_size = self._mmap.size()
        if file_size % self.bytes_per_window != 0:
            raise ValueError(
                f"File size ({file_size}) is not a multiple of window size "
                f"({self.bytes_per_window} bytes). Data may be corrupted."
            )
        self._num_windows = file_size // self.bytes_per_window

    def __getitem__(self, index):
        offset = index * self.bytes_per_window
        buf = self._mmap[offset : offset + self.bytes_per_window]
        # Read as big-endian int32 and convert to native byte order.
        arr = np.frombuffer(buf, dtype=">i4").astype(np.int64)
        return torch.from_numpy(arr).to(self.device)

    def __len__(self):
        return self._num_windows - 1

    def __del__(self):
        self._mmap.close()
        self._file.close()


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
