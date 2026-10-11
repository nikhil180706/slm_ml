"""
Pre-Training Dataset and Data Loading Utilities.
Supports:
1. In-memory tensor dataset for prototyping.
2. Memory-mapped binary dataset (np.memmap) for loading multi-gigabyte corpora with zero RAM bloat.
3. Auto-regressive target shifting: input = tokens[:-1], target = tokens[1:].
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class PretrainingDataset(Dataset):
    """
    In-memory dataset for small datasets and unit testing.
    """
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        # Total number of complete sequence chunks we can extract
        self.num_samples = (len(token_ids) - 1) // seq_len
        self.token_ids = token_ids

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = self.token_ids[start_idx:end_idx]

        # Auto-regressive shift:
        # x is the input sequence [0 : seq_len]
        # y is the target sequence [1 : seq_len + 1] (next token prediction)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class MemmapDataset(Dataset):
    """
    Memory-mapped dataset for ultra-large pre-training corpora.
    Reads directly from disk on-demand without loading the full file into system RAM.
    """
    def __init__(self, bin_path: str, seq_len: int, dtype=np.uint16):
        assert os.path.exists(bin_path), f"Binary dataset file not found: {bin_path}"
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.num_samples = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = torch.from_numpy((self.data[start_idx:end_idx]).astype(np.int64))

        x = chunk[:-1]
        y = chunk[1:]
        return x, y
