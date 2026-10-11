"""
Tokenizer Wrapper and Fallback Utilities.
Supports:
1. Standard HuggingFace Tokenizers (for production BPE).
2. Standalone SimpleCharTokenizer (zero-dependency fallback for local zero-budget testing).
"""

from typing import List, Union
import torch


class SimpleCharTokenizer:
    """
    A lightweight character-level tokenizer for offline, zero-dependency testing.
    Maps every ASCII character to a unique integer ID.
    """
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, text: str) -> List[int]:
        # Simple ASCII byte mapping offset by 3 to reserve special tokens
        tokens = [self.bos_id]
        for char in text:
            val = ord(char) % (self.vocab_size - 3) + 3
            tokens.append(val)
        tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for t in tokens:
            if t in (self.bos_id, self.eos_id, self.pad_id):
                continue
            chars.append(chr(t - 3))
        return "".join(chars)


class Tokenizer:
    """
    Unified Tokenizer wrapper. Tries to load HuggingFace/TikToken if available,
    otherwise falls back smoothly to SimpleCharTokenizer.
    """
    def __init__(self, model_name_or_path: str = "gpt2"):
        self.tokenizer = None
        self.fallback = False

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            # Fallback to local character tokenizer
            self.tokenizer = SimpleCharTokenizer(vocab_size=8000)
            self.fallback = True

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, return_tensors: bool = True) -> Union[List[int], torch.Tensor]:
        if self.fallback:
            tokens = self.tokenizer.encode(text)
            return torch.tensor([tokens], dtype=torch.long) if return_tensors else tokens
        else:
            tokens = self.tokenizer.encode(text)
            return torch.tensor([tokens], dtype=torch.long) if return_tensors else tokens

    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
        return self.tokenizer.decode(tokens)
