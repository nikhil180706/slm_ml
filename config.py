"""
Model Configuration Module for Small Language Model (SLM)
Defines all architectural dimensions and hyperparameters.
All dimensions are aligned with GPU Tensor Core multiples (64/128).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    # Vocabulary & Sequence Length
    vocab_size: int = 32000          # Vocabulary size (default Llama-style or TikToken BPE)
    max_seq_len: int = 2048          # Maximum sequence context window

    # Transformer Core Dimensions
    dim: int = 2048                  # Hidden dimension (embedding dimension)
    n_layers: int = 16               # Number of Transformer Decoder blocks
    
    # Attention Dimensions (Grouped-Query Attention)
    n_heads: int = 16                # Number of Query attention heads
    n_kv_heads: Optional[int] = 4    # Number of Key/Value heads. If None, defaults to n_heads (MHA)
    head_dim: Optional[int] = None   # Dimension of each attention head. Defaults to dim // n_heads

    # Feed-Forward / SwiGLU Dimensions
    hidden_dim: Optional[int] = None # Intermediate dimension for SwiGLU MLP
    multiple_of: int = 128           # Force hidden_dim to be a multiple of this value (Blackwell optimization)
    ffn_dim_multiplier: Optional[float] = None

    # Normalization & Numerical Stability
    norm_eps: float = 1e-5           # Epsilon added to RMSNorm denominator to avoid division by zero
    
    # Rotary Positional Embeddings (RoPE)
    rope_theta: float = 10000.0      # Base frequency theta for RoPE calculation

    # Weight Sharing
    tie_word_embeddings: bool = False # Whether lm_head shares weights with token_embeddings

    def __post_init__(self):
        # Calculate head dimension if not explicitly set
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
        
        # Verify dimension divisibility
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
            
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA"
        )

        # Calculate SwiGLU hidden dimension if not provided
        # Standard SwiGLU uses ~ (8/3) * dim, rounded up to a multiple of 128
        if self.hidden_dim is None:
            hidden_dim = int(2 * (4 * self.dim) / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
            # Round up to multiple_of for peak Tensor Core efficiency on Blackwell
            self.hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """
        Preset for local testing on CPU / Mac laptop with 0 budget.
        Parameters: ~15M - 20M parameters.
        Fast forward and backward pass, ideal for debugging.
        """
        return cls(
            vocab_size=8000,
            max_seq_len=512,
            dim=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            multiple_of=64,
            norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )

    @classmethod
    def slm_1b(cls) -> "ModelConfig":
        """
        Target production SLM (~1.2B Parameters).
        Hardware-aligned for NVIDIA Blackwell training.
        """
        return cls(
            vocab_size=64000,
            max_seq_len=4096,
            dim=2048,
            n_layers=22,
            n_heads=16,
            n_kv_heads=4,            # 4x GQA ratio (4 Query heads per 1 KV head)
            multiple_of=128,
            norm_eps=1e-5,
            rope_theta=500000.0,     # Higher theta for long-context extrapolation
            tie_word_embeddings=False,
        )
