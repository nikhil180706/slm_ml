"""
Rotary Positional Embedding (RoPE) Module.
Reference: Su et al. (2021) - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
Encodes relative position by rotating Query and Key vectors in 2D coordinate planes.
"""

import torch
import torch.nn as nn
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes frequency tensor for complex exponentials (cos and sin) up to max sequence length.
    
    Args:
        dim: Head dimension (must be an even number because pairs of coordinates are rotated)
        end: Maximum sequence length (max_seq_len)
        theta: Base frequency scalar
        
    Returns:
        cos, sin tensors of shape [end, dim]
    """
    assert dim % 2 == 0, f"Head dimension must be even, got {dim}"
    
    # Generate indices: [0, 2, 4, ..., dim-2]
    # theta_i = 1.0 / (theta ** (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Generate position indices: [0, 1, 2, ..., end-1]
    t = torch.arange(end, dtype=torch.float32)
    
    # Outer product: [end, dim // 2]
    freqs = torch.outer(t, freqs)
    
    # Duplicate frequencies so each 2D coordinate pair shares the same frequency
    # Shape becomes [end, dim]
    freqs = torch.cat([freqs, freqs], dim=-1)
    
    # Compute cosine and sine
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.
    For input [x1, x2], returns [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to input Query or Key tensor.
    
    Args:
        x: Tensor of shape [batch_size, n_heads, seq_len, head_dim]
        cos: Tensor of shape [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        sin: Tensor of shape [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        
    Returns:
        Tensor of shape [batch_size, n_heads, seq_len, head_dim] with RoPE applied
    """
    # Ensure cos and sin match tensor dimensions for broadcasting: [1, 1, seq_len, head_dim]
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        
    # Standard 2D rotation formula:
    # x_rotated = (x * cos) + (rotate_half(x) * sin)
    return (x * cos) + (rotate_half(x) * sin)
