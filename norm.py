"""
Root Mean Square Normalization (RMSNorm) Module.
Reference: Zhang & Sennrich (2019) - "Root Mean Square Layer Normalization"
Used in modern LLMs (Llama 3, Mistral) instead of standard LayerNorm.
Skips mean centering to reduce GPU memory bandwidth traffic.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: Input feature dimension to normalize across.
            eps: Small constant for numerical stability (prevents division by zero).
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma in literature), initialized to ones
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes x / RMS(x)
        x shape: [batch_size, seq_len, dim]
        """
        # Calculate mean of squares along the last dimension (dim)
        # Keepdim=True preserves the tensor rank for broadcasting
        # We compute in float32 for numerical stability during mixed-precision training
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stable normalization, then back to original dtype
        output = self._norm(x.float()).type_as(x)
        # Element-wise scaling by learnable weight
        return output * self.weight
