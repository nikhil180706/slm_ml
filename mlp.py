"""
SwiGLU Feed-Forward Network (MLP) Module.
Reference: Shazeer (2020) - "GLU Variants Improve Transformer"
Used in LLaMA 3, Mistral, and modern SLMs.
Replaces the classic ReLU/GELU two-layer MLP with a gated mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Gate and Up projections: map from dim to hidden_dim
        # Modern LLMs omit bias terms (bias=False) to improve training stability
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        # Down projection: maps back from hidden_dim to dim
        self.down_proj = nn.Linear(config.dim, config.dim, bias=False) if config.hidden_dim is None else nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
        x shape: [batch_size, seq_len, dim]
        """
        # Element-wise product of activated gate projection and linear up projection
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
