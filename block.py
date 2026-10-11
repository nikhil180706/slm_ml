"""
Transformer Decoder Block.
Combines Pre-RMSNorm, Grouped-Query Attention, SwiGLU MLP, and Residual connections.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from config import ModelConfig
from norm import RMSNorm
from attention import GroupedQueryAttention
from mlp import SwiGLUMLP


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Pre-attention normalization
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        # Self-attention module
        self.self_attn = GroupedQueryAttention(config)
        
        # Post-attention / Pre-MLP normalization
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        # Feed-forward SwiGLU network
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Standard Pre-Norm Transformer structure:
        h = x + Attention(RMSNorm(x))
        out = h + MLP(RMSNorm(h))
        """
        # 1. Pre-Norm + Attention + Residual
        normed_x = self.input_layernorm(x)
        attn_out, new_kv_cache = self.self_attn(
            normed_x,
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        h = x + attn_out

        # 2. Pre-Norm + MLP + Residual
        out = h + self.mlp(self.post_attention_layernorm(h))
        
        return out, new_kv_cache
