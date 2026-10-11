"""
Grouped-Query Attention (GQA) Module with KV-Cache Support.
Reference: Ainslie et al. (2023) - "GQA: Training Generalized Multi-Query Transformer Models"
Allows multiple Query heads to share a single Key/Value head group.
Reduces KV-cache memory usage by 4x to 8x during inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import ModelConfig
from rotary import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands Key/Value heads to match Query head count in Grouped-Query Attention.
    Input shape: [batch, n_kv_heads, seq_len, head_dim]
    Output shape: [batch, n_heads, seq_len, head_dim] (where n_heads = n_kv_heads * n_rep)
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.n_heads // self.n_kv_heads

        # Projections (no bias for modern LLMs)
        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            cos, sin: RoPE frequency tensors
            kv_cache: Optional tuple (past_keys, past_values) for fast auto-regressive generation
            use_cache: Whether to return updated KV cache
            
        Returns:
            output: [batch, seq_len, dim]
            new_kv_cache: (keys, values) if use_cache is True, else None
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape to multi-head format: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Positional Embeddings to Q and K (never V)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 4. Handle KV-Cache for inference
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # 5. Expand Key and Value heads to match Query heads for GQA
        k_expanded = repeat_kv(k, self.num_key_value_groups)
        v_expanded = repeat_kv(v, self.num_key_value_groups)

        # 6. Compute Scaled Dot-Product Attention
        # PyTorch's native F.scaled_dot_product_attention automatically uses FlashAttention
        # when available on NVIDIA GPUs, and falls back to efficient C++ kernels elsewhere.
        # If cache is active during token-by-token generation (seq_len == 1), causal mask is not needed.
        is_causal = (kv_cache is None and seq_len > 1)
        
        output = F.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # 7. Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 8. Output projection
        return self.o_proj(output), new_kv_cache
