"""
Full Small Language Model (SLM) Architecture.
Assembles Token Embeddings, Stacks of Transformer Blocks, Final RMSNorm, and Language Modeling Head.
Supports training loss computation, KV caching for inference, and weight tying.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from config import ModelConfig
from norm import RMSNorm
from rotary import precompute_freqs_cis
from block import TransformerBlock


class SLMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final RMS Normalization
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Output LM Head (projects hidden state back to vocabulary logits)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Optional Weight Tying (reduces memory footprint on tiny models)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE cos/sin tables up to max_seq_len
        cos, sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initializes weights using standard normal distribution scaled for deep Transformers.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids: Tensor of token indices [batch_size, seq_len]
            targets: Optional ground-truth token indices [batch_size, seq_len] for computing training loss
            kv_caches: List of (k, v) caches per layer for generation
            start_pos: Starting index in sequence (used when generating with KV cache)
            use_cache: Whether to return new KV caches
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Cross-entropy scalar loss (if targets is provided) else None
            new_kv_caches: Updated KV caches (if use_cache is True) else None
        """
        batch_size, seq_len = input_ids.shape
        assert start_pos + seq_len <= self.config.max_seq_len, (
            f"Sequence length {start_pos + seq_len} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # 1. Look up token embeddings
        h = self.embed_tokens(input_ids)

        # 2. Slice RoPE cos and sin tables for current sequence positions
        cos = self.cos_cached[start_pos : start_pos + seq_len]
        sin = self.sin_cached[start_pos : start_pos + seq_len]

        # 3. Pass through all Transformer blocks
        new_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, updated_cache = layer(
                h,
                cos=cos,
                sin=sin,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(updated_cache)

        # 4. Final normalization
        h = self.norm(h)

        # 5. Compute vocabulary logits
        logits = self.lm_head(h)

        # 6. Compute loss if ground-truth targets are provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross-entropy:
            # logits: [batch * seq_len, vocab_size]
            # targets: [batch * seq_len]
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,  # Standard PyTorch mask token
            )

        return logits, loss, new_kv_caches

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_non_embedding_parameters(self) -> int:
        """Returns parameter count excluding token embeddings."""
        embedding_params = self.embed_tokens.weight.numel()
        return self.count_parameters() - embedding_params
