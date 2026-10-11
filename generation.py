"""
Auto-Regressive Text Generation Engine.
Implements:
- KV-Cached generation for fast token generation.
- Sampling strategies: Greedy, Temperature, Top-K, and Top-P (Nucleus).
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from model import SLMModel


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Applies Top-P (Nucleus) filtering to probability distribution.
    Keeps only the smallest set of tokens whose cumulative probability exceeds p.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


@torch.no_grad()
def generate(
    model: SLMModel,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generates text auto-regressively using KV caching.
    
    Args:
        model: Trained SLMModel instance
        prompt_tokens: Tensor of shape [batch_size, prompt_len]
        max_new_tokens: Maximum number of tokens to generate
        temperature: Softmax temperature (>0.0). Higher = more creative, lower = more deterministic.
        top_k: Top-K filtering threshold.
        top_p: Nucleus filtering threshold.
        eos_token_id: Stop generation if this token is produced.
        
    Returns:
        Tensor of shape [batch_size, prompt_len + generated_tokens]
    """
    model.eval()
    device = prompt_tokens.device
    batch_size, prompt_len = prompt_tokens.shape
    
    # 1. Prefill stage: Process the entire prompt at once to populate initial KV caches
    logits, _, kv_caches = model(
        input_ids=prompt_tokens,
        start_pos=0,
        use_cache=True,
    )
    
    # Get logits of the very last prompt token
    next_token_logits = logits[:, -1, :]
    
    generated = prompt_tokens
    cur_pos = prompt_len

    # 2. Decode stage: Generate one token at a time using cached past keys/values
    for _ in range(max_new_tokens):
        if temperature <= 0.0:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Temperature scaling
            scaled_logits = next_token_logits / temperature

            # Top-K filtering
            if top_k > 0:
                values, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < values[:, [-1]]] = -float("Inf")

            probs = F.softmax(scaled_logits, dim=-1)

            # Top-P filtering or standard sampling
            if top_p < 1.0:
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        # Append generated token
        generated = torch.cat([generated, next_token], dim=1)

        # Check early stopping condition
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # Forward pass for only the SINGLE new token using KV caches
        logits, _, kv_caches = model(
            input_ids=next_token,
            kv_caches=kv_caches,
            start_pos=cur_pos,
            use_cache=True,
        )
        next_token_logits = logits[:, -1, :]
        cur_pos += 1

    return generated
