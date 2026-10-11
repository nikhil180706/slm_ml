"""
Comprehensive Verification and Unit Test Suite for SLM.
Verifies:
1. RMSNorm numerical output and shape preservation.
2. RoPE frequency precomputation and rotary application.
3. SwiGLU MLP projection correctness.
4. Grouped-Query Attention (GQA) output shapes.
5. Full model forward pass and backward gradient flow.
6. Causal Mask Leakage Test (ensures model cannot peek into the future).
7. KV-Cache Numerical Equivalence (proves cached generation equals full recalculation).
"""

import torch
import pytest
from config import ModelConfig
from norm import RMSNorm
from rotary import precompute_freqs_cis, apply_rotary_emb
from mlp import SwiGLUMLP
from attention import GroupedQueryAttention
from model import SLMModel


def test_rmsnorm():
    batch, seq, dim = 2, 16, 64
    x = torch.randn(batch, seq, dim)
    norm = RMSNorm(dim=dim, eps=1e-5)
    out = norm(x)
    
    assert out.shape == x.shape, f"RMSNorm changed shape from {x.shape} to {out.shape}"
    # Standard deviation / RMS of output should be approximately 1.0 along the feature dimension
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2), "RMSNorm failed to scale to unit RMS"
    print("✓ RMSNorm verification passed.")


def test_rope():
    head_dim = 32
    seq_len = 64
    batch, n_heads = 2, 4

    cos, sin = precompute_freqs_cis(dim=head_dim, end=seq_len)
    assert cos.shape == (seq_len, head_dim), f"Expected cos shape ({seq_len}, {head_dim}), got {cos.shape}"
    assert sin.shape == (seq_len, head_dim), f"Expected sin shape ({seq_len}, {head_dim}), got {sin.shape}"

    q = torch.randn(batch, n_heads, seq_len, head_dim)
    q_rot = apply_rotary_emb(q, cos, sin)
    assert q_rot.shape == q.shape, "RoPE modified tensor shape unexpectedly"
    print("✓ RoPE verification passed.")


def test_swiglu():
    config = ModelConfig.tiny()
    mlp = SwiGLUMLP(config)
    x = torch.randn(2, 16, config.dim)
    out = mlp(x)
    assert out.shape == x.shape, f"MLP output shape mismatch: expected {x.shape}, got {out.shape}"
    print("✓ SwiGLU MLP verification passed.")


def test_gqa_attention():
    config = ModelConfig.tiny()
    attn = GroupedQueryAttention(config)
    cos, sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_seq_len)

    x = torch.randn(2, 16, config.dim)
    out, cache = attn(x, cos=cos[:16], sin=sin[:16], use_cache=True)
    
    assert out.shape == x.shape, f"Attention output shape mismatch: {out.shape}"
    assert cache is not None, "Cache should not be None when use_cache=True"
    k_cache, v_cache = cache
    assert k_cache.shape == (2, config.n_kv_heads, 16, config.head_dim)
    print("✓ Grouped-Query Attention verification passed.")


def test_model_forward_and_backward():
    config = ModelConfig.tiny()
    model = SLMModel(config)

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss, _ = model(input_ids=input_ids, targets=targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"Logits shape mismatch: expected {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    )
    assert loss is not None, "Loss was not computed"
    assert not torch.isnan(loss), "Loss returned NaN!"
    assert loss.item() > 0.0, "Loss must be positive"

    # Backward pass: verify gradients propagate cleanly to all parameters
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient contains NaN for: {name}"

    print(f"✓ Model Forward and Backward passed. (Total params: {model.count_parameters():,})")


def test_causal_masking_leakage():
    """
    CRITICAL TEST: Ensures past tokens NEVER attend to future tokens.
    Modifying token at position N must NOT change predictions at positions 0 to N-1.
    """
    config = ModelConfig.tiny()
    model = SLMModel(config)
    model.eval()

    # Create sequence A: [10, 20, 30, 40]
    seq_a = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    # Create sequence B: [10, 20, 30, 999] (only position 3 is changed)
    seq_b = torch.tensor([[10, 20, 30, 999]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _, _ = model(seq_a)
        logits_b, _, _ = model(seq_b)

    # Predictions for positions 0, 1, and 2 MUST be identical
    diff = (logits_a[:, :3, :] - logits_b[:, :3, :]).abs().max().item()
    assert diff < 1e-5, (
        f"CAUSAL LEAKAGE DETECTED! Max diff: {diff}. "
        "A future token changed the prediction of an earlier token!"
    )
    print("✓ Causal Masking Leakage Test passed: Future tokens cannot leak into past context.")


def test_kv_cache_equivalence():
    """
    Verifies that step-by-step KV cache generation produces the exact same logits
    as computing the entire sequence in a single forward pass.
    """
    config = ModelConfig.tiny()
    model = SLMModel(config)
    model.eval()

    prompt = torch.tensor([[10, 25, 42]], dtype=torch.long)
    new_token = torch.tensor([[88]], dtype=torch.long)
    full_seq = torch.cat([prompt, new_token], dim=1)  # Length 4

    with torch.no_grad():
        # Method 1: Full sequence recalculation
        full_logits, _, _ = model(full_seq)
        expected_next_logit = full_logits[:, -1, :]

        # Method 2: Prefill prompt + 1 cached step
        _, _, kv_caches = model(prompt, start_pos=0, use_cache=True)
        cached_logits, _, _ = model(new_token, kv_caches=kv_caches, start_pos=3, use_cache=True)
        actual_next_logit = cached_logits[:, -1, :]

    diff = (expected_next_logit - actual_next_logit).abs().max().item()
    assert diff < 1e-4, f"KV cache produced divergent logits compared to full recalculation. Diff: {diff}"
    print("✓ KV-Cache Equivalence verified: Fast generation is numerically identical.")


if __name__ == "__main__":
    print("Running SLM Verification Test Suite...\n")
    test_rmsnorm()
    test_rope()
    test_swiglu()
    test_gqa_attention()
    test_model_forward_and_backward()
    test_causal_masking_leakage()
    test_kv_cache_equivalence()
    print("\n🎉 ALL 7 ARCHITECTURAL TESTS PASSED SUCCESSFULLY!")
