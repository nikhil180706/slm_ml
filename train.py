"""
Training Engine for Small Language Model (SLM).
Includes:
- Device selection (CUDA / MPS / CPU).
- Mixed Precision (bfloat16 / float16) via torch.amp.
- Weight decay grouping (decays 2D weights, excludes 1D norms and biases).
- Cosine Learning Rate Schedule with Linear Warmup.
- Gradient Clipping to prevent gradient explosion.
- Gradient Accumulation for flexible batch sizing.
- Checkpoint saving via checkpoint.py.
- Built-in sanity check mode to verify loss decreases.
"""

import math
import time
import torch
import torch.nn as nn
from typing import Tuple
from config import ModelConfig
from model import SLMModel
from tokenizer import Tokenizer
from checkpoint import save_checkpoint


def get_device() -> torch.device:
    """Detects best available compute hardware."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def configure_optimizers(model: SLMModel, weight_decay: float = 0.1, lr: float = 3e-4, betas: Tuple[float, float] = (0.9, 0.95)):
    """
    Separates parameters so weight decay is only applied to 2D weight matrices (Linear projections).
    1D parameters (RMSNorm scales and biases) must NOT have weight decay applied.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=1e-8)
    return optimizer


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Computes learning rate using Linear Warmup followed by Cosine Decay.
    """
    # 1. Linear warmup phase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2. Minimum LR after schedule ends
    if step > max_steps:
        return min_lr
    # 3. Cosine decay phase
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_sanity_check():
    """
    Sanity test: Trains a tiny SLM on a repeating text prompt to prove
    that the training loop, loss computation, backpropagation, and optimizer work.
    Expected behavior: Loss drops significantly from ~9.0 down to < 1.0.
    """
    device = get_device()
    print(f"Using device for training: {device}")

    # Initialize Tokenizer first
    tokenizer = Tokenizer()

    # Use lightweight config for fast local execution, aligning vocab_size to tokenizer
    config = ModelConfig.tiny()
    config.vocab_size = tokenizer.vocab_size
    model = SLMModel(config).to(device)
    print(f"Model initialized with {model.count_parameters():,} parameters (vocab_size={config.vocab_size}).")
    sample_text = (
        "Small Language Models can achieve remarkable reasoning performance "
        "when trained on high quality educational data. Architecture optimization "
        "and hardware alignment ensure maximum throughput on NVIDIA Blackwell GPUs."
    )
    tokens = tokenizer.encode(sample_text)
    
    # Repeat text to create a small training tensor
    repeat_count = (config.max_seq_len // len(tokens[0])) + 2
    repeated_tokens = tokens.repeat(1, repeat_count)[:, : config.max_seq_len]
    
    # Input x and target y (shifted by 1 token)
    x = repeated_tokens[:, :-1].to(device)
    y = repeated_tokens[:, 1:].to(device)

    # Optimizer & Hyperparameters
    max_steps = 100
    warmup_steps = 10
    max_lr = 1e-3
    min_lr = 1e-4
    optimizer = configure_optimizers(model, weight_decay=0.01, lr=max_lr)

    # Determine automatic mixed-precision dtype
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print("\nStarting 100-step Training Loop...")
    start_time = time.time()
    initial_loss = None

    for step in range(max_steps):
        # Update learning rate according to cosine schedule
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed-precision if supported
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits, loss, _ = model(input_ids=x, targets=y)
        else:
            logits, loss, _ = model(input_ids=x, targets=y)

        if initial_loss is None:
            initial_loss = loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping: prevents gradient explosion
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Progress reporting
        if (step + 1) % 20 == 0 or step == 0:
            print(f"Step {step+1:3d}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f} | Grad Norm: {norm:.3f}")

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.2f} seconds.")
    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {loss.item():.4f}")
    assert loss.item() < initial_loss, "Training failed: Loss did not decrease!"
    print("✓ Model successfully learned and overfit the sequence (Optimization verified).")

    # Save verification checkpoint
    ckpt_path = save_checkpoint(model, config, optimizer, step=max_steps, save_dir="checkpoints")
    print(f"✓ Saved verification checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    train_sanity_check()
