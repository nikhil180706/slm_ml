"""
Checkpointing Utilities for SLM Model.
Saves and restores model weights, optimizer states, step counters, and architecture configurations.
"""

import os
import torch
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple
from config import ModelConfig
from model import SLMModel


def save_checkpoint(
    model: SLMModel,
    config: ModelConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    save_dir: str = "checkpoints",
    filename: Optional[str] = None,
) -> str:
    """
    Saves a checkpoint containing model weights, configuration, and optimizer state.
    """
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"slm_step_{step}.pt"
    save_path = os.path.join(save_dir, filename)

    checkpoint = {
        "step": step,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[SLMModel, ModelConfig, int, Optional[Dict[str, Any]]]:
    """
    Loads model and configuration from a saved checkpoint.
    
    Returns:
        model: Restored SLMModel
        config: Restored ModelConfig
        step: Checkpoint step integer
        optimizer_state: Dict or None
    """
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct configuration
    config = ModelConfig(**checkpoint["config"])
    
    # Initialize model with exact config and load weights
    model = SLMModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    step = checkpoint.get("step", 0)
    optimizer_state = checkpoint.get("optimizer_state_dict", None)

    return model, config, step, optimizer_state
