"""Checkpoint saving and loading for STRATOS AI models."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import yaml
from agents.policies import ActorCriticPolicy


def save_checkpoint(
    path: str,
    policy: ActorCriticPolicy,
    optimizer: Optional[torch.optim.Optimizer] = None,
    total_steps: int = 0,
    episode: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Save policy weights, optimizer state, and training metrics."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "total_steps": total_steps,
        "episode": episode,
        "metrics": metrics or {},
        "config": config or {},
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
    }

    torch.save(checkpoint_data, target_path)
    return str(target_path)


def load_checkpoint(
    path: str,
    policy: Optional[ActorCriticPolicy] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[ActorCriticPolicy, Dict[str, Any]]:
    """Load policy weights and metadata from checkpoint."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {target_path}")

    checkpoint = torch.load(target_path, map_location=device)

    if policy is None:
        obs_dim = checkpoint.get("obs_dim", 49)
        act_dim = checkpoint.get("act_dim", 7)
        policy = ActorCriticPolicy(obs_dim=obs_dim, act_dim=act_dim).to(device)

    # Check for dimension mismatch on the input layer (e.g., warm-starting 52-dim role policy from 49-dim base)
    ckpt_state = checkpoint["model_state_dict"]
    curr_state = policy.state_dict()
    is_dim_expansion = False
    if "network.0.weight" in ckpt_state and "network.0.weight" in curr_state:
        ckpt_w = ckpt_state["network.0.weight"]
        curr_w = curr_state["network.0.weight"]
        if ckpt_w.shape != curr_w.shape and ckpt_w.shape[0] == curr_w.shape[0]:
            is_dim_expansion = True
            min_cols = min(ckpt_w.shape[1], curr_w.shape[1])
            adapted_w = curr_w.clone()
            adapted_w[:, :min_cols] = ckpt_w[:, :min_cols]
            ckpt_state["network.0.weight"] = adapted_w

    policy.load_state_dict(ckpt_state)
    policy.eval()

    if not is_dim_expansion and optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metadata = {
        "total_steps": checkpoint.get("total_steps", 0),
        "episode": checkpoint.get("episode", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "obs_dim": checkpoint.get("obs_dim", policy.obs_dim),
    }

    return policy, metadata
