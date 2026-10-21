"""Actor-Critic Neural Network Policies for STRATOS AI."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
    """Initialize layer weights orthogonally for stable RL dynamics."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticPolicy(nn.Module):
    """
    Compact Multi-Agent Actor-Critic Network.
    Processes normalized 49-dim observations and outputs:
    1. Categorical action distribution logits (7 discrete actions)
    2. Scalar state value estimation V(s)
    """

    def __init__(self, obs_dim: int = 49, act_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared feature representation with layer normalization
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Actor head (action logits)
        self.actor = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)

        # Critic head (state value)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value."""
        features = self.network(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate state value V(s)."""
        features = self.network(obs)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate action and compute log_prob, entropy, and value.
        """
        features = self.network(obs)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(features)

        return action, log_prob, entropy, value
