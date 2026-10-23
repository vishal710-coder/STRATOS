"""Actor-Critic Neural Network Policies for STRATOS AI (v2 — Learned Multi-Agent Communication)."""

from typing import Dict, Optional, Tuple
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
    Multi-Agent Actor-Critic Network with Dual Action Heads:
    1. Movement / Combat action head (7 discrete actions)
    2. Communication message head (K+1 discrete tokens, e.g. 9 tokens: 0=silent, 1..8=signals)
    3. State-value estimator V(s)
    """

    def __init__(
        self,
        obs_dim: int = 73,
        act_dim: int = 9,
        msg_dim: int = 9,
        hidden_dim: int = 256,
        has_comm_head: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.has_comm_head = has_comm_head and (msg_dim > 0)

        # Shared feature representation trunk
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        # Action head (movement / combat kinematics)
        self.actor = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)

        # Message head (learned communication token emission)
        if self.has_comm_head:
            self.comm_actor = layer_init(nn.Linear(hidden_dim, msg_dim), std=0.01)
        else:
            self.comm_actor = None

        # Decentralized state value estimator
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value (for backward compatibility)."""
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
        """Standard 1-head API: sample or evaluate action, computing log_prob, entropy, and value."""
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

    def get_actions_and_values(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        message: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dual-head API: sample or evaluate both movement action and communication token.
        Returns:
            (actions, messages, action_log_probs, msg_log_probs, total_entropy, values)
        """
        features = self.network(obs)
        act_logits = self.actor(features)
        act_probs = Categorical(logits=act_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(act_logits, dim=-1)
            else:
                action = act_probs.sample()

        act_log_prob = act_probs.log_prob(action)
        act_entropy = act_probs.entropy()

        if self.has_comm_head and self.comm_actor is not None:
            msg_logits = self.comm_actor(features)
            msg_probs = Categorical(logits=msg_logits)
            if message is None:
                if deterministic:
                    message = torch.argmax(msg_logits, dim=-1)
                else:
                    message = msg_probs.sample()
            msg_log_prob = msg_probs.log_prob(message)
            msg_entropy = msg_probs.entropy()
        else:
            message = torch.zeros_like(action)
            msg_log_prob = torch.zeros_like(act_log_prob)
            msg_entropy = torch.zeros_like(act_entropy)

        total_entropy = act_entropy + msg_entropy
        value = self.critic(features)

        return action, message, act_log_prob, msg_log_prob, total_entropy, value


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for MAPPO (Multi-Agent PPO).
    Takes concatenated joint observations or global environment state to compute V_tot(s).
    """

    def __init__(self, state_dim: int = 67 * 3, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Evaluate centralized state value V(s_global)."""
        return self.network(global_state)
