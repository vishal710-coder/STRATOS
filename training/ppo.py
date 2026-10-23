"""Proximal Policy Optimization (PPO) Multi-Agent Algorithm with Learned Communication (v2)."""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import Adam
from agents.policies import ActorCriticPolicy
from training.rollout import MultiAgentRolloutBuffer


class PPOTrainer:
    """
    PPO optimization engine supporting dual-head policies (Action + Communication message).
    Includes clipped surrogate objective, value loss, entropy regularization, and LR scheduling.
    """

    def __init__(
        self,
        policy: ActorCriticPolicy,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.policy.to(self.device)

        self.initial_lr = lr
        self.current_lr = lr
        self.optimizer = Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def set_learning_rate(self, lr: float) -> None:
        """Update optimizer learning rate (for linear annealing)."""
        self.current_lr = max(1e-7, lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

    def set_entropy_coef(self, coef: float) -> None:
        """Update entropy regularization coefficient (for entropy annealing)."""
        self.entropy_coef = coef

    def update(
        self,
        buffer: MultiAgentRolloutBuffer,
        batch_size: int = 64,
        ppo_epochs: int = 4,
    ) -> Dict[str, float]:
        """Run PPO gradient updates across rollout buffer."""
        pg_losses = []
        v_losses = []
        entropies = []
        approx_kls = []

        has_comm = getattr(self.policy, "has_comm_head", False)

        for _ in range(ppo_epochs):
            for batch in buffer.get_batches(batch_size, with_messages=has_comm):
                if has_comm:
                    b_obs, b_actions, b_messages, b_old_logprobs, b_advantages, b_returns = batch
                    # Dual-head forward
                    _, _, new_act_logprobs, new_msg_logprobs, entropy, new_values = (
                        self.policy.get_actions_and_values(b_obs, b_actions, b_messages)
                    )
                    new_logprobs = new_act_logprobs + new_msg_logprobs
                else:
                    b_obs, b_actions, b_old_logprobs, b_advantages, b_returns = batch
                    # Single-head forward
                    _, new_logprobs, entropy, new_values = self.policy.get_action_and_value(b_obs, b_actions)

                new_values = new_values.squeeze(-1)

                # Compute importance ratio
                logratio = new_logprobs - b_old_logprobs
                ratio = torch.exp(logratio)

                # Track approximate KL divergence for monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    approx_kls.append(approx_kl.item())

                # Surrogate PPO loss
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Mean Squared Error)
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # Entropy regularization
                entropy_loss = entropy.mean()

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

        return {
            "policy_loss": float(sum(pg_losses) / max(1, len(pg_losses))),
            "value_loss": float(sum(v_losses) / max(1, len(v_losses))),
            "entropy": float(sum(entropies) / max(1, len(entropies))),
            "approx_kl": float(sum(approx_kls) / max(1, len(approx_kls))),
            "learning_rate": self.current_lr,
        }
