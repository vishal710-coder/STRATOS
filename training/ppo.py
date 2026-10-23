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
        comm_entropy_coef: Optional[float] = None,
        aux_loss_coef: float = 0.0,
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
        self.comm_entropy_coef = comm_entropy_coef if comm_entropy_coef is not None else entropy_coef
        self.aux_loss_coef = aux_loss_coef
        self.max_grad_norm = max_grad_norm

    def set_learning_rate(self, lr: float) -> None:
        """Update optimizer learning rate (for linear annealing)."""
        self.current_lr = max(1e-7, lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

    def set_entropy_coef(self, coef: float, comm_coef: Optional[float] = None) -> None:
        """Update entropy regularization coefficient."""
        self.entropy_coef = coef
        if comm_coef is not None:
            self.comm_entropy_coef = comm_coef

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
        comm_entropies = []
        aux_losses = []
        approx_kls = []

        has_comm = getattr(self.policy, "has_comm_head", False)
        use_aux = (self.aux_loss_coef > 0.0) and getattr(self.policy, "has_aux_head", False)

        for _ in range(ppo_epochs):
            for batch in buffer.get_batches(batch_size, with_messages=has_comm, with_aux=use_aux):
                if has_comm:
                    if use_aux:
                        b_obs, b_actions, b_messages, b_old_logprobs, b_advantages, b_returns, b_hidden = batch
                    else:
                        b_obs, b_actions, b_messages, b_old_logprobs, b_advantages, b_returns = batch
                        b_hidden = None

                    act_logp, msg_logp, act_ent, msg_ent, new_values, aux_logits = (
                        self.policy.evaluate_actions_and_aux(b_obs, b_actions, b_messages)
                    )
                    new_logprobs = act_logp + msg_logp
                    entropy_loss = self.entropy_coef * act_ent.mean() + self.comm_entropy_coef * msg_ent.mean()
                    comm_entropies.append(msg_ent.mean().item())
                else:
                    b_obs, b_actions, b_old_logprobs, b_advantages, b_returns = batch
                    _, new_logprobs, ent, new_values = self.policy.get_action_and_value(b_obs, b_actions)
                    entropy_loss = self.entropy_coef * ent.mean()
                    aux_logits = None
                    b_hidden = None

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

                total_loss = policy_loss + self.value_coef * value_loss - entropy_loss

                # Auxiliary fact prediction loss
                if use_aux and aux_logits is not None and b_hidden is not None:
                    loss_weak = torch.nn.functional.cross_entropy(aux_logits[:, :3], b_hidden[:, 0])
                    loss_flank = torch.nn.functional.cross_entropy(aux_logits[:, 3:6], b_hidden[:, 1])
                    aux_l = loss_weak + loss_flank
                    total_loss = total_loss + self.aux_loss_coef * aux_l
                    aux_losses.append(aux_l.item())

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
            "comm_entropy": float(sum(comm_entropies) / max(1, len(comm_entropies))) if comm_entropies else 0.0,
            "aux_loss": float(sum(aux_losses) / max(1, len(aux_losses))) if aux_losses else 0.0,
            "approx_kl": float(sum(approx_kls) / max(1, len(approx_kls))),
            "learning_rate": self.current_lr,
        }
