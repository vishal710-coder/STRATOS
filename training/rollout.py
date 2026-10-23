"""Rollout Buffer and Generalized Advantage Estimation (GAE) for Multi-Agent PPO (v2)."""

from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import torch


class MultiAgentRolloutBuffer:
    """
    Stores multi-agent rollout trajectory data and computes GAE advantages.
    Supports both single-head and dual-head (action + learned message) PPO mini-batches.
    """

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Allocate memory tensors
        self.obs = torch.zeros((buffer_size, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, num_agents), dtype=torch.long, device=device)
        self.messages = torch.zeros((buffer_size, num_agents), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.hidden_facts = torch.zeros((buffer_size, num_agents, 2), dtype=torch.long, device=device)

        self.step = 0
        self.actual_size = buffer_size

    def insert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        message: Optional[torch.Tensor] = None,
        hidden_fact: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert single timestep transition for all agents."""
        assert self.step < self.buffer_size, "Rollout buffer overflow!"
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(action)
        if message is not None:
            self.messages[self.step].copy_(message)
        else:
            self.messages[self.step].zero_()
        if hidden_fact is not None:
            self.hidden_facts[self.step].copy_(hidden_fact)
        else:
            self.hidden_facts[self.step].zero_()
        self.logprobs[self.step].copy_(logprob)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)
        self.step += 1

    def compute_returns_and_advantages(self, next_value: torch.Tensor, next_done: torch.Tensor) -> None:
        """Compute Generalized Advantage Estimation (GAE-Lambda) across rollout."""
        actual_size = self.step if self.step > 0 else self.buffer_size
        self.actual_size = actual_size
        last_gae_lam = torch.zeros((self.num_agents,), device=self.device)

        for t in reversed(range(actual_size)):
            if t == actual_size - 1:
                next_non_terminal = 1.0 - next_done
                next_vals = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_vals = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_vals * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns[:actual_size] = self.advantages[:actual_size] + self.values[:actual_size]
        self.step = 0

    def get_batches(
        self,
        batch_size: int,
        with_messages: bool = False,
        with_aux: bool = False,
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Flatten buffer across (actual_size * num_agents) and yield mini-batches.
        If with_messages=True and with_aux=True, returns (b_obs, b_actions, b_messages, b_logprobs, b_advantages, b_returns, b_hidden_facts).
        If with_messages=True and with_aux=False, returns (b_obs, b_actions, b_messages, b_logprobs, b_advantages, b_returns).
        If with_messages=False, returns (b_obs, b_actions, b_logprobs, b_advantages, b_returns).
        """
        actual_size = getattr(self, "actual_size", self.buffer_size)
        total_samples = actual_size * self.num_agents
        b_obs = self.obs[:actual_size].reshape(total_samples, self.obs_dim)
        b_actions = self.actions[:actual_size].reshape(total_samples)
        b_messages = self.messages[:actual_size].reshape(total_samples)
        b_hidden_facts = self.hidden_facts[:actual_size].reshape(total_samples, 2)
        b_logprobs = self.logprobs[:actual_size].reshape(total_samples)
        b_advantages = self.advantages[:actual_size].reshape(total_samples)
        b_returns = self.returns[:actual_size].reshape(total_samples)

        # Normalize advantages across batch for gradient stability
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std() + 1e-8
        b_advantages = (b_advantages - adv_mean) / adv_std

        indices = np.random.permutation(total_samples)
        for start_idx in range(0, total_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_inds = indices[start_idx:end_idx]
            if with_messages and with_aux:
                yield (
                    b_obs[batch_inds],
                    b_actions[batch_inds],
                    b_messages[batch_inds],
                    b_logprobs[batch_inds],
                    b_advantages[batch_inds],
                    b_returns[batch_inds],
                    b_hidden_facts[batch_inds],
                )
            elif with_messages:
                yield (
                    b_obs[batch_inds],
                    b_actions[batch_inds],
                    b_messages[batch_inds],
                    b_logprobs[batch_inds],
                    b_advantages[batch_inds],
                    b_returns[batch_inds],
                )
            else:
                yield (
                    b_obs[batch_inds],
                    b_actions[batch_inds],
                    b_logprobs[batch_inds],
                    b_advantages[batch_inds],
                    b_returns[batch_inds],
                )
