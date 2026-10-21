"""Attack Agent for STRATOS AI."""

from typing import Optional
import numpy as np
import torch
from agents.policies import ActorCriticPolicy
from environment.actions import ActionType


class AttackAgent:
    """
    Attack Agent:
    - High-mobility offensive strike unit.
    - Seeks out tactical engagements and targets enemy structures.
    - Incorporates squad communication to coordinate flanking or fallback.
    """

    def __init__(self, policy: Optional[ActorCriticPolicy] = None, role: str = "attack"):
        self.role = role
        self.policy = policy
        self.last_action: int = ActionType.MOVE_ENEMY_BASE
        self.decision_mode: str = "learned_policy" if policy is not None else "fallback_rule"
        self.current_objective: str = "Assault Enemy Stronghold"

    def set_policy(self, policy: ActorCriticPolicy) -> None:
        self.policy = policy
        self.decision_mode = "learned_policy"

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select action via neural policy or rule fallback."""
        if self.policy is not None:
            self.decision_mode = "learned_policy"
            with torch.no_grad():
                obs_t = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                action_t, _, _, _ = self.policy.get_action_and_value(obs_t, deterministic=deterministic)
                action = int(action_t.item())
        else:
            self.decision_mode = "fallback_rule"
            action = self._heuristic_action(observation)

        self.last_action = action
        return action

    def _heuristic_action(self, obs: np.ndarray) -> int:
        """Heuristic fallback when evaluating without neural weights."""
        hp_ratio = obs[2]
        support_requested = obs[47]  # Communication support request channel

        if hp_ratio < 0.2:
            return ActionType.TACTICAL_RETREAT
        # If defense agent broadcasted emergency support, pivot to assist
        if support_requested > 0.5:
            return ActionType.ATTACK_NEAREST_ENEMY
        return ActionType.MOVE_ENEMY_BASE
