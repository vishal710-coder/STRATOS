"""Commander Agent for STRATOS AI."""

from typing import Optional, Tuple
import numpy as np
import torch
from agents.policies import ActorCriticPolicy
from environment.actions import ActionType


class CommanderAgent:
    """
    Commander Agent:
    - Assesses overarching battlefield situation and resource state.
    - Decides macro-level actions using learned PPO policy.
    - Emits high-level tactical directives via coordinator.
    """

    def __init__(self, policy: Optional[ActorCriticPolicy] = None, role: str = "commander"):
        self.role = role
        self.policy = policy
        self.last_action: int = ActionType.HOLD
        self.decision_mode: str = "learned_policy" if policy is not None else "fallback_rule"
        self.current_objective: str = "Defend Territory & Direct Squad"

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
        """Heuristic fallback when no neural checkpoint is loaded."""
        # Check threat level from communication channels (obs[44])
        threat_level = obs[44]
        friendly_base_hp = obs[7]

        if threat_level > 0.4 or friendly_base_hp < 0.8:
            return ActionType.DEFEND_PERIMETER
        # Check resources
        friendly_res = obs[40]
        if friendly_res < 0.3:
            return ActionType.MOVE_RESOURCE
        return ActionType.ATTACK_NEAREST_ENEMY
