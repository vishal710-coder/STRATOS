"""Reward calculation system for STRATOS AI multi-agent reinforcement learning (v2).

Team-level shaped reward for enemy damage/kills, own base HP, friendly survival,
plus a small step penalty. No role-position hacks. No resource-hoarding incentive.
"""

from typing import Dict, Any
from environment.game_state import GameState


class RewardCalculator:
    """Calculates team-level shaped rewards for all agents."""

    def __init__(self):
        # Terminal
        self.win_reward = 100.0
        self.loss_penalty = -100.0
        self.draw_reward = -10.0  # small penalty to discourage passive timeouts

        # Shaping weights (team-level, applied to all agents equally)
        self.enemy_base_dmg_scale = 0.08
        self.friendly_base_dmg_scale = -0.06
        self.enemy_unit_dmg_scale = 0.10
        self.friendly_unit_dmg_scale = -0.08
        self.enemy_unit_destroyed_reward = 25.0
        self.friendly_unit_lost_penalty = -20.0
        self.step_penalty = -0.02
        self.survival_bonus = 0.02  # per alive friendly unit per step

    def compute_rewards(
        self,
        prev_state: GameState,
        curr_state: GameState,
        step_events: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute identical team reward for each agent role."""
        reward = self.step_penalty

        # Terminal rewards
        if curr_state.terminated or curr_state.truncated:
            if curr_state.winner == "friendly":
                reward += self.win_reward
            elif curr_state.winner == "enemy":
                reward += self.loss_penalty
            elif curr_state.winner == "draw":
                reward += self.draw_reward

        # Damage events
        enemy_base_dmg = step_events.get("enemy_base_damage", 0.0)
        friendly_base_dmg = step_events.get("friendly_base_damage", 0.0)
        enemy_unit_dmg = step_events.get("enemy_unit_damage", 0.0)
        friendly_unit_dmg = step_events.get("friendly_unit_damage", 0.0)

        enemy_units_killed = step_events.get("enemy_units_killed", 0)
        friendly_units_lost = step_events.get("friendly_units_lost", 0)

        # Team performance shaping
        reward += (
            enemy_base_dmg * self.enemy_base_dmg_scale
            + friendly_base_dmg * self.friendly_base_dmg_scale
            + enemy_unit_dmg * self.enemy_unit_dmg_scale
            + friendly_unit_dmg * self.friendly_unit_dmg_scale
            + enemy_units_killed * self.enemy_unit_destroyed_reward
            + friendly_units_lost * self.friendly_unit_lost_penalty
        )

        # Survival incentive: bonus per alive friendly unit
        alive_count = sum(1 for u in curr_state.friendly_units.values() if u.is_alive)
        reward += alive_count * self.survival_bonus

        # Identical reward for all agents (team reward, no role-specific hacks)
        rewards: Dict[str, float] = {
            "commander": reward,
            "defense": reward,
            "attack": reward,
        }
        return rewards
