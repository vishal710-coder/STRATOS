"""Reward calculation system for STRATOS AI multi-agent reinforcement learning."""

from typing import Dict, Any
from environment.game_state import GameState


class RewardCalculator:
    """Calculates granular, dense, and terminal rewards for agents."""

    def __init__(self):
        # Weight constants
        self.win_reward = 100.0
        self.loss_penalty = -100.0
        self.draw_reward = 0.0

        self.enemy_base_dmg_scale = 0.08
        self.friendly_base_dmg_scale = -0.06
        self.enemy_unit_dmg_scale = 0.10
        self.friendly_unit_dmg_scale = -0.08

        self.enemy_unit_destroyed_reward = 25.0
        self.friendly_unit_lost_penalty = -20.0
        self.resource_harvest_scale = 0.05
        self.step_penalty = -0.02
        self.coordination_bonus = 0.10

    def compute_rewards(
        self,
        prev_state: GameState,
        curr_state: GameState,
        step_events: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute individual and team rewards for each agent role."""
        rewards: Dict[str, float] = {
            "commander": self.step_penalty,
            "defense": self.step_penalty,
            "attack": self.step_penalty,
        }

        # Terminal rewards
        if curr_state.terminated or curr_state.truncated:
            if curr_state.winner == "friendly":
                for agent_id in rewards:
                    rewards[agent_id] += self.win_reward
            elif curr_state.winner == "enemy":
                for agent_id in rewards:
                    rewards[agent_id] += self.loss_penalty
            elif curr_state.winner == "draw":
                for agent_id in rewards:
                    rewards[agent_id] += self.draw_reward

        # Damage events
        enemy_base_dmg = step_events.get("enemy_base_damage", 0.0)
        friendly_base_dmg = step_events.get("friendly_base_damage", 0.0)
        enemy_unit_dmg = step_events.get("enemy_unit_damage", 0.0)
        friendly_unit_dmg = step_events.get("friendly_unit_damage", 0.0)

        enemy_units_killed = step_events.get("enemy_units_killed", 0)
        friendly_units_lost = step_events.get("friendly_units_lost", 0)
        resources_harvested = step_events.get("resources_harvested", 0.0)
        coordinated_strikes = step_events.get("coordinated_strikes", 0)

        # Team performance component
        team_reward = (
            enemy_base_dmg * self.enemy_base_dmg_scale
            + friendly_base_dmg * self.friendly_base_dmg_scale
            + enemy_unit_dmg * self.enemy_unit_dmg_scale
            + friendly_unit_dmg * self.friendly_unit_dmg_scale
            + enemy_units_killed * self.enemy_unit_destroyed_reward
            + friendly_units_lost * self.friendly_unit_lost_penalty
            + resources_harvested * self.resource_harvest_scale
        )

        for agent_id in rewards:
            rewards[agent_id] += team_reward

        # Role-specific tailored incentives:
        # Commander: rewards overall resource management and squad survival
        rewards["commander"] += resources_harvested * 0.03
        if friendly_units_lost == 0:
            rewards["commander"] += 0.01

        # Defense: heavily incentivized on base safety and intercepting approaching enemies
        if friendly_base_dmg == 0.0:
            rewards["defense"] += 0.02
        else:
            rewards["defense"] -= friendly_base_dmg * 0.04

        # Perimeter guard incentive: when enemies breach friendly territory (x < 440), reward maintaining base perimeter
        def_unit = curr_state.friendly_units.get("defense")
        if def_unit and def_unit.is_alive:
            invading_enemies = any(e.is_alive and e.x < 440.0 for e in curr_state.enemy_units.values())
            if invading_enemies:
                if def_unit.x < 350.0:
                    rewards["defense"] += 0.04
                elif def_unit.x > 500.0:
                    rewards["defense"] -= 0.04

        # Attack: incentivized on offense, enemy unit kills and base destruction
        rewards["attack"] += enemy_base_dmg * 0.05 + enemy_units_killed * 5.0

        # Coordinated interaction bonus
        if coordinated_strikes > 0:
            rewards["attack"] += coordinated_strikes * self.coordination_bonus
            rewards["defense"] += coordinated_strikes * self.coordination_bonus
            rewards["commander"] += coordinated_strikes * self.coordination_bonus

        return rewards
