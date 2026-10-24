"""Reward calculation system for STRATOS AI multi-agent reinforcement learning (v3).

Clean, potential-based shaping with strict separation of terminal outcomes:
Win (+200) >> Draw (-30) > Loss (-100).
Stalling is strictly penalized via a step penalty (-0.05) and draw penalty (-30).
No passivity subsidies or survival-time bonuses.
"""

from typing import Dict, Any
from environment.game_state import GameState


class RewardCalculator:
    """Calculates team-level shaped rewards for all agents."""

    def __init__(self):
        # Terminal outcomes (clearly separated: Win >> Draw > Loss)
        self.win_reward = 200.0      # +600 team sum for genuine victory
        self.loss_penalty = -100.0   # -300 team sum for base destruction / wipeout
        self.draw_reward = -30.0     # -90 team sum for timeout draw (stalling penalized)

        # Step penalty to encourage decisive, efficient action
        self.step_penalty = -0.05

        # Potential-based shaping for spatial progress toward enemy base
        self.potential_scale = 0.5

        # Combat shaping weights (team-level, symmetric)
        self.enemy_base_dmg_scale = 0.08
        self.friendly_base_dmg_scale = -0.08
        self.enemy_unit_dmg_scale = 0.05
        self.friendly_unit_dmg_scale = -0.05
        self.enemy_unit_destroyed_reward = 15.0
        self.friendly_unit_lost_penalty = -15.0

        # Team cohesion shaping (behind config flag)
        self.enable_team_cohesion = False
        self.cohesion_dist = 120.0
        self.cohesion_bonus = 0.05

    def compute_rewards(
        self,
        prev_state: GameState,
        curr_state: GameState,
        step_events: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute identical team reward for each agent role."""
        reward = self.step_penalty

        # 1. Terminal rewards
        if curr_state.terminated or curr_state.truncated:
            if curr_state.winner == "friendly":
                reward += self.win_reward
            elif curr_state.winner == "enemy":
                reward += self.loss_penalty
            elif curr_state.winner == "draw":
                reward += self.draw_reward

        # 2. Damage and elimination events
        enemy_base_dmg = step_events.get("enemy_base_damage", 0.0)
        friendly_base_dmg = step_events.get("friendly_base_damage", 0.0)
        enemy_unit_dmg = step_events.get("enemy_unit_damage", 0.0)
        friendly_unit_dmg = step_events.get("friendly_unit_damage", 0.0)

        enemy_units_killed = step_events.get("enemy_units_killed", 0)
        friendly_units_lost = step_events.get("friendly_units_lost", 0)

        reward += (
            enemy_base_dmg * self.enemy_base_dmg_scale
            + friendly_base_dmg * self.friendly_base_dmg_scale
            + enemy_unit_dmg * self.enemy_unit_dmg_scale
            + friendly_unit_dmg * self.friendly_unit_dmg_scale
            + enemy_units_killed * self.enemy_unit_destroyed_reward
            + friendly_units_lost * self.friendly_unit_lost_penalty
        )

        # 3. Potential-based progress shaping toward enemy objective
        eb = curr_state.bases.get("enemy")
        prev_eb = prev_state.bases.get("enemy")
        if eb and prev_eb:
            map_diag = (curr_state.map_width**2 + curr_state.map_height**2)**0.5
            curr_phi = sum(
                (1.0 - u.distance_to(eb.x, eb.y) / map_diag)
                for u in curr_state.friendly_units.values() if u.is_alive
            )
            prev_phi = sum(
                (1.0 - u.distance_to(prev_eb.x, prev_eb.y) / map_diag)
                for u in prev_state.friendly_units.values() if u.is_alive
            )
            reward += (curr_phi - prev_phi) * self.potential_scale

        # 4. Configurable team cohesion bonus (when enemies are in view of any friendly unit)
        if self.enable_team_cohesion:
            alive_f = [u for u in curr_state.friendly_units.values() if u.is_alive]
            alive_e = [e for e in curr_state.enemy_units.values() if e.is_alive]
            # Check if any enemy is visible to any friendly unit
            enemy_visible = any(any(u.can_see(e.x, e.y) for e in alive_e) for u in alive_f)
            if enemy_visible and len(alive_f) >= 2:
                # Check if at least two friendly units are within cohesion_dist of each other
                is_cohesive = False
                for i in range(len(alive_f)):
                    for j in range(i + 1, len(alive_f)):
                        if alive_f[i].distance_to(alive_f[j].x, alive_f[j].y) <= self.cohesion_dist:
                            is_cohesive = True
                            break
                    if is_cohesive:
                        break
                if is_cohesive:
                    reward += self.cohesion_bonus

        # Identical reward for all agents (pure team reward)
        rewards: Dict[str, float] = {
            "commander": reward,
            "defense": reward,
            "attack": reward,
        }
        return rewards
