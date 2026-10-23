"""Hand-written competent tactical policy for STRATOS AI.

Acts as a high-performance benchmark:
- Focuses fire on visible hostile units to secure rapid eliminations.
- Pushes aggressively toward enemy base to siege and destroy the objective.
- Defends headquarters when threatened.
"""

import numpy as np
from environment.actions import ActionType


class TargetedCombatPolicy:
    """Hand-written aggressive coordination policy."""

    def select_action(self, agent: str, obs: np.ndarray, rng: np.random.Generator = None) -> int:
        # Obs layout:
        # self status: 0..4 (x, y, hp_ratio, cooldown, is_alive)
        # friendly base: 5..7 (x, y, hp_ratio)
        # enemy base: 8..10 (x, y, hp_ratio)
        # teammates: 11..18
        # visible enemies: 19..30 (3 enemies * 4: rel_x, rel_y, hp, is_present)
        # visible resources: 31..39
        # global: 40..42
        # storm: 43..45

        self_hp = obs[2]
        friendly_base_hp = obs[7]
        enemy_visible = any(obs[19 + i * 4 + 3] > 0.5 for i in range(3))

        # Critical health: brief retreat if under heavy fire
        if self_hp < 0.15:
            return ActionType.TACTICAL_RETREAT

        # If base is in mortal danger, return to defend
        if friendly_base_hp < 0.5 and enemy_visible and agent in ("commander", "defense"):
            return ActionType.ATTACK_NEAREST_ENEMY

        # Focus fire: if any enemy is visible within vision cone, eliminate them
        if enemy_visible:
            return ActionType.ATTACK_NEAREST_ENEMY

        # Otherwise, advance directly to siege the enemy headquarters
        return ActionType.MOVE_ENEMY_BASE
