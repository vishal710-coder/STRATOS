"""Action definitions for STRATOS AI agents."""

from enum import IntEnum
from typing import Dict, Any


class ActionType(IntEnum):
    """Discrete tactical action primitives for agents."""
    HOLD = 0                  # Remain stationary, conserving position
    MOVE_BASE = 1             # Move directly toward friendly base
    MOVE_ENEMY_BASE = 2       # Move directly toward enemy base
    MOVE_RESOURCE = 3         # Move to nearest active resource depot
    ATTACK_NEAREST_ENEMY = 4   # Engage nearest visible enemy combat unit
    DEFEND_PERIMETER = 5      # Patrol / defend friendly perimeter zone
    TACTICAL_RETREAT = 6      # Move away from highest threat concentration


ACTION_NAMES = {
    ActionType.HOLD: "Hold Position",
    ActionType.MOVE_BASE: "Fallback to Base",
    ActionType.MOVE_ENEMY_BASE: "Siege Enemy Base",
    ActionType.MOVE_RESOURCE: "Harvest Resource",
    ActionType.ATTACK_NEAREST_ENEMY: "Engage Enemy Unit",
    ActionType.DEFEND_PERIMETER: "Defend Perimeter",
    ActionType.TACTICAL_RETREAT: "Tactical Retreat",
}

NUM_ACTIONS = len(ActionType)
