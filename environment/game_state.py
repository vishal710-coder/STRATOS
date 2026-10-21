"""Game state and entity models for STRATOS AI."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Unit:
    """Combat or support unit in the STRATOS simulation."""
    id: str
    team: str  # "friendly" or "enemy"
    role: str  # "commander", "defense", "attack", "enemy_grunt"
    name: str
    x: float
    y: float
    hp: float
    max_hp: float
    speed: float
    damage: float
    attack_range: float
    vision_range: float
    cooldown: int = 5
    cooldown_timer: int = 0
    is_alive: bool = True
    resources_held: float = 0.0
    kills: int = 0
    damage_dealt: float = 0.0
    current_action: int = 0
    target_id: Optional[str] = None

    def distance_to(self, target_x: float, target_y: float) -> float:
        """Compute Euclidean distance to coordinate."""
        return math.hypot(self.x - target_x, self.y - target_y)

    def move_towards(self, target_x: float, target_y: float, map_w: float, map_h: float) -> None:
        """Move towards a target coordinate at unit speed, clamped to boundaries."""
        dist = self.distance_to(target_x, target_y)
        if dist > 1e-4:
            step = min(self.speed, dist)
            dx = (target_x - self.x) / dist * step
            dy = (target_y - self.y) / dist * step
            self.x = max(10.0, min(map_w - 10.0, self.x + dx))
            self.y = max(10.0, min(map_h - 10.0, self.y + dy))

    def move_away_from(self, threat_x: float, threat_y: float, map_w: float, map_h: float) -> None:
        """Move away from threat coordinate at unit speed, clamped to boundaries."""
        dist = self.distance_to(threat_x, threat_y)
        if dist > 1e-4:
            dx = -(threat_x - self.x) / dist * self.speed
            dy = -(threat_y - self.y) / dist * self.speed
            self.x = max(10.0, min(map_w - 10.0, self.x + dx))
            self.y = max(10.0, min(map_h - 10.0, self.y + dy))


@dataclass
class Base:
    """Headquarters base structure."""
    team: str  # "friendly" or "enemy"
    x: float
    y: float
    hp: float
    max_hp: float = 1000.0
    is_alive: bool = True

    def distance_to(self, target_x: float, target_y: float) -> float:
        return math.hypot(self.x - target_x, self.y - target_y)


@dataclass
class ResourceNode:
    """Harvestable crystal/energy deposit."""
    id: int
    x: float
    y: float
    amount: float
    max_amount: float = 500.0
    regen_rate: float = 1.0


@dataclass
class GameState:
    """Complete simulation snapshot at any step."""
    step_count: int = 0
    max_steps: int = 300
    map_width: float = 800.0
    map_height: float = 600.0

    bases: Dict[str, Base] = field(default_factory=dict)
    friendly_units: Dict[str, Unit] = field(default_factory=dict)
    enemy_units: Dict[str, Unit] = field(default_factory=dict)
    resource_nodes: List[ResourceNode] = field(default_factory=list)

    friendly_resources: float = 0.0
    enemy_resources: float = 0.0

    winner: Optional[str] = None  # "friendly", "enemy", "draw", or None
    terminated: bool = False
    truncated: bool = False

    def is_game_over(self) -> bool:
        """Check if terminal condition or truncation limit is met."""
        friendly_base = self.bases.get("friendly")
        enemy_base = self.bases.get("enemy")

        if friendly_base and friendly_base.hp <= 0:
            self.winner = "enemy"
            self.terminated = True
            return True

        if enemy_base and enemy_base.hp <= 0:
            self.winner = "friendly"
            self.terminated = True
            return True

        # Check friendly squad wipeout
        friendly_alive = any(u.is_alive for u in self.friendly_units.values())
        enemy_alive = any(u.is_alive for u in self.enemy_units.values())

        if not friendly_alive and not enemy_alive:
            self.winner = "draw"
            self.terminated = True
            return True
        elif not friendly_alive:
            self.winner = "enemy"
            self.terminated = True
            return True
        elif not enemy_alive:
            self.winner = "friendly"
            self.terminated = True
            return True

        if self.step_count >= self.max_steps:
            self.truncated = True
            # Adjudicate winner based on remaining base health & resources
            f_score = (friendly_base.hp if friendly_base else 0) + self.friendly_resources * 0.5
            e_score = (enemy_base.hp if enemy_base else 0) + self.enemy_resources * 0.5
            if f_score > e_score:
                self.winner = "friendly"
            elif e_score > f_score:
                self.winner = "enemy"
            else:
                self.winner = "draw"
            return True

        return False
