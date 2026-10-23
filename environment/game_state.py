"""Game state and entity models for STRATOS AI (v2 — Dynamic & Stochastic)."""

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

    def can_see(self, target_x: float, target_y: float) -> bool:
        """Check if a target position is within this unit's vision radius."""
        return self.is_alive and self.distance_to(target_x, target_y) <= self.vision_range


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
class StormZone:
    """Moving hazard zone that damages units inside."""
    x: float
    y: float
    radius: float
    damage_per_step: float
    drift_dx: float  # px/step drift direction
    drift_dy: float
    spawn_step: int
    end_step: int
    active: bool = True

    def contains(self, px: float, py: float) -> bool:
        return math.hypot(self.x - px, self.y - py) <= self.radius

    def advance(self, map_w: float, map_h: float) -> None:
        """Drift storm one step, bouncing off map edges."""
        self.x += self.drift_dx
        self.y += self.drift_dy
        if self.x < self.radius or self.x > map_w - self.radius:
            self.drift_dx = -self.drift_dx
            self.x = max(self.radius, min(map_w - self.radius, self.x))
        if self.y < self.radius or self.y > map_h - self.radius:
            self.drift_dy = -self.drift_dy
            self.y = max(self.radius, min(map_h - self.radius, self.y))


@dataclass
class SupplyDrop:
    """One-time heal pickup at a random location."""
    x: float
    y: float
    heal_fraction: float  # fraction of max_hp to restore
    spawn_step: int
    collected: bool = False
    pickup_range: float = 35.0


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
    win_type: Optional[str] = None  # "base_destroyed", "wipeout", "timeout_draw", None
    terminated: bool = False
    truncated: bool = False

    # Dynamic event state
    storm: Optional[StormZone] = None
    supply_drop: Optional[SupplyDrop] = None
    enemy_strategy: str = "aggressive"  # current active enemy strategy (can switch mid-game)
    next_resource_spawn_step: int = 999999
    enemy_wave_step: int = 999999
    enemy_strategy_switch_step: int = 999999
    supply_drop_step: int = 999999
    storm_spawn_step: int = 999999

    # Hidden asymmetric information
    weak_lane: int = 1                  # 0: Top, 1: Middle, 2: Bottom lane
    wave_flank: int = 0                 # 0: Top, 1: Middle, 2: Bottom flank
    wave_warning_active: bool = False   # True when defense warning sensor is active

    def is_game_over(self) -> bool:
        """Check if terminal condition or truncation limit is met.

        Win conditions:
          1. Enemy base destroyed → friendly wins (base_destroyed)
          2. All enemy units eliminated → friendly wins (wipeout)
        Loss conditions:
          3. Friendly base destroyed → enemy wins
          4. All friendly units eliminated → enemy wins
        Timeout:
          5. max_steps reached → DRAW (no resource-based adjudication)
        """
        friendly_base = self.bases.get("friendly")
        enemy_base = self.bases.get("enemy")

        # Base destruction
        if friendly_base and friendly_base.hp <= 0:
            self.winner = "enemy"
            self.win_type = "base_destroyed"
            self.terminated = True
            return True

        if enemy_base and enemy_base.hp <= 0:
            self.winner = "friendly"
            self.win_type = "base_destroyed"
            self.terminated = True
            return True

        # Squad wipeout
        friendly_alive = any(u.is_alive for u in self.friendly_units.values())
        enemy_alive = any(u.is_alive for u in self.enemy_units.values())

        if not friendly_alive and not enemy_alive:
            self.winner = "draw"
            self.win_type = "wipeout"
            self.terminated = True
            return True
        elif not friendly_alive:
            self.winner = "enemy"
            self.win_type = "wipeout"
            self.terminated = True
            return True
        elif not enemy_alive:
            self.winner = "friendly"
            self.win_type = "wipeout"
            self.terminated = True
            return True

        # Timeout → DRAW (no resource-based adjudication)
        if self.step_count >= self.max_steps:
            self.truncated = True
            self.winner = "draw"
            self.win_type = "timeout_draw"
            return True

        return False
