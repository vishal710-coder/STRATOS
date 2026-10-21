"""PettingZoo Parallel-compatible Multi-Agent Strategy Game Environment."""

import math
from typing import Any, Dict, List, Optional, Tuple
import gymnasium
from gymnasium import spaces
import numpy as np
import yaml

from environment.actions import ActionType, NUM_ACTIONS
from environment.game_state import Base, GameState, ResourceNode, Unit
from environment.rewards import RewardCalculator
from communication.coordinator import TacticalCoordinator
from communication.messages import Message, MessageType


class StrategyGameEnv:
    """
    STRATOS AI Strategy Simulation Environment.
    Follows PettingZoo Parallel API standard for multi-agent reinforcement learning.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "stratos_v1"}

    def __init__(
        self,
        config_path: str = "config/game.yaml",
        coordination_mode: str = "coordinated",
        opponent_strategy: str = "aggressive",
        include_role_id: bool = False,
    ):
        self.config_path = config_path
        self.coordination_mode = coordination_mode
        self.opponent_strategy = opponent_strategy
        self.include_role_id = include_role_id

        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.map_w = float(self.cfg["map"]["width"])
        self.map_h = float(self.cfg["map"]["height"])
        self.max_steps = int(self.cfg["simulation"]["max_steps"])

        self.possible_agents = ["commander", "defense", "attack"]
        self.agents = list(self.possible_agents)

        # Observation dimension: 49 floats without role_id, or 52 with 3-dim role one-hot badge
        self.obs_dim = 52 if include_role_id else 49
        self._obs_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(NUM_ACTIONS)
            for agent in self.possible_agents
        }

        self.reward_calculator = RewardCalculator()
        self.coordinator = TacticalCoordinator(coordination_mode=coordination_mode)
        self.state: Optional[GameState] = None
        self.np_random = np.random.RandomState(42)

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        if options and "opponent_strategy" in options:
            self.opponent_strategy = options["opponent_strategy"]
        if options and "coordination_mode" in options:
            self.coordination_mode = options["coordination_mode"]
            self.coordinator.coordination_mode = self.coordination_mode

        self.agents = list(self.possible_agents)
        self.coordinator.reset()

        # Initialize Bases
        fb_cfg = self.cfg["bases"]["friendly"]
        eb_cfg = self.cfg["bases"]["enemy"]
        bases = {
            "friendly": Base(team="friendly", x=float(fb_cfg["x"]), y=float(fb_cfg["y"]), hp=float(fb_cfg["hp"])),
            "enemy": Base(team="enemy", x=float(eb_cfg["x"]), y=float(eb_cfg["y"]), hp=float(eb_cfg["hp"])),
        }

        # Initialize Resource Nodes
        nodes = []
        for n in self.cfg["resources"]["nodes"]:
            nodes.append(
                ResourceNode(
                    id=int(n["id"]),
                    x=float(n["x"]),
                    y=float(n["y"]),
                    amount=float(n["max_amount"]),
                    max_amount=float(n["max_amount"]),
                    regen_rate=float(n["regen_rate"]),
                )
            )

        # Initialize Friendly Units
        friendly_units = {}
        for role in ["commander", "defense", "attack"]:
            u_cfg = self.cfg["units"][role]
            friendly_units[role] = Unit(
                id=role,
                team="friendly",
                role=role,
                name=u_cfg["name"],
                x=float(u_cfg["start_pos"][0]),
                y=float(u_cfg["start_pos"][1]),
                hp=float(u_cfg["hp"]),
                max_hp=float(u_cfg["max_hp"]),
                speed=float(u_cfg["speed"]),
                damage=float(u_cfg["damage"]),
                attack_range=float(u_cfg["attack_range"]),
                vision_range=float(u_cfg["vision_range"]),
                cooldown=int(u_cfg["cooldown"]),
            )

        # Initialize Enemy Units
        enemy_units = {}
        e_defaults = self.cfg["enemy_squad"]["unit_defaults"]
        e_positions = self.cfg["enemy_squad"]["initial_positions"]
        for i, pos in enumerate(e_positions):
            eid = f"enemy_{i+1}"
            enemy_units[eid] = Unit(
                id=eid,
                team="enemy",
                role="enemy_grunt",
                name=f"Enemy_{i+1}",
                x=float(pos[0]),
                y=float(pos[1]),
                hp=float(e_defaults["hp"]),
                max_hp=float(e_defaults["max_hp"]),
                speed=float(e_defaults["speed"]),
                damage=float(e_defaults["damage"]),
                attack_range=float(e_defaults["attack_range"]),
                vision_range=float(e_defaults["vision_range"]),
                cooldown=int(e_defaults["cooldown"]),
            )

        self.state = GameState(
            step_count=0,
            max_steps=self.max_steps,
            map_width=self.map_w,
            map_height=self.map_h,
            bases=bases,
            friendly_units=friendly_units,
            enemy_units=enemy_units,
            resource_nodes=nodes,
            friendly_resources=0.0,
            enemy_resources=0.0,
            winner=None,
            terminated=False,
            truncated=False,
        )

        obs = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Advance simulation by 1 discrete step."""
        assert self.state is not None, "Call reset() before step()."
        self.state.step_count += 1
        step_events: Dict[str, Any] = {
            "enemy_base_damage": 0.0,
            "friendly_base_damage": 0.0,
            "enemy_unit_damage": 0.0,
            "friendly_unit_damage": 0.0,
            "enemy_units_killed": 0,
            "friendly_units_lost": 0,
            "resources_harvested": 0.0,
            "coordinated_strikes": 0,
        }

        # 1. Update Cooldowns & Regen
        for u in list(self.state.friendly_units.values()) + list(self.state.enemy_units.values()):
            if u.cooldown_timer > 0:
                u.cooldown_timer -= 1
        for node in self.state.resource_nodes:
            node.amount = min(node.max_amount, node.amount + node.regen_rate)

        # 2. Execute Friendly Actions
        for agent_id, action_idx in actions.items():
            if agent_id in self.state.friendly_units:
                unit = self.state.friendly_units[agent_id]
                if unit.is_alive:
                    unit.current_action = action_idx
                    self._apply_unit_action(unit, action_idx)

        # 3. Autonomous Communication Layer (triggers messages based on dynamic events)
        self._process_autonomous_communication()

        # 4. Execute Opponent Strategy AI
        self._execute_opponent_behavior()

        # 5. Resolve Combat Engagements
        combat_events = self._resolve_combat()
        for k, v in combat_events.items():
            step_events[k] = step_events.get(k, 0.0) + v

        # 6. Resolve Resource Harvesting
        harvest_val = self._resolve_harvesting()
        step_events["resources_harvested"] = harvest_val

        # 7. Decay Coordinator transient alerts
        self.coordinator.step_decay()

        # 8. Check Game Over
        is_over = self.state.is_game_over()

        # 9. Compute Rewards
        rewards = self.reward_calculator.compute_rewards(self.state, self.state, step_events)

        terminations = {agent: self.state.terminated for agent in self.possible_agents}
        truncations = {agent: self.state.truncated for agent in self.possible_agents}

        # Remove dead agents from active agents list if dead, but preserve terminal signal
        for agent in self.possible_agents:
            if not self.state.friendly_units[agent].is_alive and agent in self.agents:
                pass  # Keep in list until episode ends for uniform rollout handling

        obs = {agent: self._get_observation(agent) for agent in self.possible_agents}
        infos = {
            agent: {
                "step": self.state.step_count,
                "winner": self.state.winner,
                "friendly_base_hp": self.state.bases["friendly"].hp,
                "enemy_base_hp": self.state.bases["enemy"].hp,
                "friendly_resources": self.state.friendly_resources,
            }
            for agent in self.possible_agents
        }

        return obs, rewards, terminations, truncations, infos

    def _apply_unit_action(self, unit: Unit, action_idx: int) -> None:
        """Translate discrete action index into unit kinematics."""
        action = ActionType(action_idx)
        fb = self.state.bases["friendly"]
        eb = self.state.bases["enemy"]

        if action == ActionType.HOLD:
            pass  # Stationary

        elif action == ActionType.MOVE_BASE:
            unit.move_towards(fb.x + 30.0, fb.y, self.map_w, self.map_h)

        elif action == ActionType.MOVE_ENEMY_BASE:
            unit.move_towards(eb.x - 30.0, eb.y, self.map_w, self.map_h)

        elif action == ActionType.MOVE_RESOURCE:
            # Move to closest non-empty resource node
            active_nodes = [n for n in self.state.resource_nodes if n.amount > 10.0]
            if active_nodes:
                closest = min(active_nodes, key=lambda n: unit.distance_to(n.x, n.y))
                unit.move_towards(closest.x, closest.y, self.map_w, self.map_h)

        elif action == ActionType.ATTACK_NEAREST_ENEMY:
            alive_enemies = [e for e in self.state.enemy_units.values() if e.is_alive]
            if alive_enemies:
                closest_enemy = min(alive_enemies, key=lambda e: unit.distance_to(e.x, e.y))
                # Move within firing range
                if unit.distance_to(closest_enemy.x, closest_enemy.y) > unit.attack_range * 0.8:
                    unit.move_towards(closest_enemy.x, closest_enemy.y, self.map_w, self.map_h)

        elif action == ActionType.DEFEND_PERIMETER:
            # Patrol perimeter 120px in front of friendly base
            target_x = fb.x + 120.0
            target_y = fb.y + math.sin(self.state.step_count * 0.1) * 100.0
            unit.move_towards(target_x, target_y, self.map_w, self.map_h)

        elif action == ActionType.TACTICAL_RETREAT:
            alive_enemies = [e for e in self.state.enemy_units.values() if e.is_alive]
            if alive_enemies:
                closest_enemy = min(alive_enemies, key=lambda e: unit.distance_to(e.x, e.y))
                unit.move_away_from(closest_enemy.x, closest_enemy.y, self.map_w, self.map_h)
            else:
                unit.move_towards(fb.x, fb.y, self.map_w, self.map_h)

    def _process_autonomous_communication(self) -> None:
        """
        Symbolic tactical communication protocol triggered by real battlefield events.
        """
        if self.coordination_mode != "coordinated":
            return

        fb = self.state.bases["friendly"]
        eb = self.state.bases["enemy"]
        defense_unit = self.state.friendly_units.get("defense")
        commander_unit = self.state.friendly_units.get("commander")
        attack_unit = self.state.friendly_units.get("attack")

        alive_enemies = [e for e in self.state.enemy_units.values() if e.is_alive]
        alive_friendlies = [f for f in self.state.friendly_units.values() if f.is_alive]

        # 1. Threat Detection: Base perimeter (< 360px) or enemies on friendly half (x < 440.0)
        threats_near_base = [e for e in alive_enemies if e.x < 440.0 or e.distance_to(fb.x, fb.y) < 360.0]
        spotted_enemies = []
        for f in alive_friendlies:
            for e in alive_enemies:
                if f.distance_to(e.x, e.y) <= f.vision_range:
                    spotted_enemies.append(e)

        if threats_near_base and defense_unit and defense_unit.is_alive:
            nearest_threat = min(threats_near_base, key=lambda e: e.distance_to(fb.x, fb.y))
            self.coordinator.post_message(
                Message(
                    sender="defense",
                    recipient="all",
                    msg_type=MessageType.ENEMY_DETECTED,
                    priority=4,
                    target_x=nearest_threat.x,
                    target_y=nearest_threat.y,
                    step=self.state.step_count,
                )
            )
        elif spotted_enemies and not threats_near_base:
            nearest_spotted = min(spotted_enemies, key=lambda e: min(f.distance_to(e.x, e.y) for f in alive_friendlies))
            self.coordinator.post_message(
                Message(
                    sender="attack" if attack_unit and attack_unit.is_alive else "defense",
                    recipient="all",
                    msg_type=MessageType.ENEMY_DETECTED,
                    priority=3,
                    target_x=nearest_spotted.x,
                    target_y=nearest_spotted.y,
                    step=self.state.step_count,
                )
            )

        # 2. Emergency Support Requests (any damaged unit below 50% HP)
        for unit in alive_friendlies:
            if unit.hp < unit.max_hp * 0.5:
                # Find surviving teammates
                other_alive = [f.id for f in alive_friendlies if f.id != unit.id]
                if other_alive:
                    # Prioritize attack if alive, otherwise any other alive teammate
                    target_recip = "attack" if "attack" in other_alive else other_alive[0]
                    self.coordinator.post_message(
                        Message(
                            sender=unit.id,
                            recipient=target_recip,
                            msg_type=MessageType.REQUEST_SUPPORT,
                            priority=5,
                            target_x=unit.x,
                            target_y=unit.y,
                            step=self.state.step_count,
                        )
                    )

        # 3. Strategic Directives with Tactical Command Succession
        # If Commander is fallen, command passes to surviving senior unit (Defense -> Attack)
        leader = None
        for role in ["commander", "defense", "attack"]:
            u = self.state.friendly_units.get(role)
            if u and u.is_alive:
                leader = u
                break

        if leader:
            # Check if any friendly units are currently engaged in active combat/skirmish
            active_skirmish = any(
                any(f.distance_to(e.x, e.y) <= max(f.attack_range, e.attack_range) + 20.0 for e in alive_enemies)
                for f in alive_friendlies
            )

            if threats_near_base:
                self.coordinator.post_message(
                    Message(
                        sender=leader.id,
                        recipient="all",
                        msg_type=MessageType.DEFEND_BASE,
                        priority=5,
                        target_x=fb.x,
                        target_y=fb.y,
                        step=self.state.step_count,
                    )
                )
            elif eb.hp < 600.0 or (len(alive_enemies) <= 1 and not threats_near_base and not active_skirmish):
                # Offensive opening: siege enemy base only if no active base threats or ongoing firefight
                self.coordinator.post_message(
                    Message(
                        sender=leader.id,
                        recipient="attack" if attack_unit and attack_unit.is_alive else "all",
                        msg_type=MessageType.ATTACK_TARGET,
                        priority=4,
                        target_x=eb.x,
                        target_y=eb.y,
                        step=self.state.step_count,
                    )
                )
            elif self.state.friendly_resources < 50.0 and any(n.amount > 20.0 for n in self.state.resource_nodes) and not active_skirmish:
                # Economic prompt
                self.coordinator.post_message(
                    Message(
                        sender=leader.id,
                        recipient="all",
                        msg_type=MessageType.RESOURCE_LOW,
                        priority=3,
                        step=self.state.step_count,
                    )
                )
            elif not threats_near_base and not active_skirmish and self.coordinator.threat_level > 0:
                self.coordinator.post_message(
                    Message(
                        sender=leader.id,
                        recipient="all",
                        msg_type=MessageType.THREAT_REDUCED,
                        priority=2,
                        step=self.state.step_count,
                    )
                )
                self.coordinator.post_message(
                    Message(
                        sender=leader.id,
                        recipient="attack" if attack_unit and attack_unit.is_alive else "all",
                        msg_type=MessageType.RESUME_OFFENSE,
                        priority=3,
                        target_x=eb.x,
                        target_y=eb.y,
                        step=self.state.step_count,
                    )
                )

    def _execute_opponent_behavior(self) -> None:
        """Configurable Opponent Strategy AI."""
        alive_enemies = [e for e in self.state.enemy_units.values() if e.is_alive]
        alive_friendlies = [f for f in self.state.friendly_units.values() if f.is_alive]
        fb = self.state.bases["friendly"]
        eb = self.state.bases["enemy"]

        for enemy in alive_enemies:
            if self.opponent_strategy == "aggressive":
                # Rush friendly base, engage anything on the way
                in_range_friendlies = [f for f in alive_friendlies if enemy.distance_to(f.x, f.y) <= enemy.attack_range]
                if in_range_friendlies:
                    pass  # Stay and attack
                else:
                    enemy.move_towards(fb.x + 30.0, fb.y, self.map_w, self.map_h)

            elif self.opponent_strategy == "defensive":
                # Guard perimeter of enemy base (x > 580)
                intruders = [f for f in alive_friendlies if f.x > 500.0]
                if intruders:
                    closest_intruder = min(intruders, key=lambda f: enemy.distance_to(f.x, f.y))
                    enemy.move_towards(closest_intruder.x, closest_intruder.y, self.map_w, self.map_h)
                else:
                    # Patrol defensive circle around enemy base
                    patrol_x = eb.x - 120.0
                    patrol_y = eb.y + math.sin(self.state.step_count * 0.08 + int(enemy.id.split('_')[-1])) * 80.0
                    enemy.move_towards(patrol_x, patrol_y, self.map_w, self.map_h)

            elif self.opponent_strategy == "resource_focused":
                # Harvest resource first until stockpile >= 100, then advance
                if self.state.enemy_resources < 100.0:
                    active_nodes = [n for n in self.state.resource_nodes if n.amount > 10.0]
                    if active_nodes:
                        closest_node = min(active_nodes, key=lambda n: enemy.distance_to(n.x, n.y))
                        enemy.move_towards(closest_node.x, closest_node.y, self.map_w, self.map_h)
                    else:
                        enemy.move_towards(fb.x, fb.y, self.map_w, self.map_h)
                else:
                    # Unleash coordinated attack once enriched
                    enemy.move_towards(fb.x, fb.y, self.map_w, self.map_h)

    def _resolve_combat(self) -> Dict[str, float]:
        """Resolve firing and damage for all units within attack range."""
        events = {
            "enemy_base_damage": 0.0,
            "friendly_base_damage": 0.0,
            "enemy_unit_damage": 0.0,
            "friendly_unit_damage": 0.0,
            "enemy_units_killed": 0,
            "friendly_units_lost": 0,
            "coordinated_strikes": 0,
        }

        alive_friendlies = [f for f in self.state.friendly_units.values() if f.is_alive]
        alive_enemies = [e for e in self.state.enemy_units.values() if e.is_alive]
        fb = self.state.bases["friendly"]
        eb = self.state.bases["enemy"]

        # Friendly Attacks
        for f in alive_friendlies:
            if f.cooldown_timer == 0:
                # Prioritize enemy units in range
                targets = [e for e in alive_enemies if f.distance_to(e.x, e.y) <= f.attack_range]
                if targets:
                    target = min(targets, key=lambda e: e.hp)
                    target.hp -= f.damage
                    f.cooldown_timer = f.cooldown
                    f.damage_dealt += f.damage
                    events["enemy_unit_damage"] += f.damage

                    # Check for coordinated strike (attack + defense targeting same enemy or adjacent)
                    other_friendlies = [other for other in alive_friendlies if other.id != f.id]
                    if any(other.distance_to(target.x, target.y) <= other.attack_range for other in other_friendlies):
                        events["coordinated_strikes"] += 1

                    if target.hp <= 0:
                        target.is_alive = False
                        f.kills += 1
                        events["enemy_units_killed"] += 1

                # If no units in range, can damage enemy base if in range
                elif f.distance_to(eb.x, eb.y) <= f.attack_range + 30.0:
                    eb.hp = max(0.0, eb.hp - f.damage)
                    f.cooldown_timer = f.cooldown
                    f.damage_dealt += f.damage
                    events["enemy_base_damage"] += f.damage

        # Enemy Attacks
        for e in alive_enemies:
            if e.cooldown_timer == 0:
                targets = [f for f in alive_friendlies if e.distance_to(f.x, f.y) <= e.attack_range]
                if targets:
                    target = min(targets, key=lambda f: f.hp)
                    target.hp -= e.damage
                    e.cooldown_timer = e.cooldown
                    e.damage_dealt += e.damage
                    events["friendly_unit_damage"] += e.damage
                    if target.hp <= 0:
                        target.is_alive = False
                        e.kills += 1
                        events["friendly_units_lost"] += 1
                elif e.distance_to(fb.x, fb.y) <= e.attack_range + 30.0:
                    fb.hp = max(0.0, fb.hp - e.damage)
                    e.cooldown_timer = e.cooldown
                    e.damage_dealt += e.damage
                    events["friendly_base_damage"] += e.damage

        return events

    def _resolve_harvesting(self) -> float:
        """Resolve resource harvesting for units within gathering range."""
        gather_range = float(self.cfg["resources"]["gather_range"])
        gather_rate = float(self.cfg["resources"]["gather_rate"])
        harvested = 0.0

        for node in self.state.resource_nodes:
            if node.amount <= 0.0:
                continue

            # Friendly harvesters
            for f in self.state.friendly_units.values():
                if f.is_alive and f.distance_to(node.x, node.y) <= gather_range:
                    take = min(node.amount, gather_rate)
                    node.amount -= take
                    self.state.friendly_resources += take
                    harvested += take

            # Enemy harvesters
            for e in self.state.enemy_units.values():
                if e.is_alive and e.distance_to(node.x, node.y) <= gather_range:
                    take = min(node.amount, gather_rate)
                    node.amount -= take
                    self.state.enemy_resources += take

        return harvested

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Construct the normalized 49-dimensional observation vector for the agent.
        """
        assert self.state is not None
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        unit = self.state.friendly_units.get(agent_id)
        fb = self.state.bases["friendly"]
        eb = self.state.bases["enemy"]

        idx = 0
        # 1. Self status (5 values)
        if unit and unit.is_alive:
            obs[idx:idx+5] = [
                (unit.x / self.map_w) * 2.0 - 1.0,
                (unit.y / self.map_h) * 2.0 - 1.0,
                unit.hp / unit.max_hp,
                unit.cooldown_timer / max(1.0, float(unit.cooldown)),
                1.0,  # is_alive
            ]
        idx += 5

        # 2. Friendly base status (3 values)
        obs[idx:idx+3] = [
            (fb.x / self.map_w) * 2.0 - 1.0,
            (fb.y / self.map_h) * 2.0 - 1.0,
            fb.hp / fb.max_hp,
        ]
        idx += 3

        # 3. Enemy base status (3 values)
        obs[idx:idx+3] = [
            (eb.x / self.map_w) * 2.0 - 1.0,
            (eb.y / self.map_h) * 2.0 - 1.0,
            eb.hp / eb.max_hp,
        ]
        idx += 3

        # 4. Teammates status (2 other agents * 4 values = 8 values)
        other_agents = [a for a in self.possible_agents if a != agent_id]
        ref_x = unit.x if unit else fb.x
        ref_y = unit.y if unit else fb.y

        for other_id in other_agents:
            other = self.state.friendly_units.get(other_id)
            if other and other.is_alive:
                rel_x = (other.x - ref_x) / (self.map_w / 2.0)
                rel_y = (other.y - ref_y) / (self.map_h / 2.0)
                obs[idx:idx+4] = [
                    np.clip(rel_x, -1.0, 1.0),
                    np.clip(rel_y, -1.0, 1.0),
                    other.hp / other.max_hp,
                    1.0,
                ]
            else:
                obs[idx:idx+4] = [0.0, 0.0, 0.0, 0.0]
            idx += 4

        # 5. Visible enemy units (up to 3 units * 4 values = 12 values)
        enemies = sorted(self.state.enemy_units.values(), key=lambda e: e.id)
        for e in enemies[:3]:
            if e.is_alive:
                rel_x = (e.x - ref_x) / (self.map_w / 2.0)
                rel_y = (e.y - ref_y) / (self.map_h / 2.0)
                obs[idx:idx+4] = [
                    np.clip(rel_x, -1.0, 1.0),
                    np.clip(rel_y, -1.0, 1.0),
                    e.hp / e.max_hp,
                    1.0,
                ]
            else:
                obs[idx:idx+4] = [0.0, 0.0, 0.0, 0.0]
            idx += 4

        # 6. Resource nodes status (3 nodes * 3 values = 9 values)
        for node in self.state.resource_nodes[:3]:
            rel_x = (node.x - ref_x) / (self.map_w / 2.0)
            rel_y = (node.y - ref_y) / (self.map_h / 2.0)
            obs[idx:idx+3] = [
                np.clip(rel_x, -1.0, 1.0),
                np.clip(rel_y, -1.0, 1.0),
                node.amount / node.max_amount,
            ]
            idx += 3

        # 7. Global State (3 values)
        obs[idx:idx+3] = [
            min(1.0, self.state.friendly_resources / 500.0),
            min(1.0, self.state.enemy_resources / 500.0),
            float(self.state.step_count) / float(self.max_steps),
        ]
        idx += 3

        # 8. Communication Features (6 values)
        comm_feats = self.coordinator.get_communication_features(agent_id, self.state)
        obs[idx:idx+6] = comm_feats
        idx += 6

        # 9. Role One-Hot Badge (3 values: commander=[1,0,0], defense=[0,1,0], attack=[0,0,1])
        if self.include_role_id:
            role_map = {"commander": 0, "defense": 1, "attack": 2}
            role_offset = role_map.get(agent_id, 0)
            obs[idx + role_offset] = 1.0
            idx += 3

        assert idx == self.obs_dim, f"Obs dimension mismatch: {idx} != {self.obs_dim}"
        return obs
