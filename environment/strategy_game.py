"""PettingZoo Parallel-compatible Multi-Agent Strategy Game Environment (v2).

Dynamic, stochastic, partially-observable team game with:
- Seeded RNG for reproducibility (same seed = identical episode)
- Per-episode randomization of spawns, stats, resources
- Dynamic events: storm, enemy waves, strategy switch, supply drops, resource respawn
- Partial observability: each unit sees only within its vision radius
- Support for learned communication (agent-emitted message tokens)
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import gymnasium
from gymnasium import spaces
import numpy as np
import yaml

from communication.coordinator import TacticalCoordinator
from communication.messages import Message, MessageType
from environment.actions import ActionType, NUM_ACTIONS
from environment.game_state import Base, GameState, ResourceNode, StormZone, SupplyDrop, Unit
from environment.rewards import RewardCalculator


class StrategyGameEnv:
    """
    STRATOS AI Strategy Simulation Environment (v2).
    Follows PettingZoo Parallel API standard for multi-agent reinforcement learning.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "stratos_v2"}

    def __init__(
        self,
        config_path: str = "config/game.yaml",
        coordination_mode: str = "coordinated",
        opponent_strategy: str = "aggressive",
        enable_communication: bool = True,
        include_role_id: bool = True,
        **kwargs,
    ):
        self.config_path = config_path
        self.coordination_mode = coordination_mode
        self.opponent_strategy = opponent_strategy
        self.enable_communication = enable_communication
        self.include_role_id = include_role_id

        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.map_w = float(self.cfg["map"]["width"])
        self.map_h = float(self.cfg["map"]["height"])
        self.max_steps = int(self.cfg["simulation"]["max_steps"])

        self.possible_agents = ["commander", "defense", "attack"]
        self.agents = list(self.possible_agents)

        # Communication config
        comm_cfg = self.cfg.get("communication", {})
        self.vocab_size = int(comm_cfg.get("vocab_size", 8))  # K meaningful tokens
        self.msg_tokens = self.vocab_size + 1  # +1 for "silent" token (index 0)
        self.msg_delay = int(comm_cfg.get("message_delay", 1))
        self.msg_cost = float(kwargs.get("message_cost", comm_cfg.get("message_cost", 0.0)))
        self.hq_regen = bool(kwargs.get("hq_regen", False))
        self.supply_broadcast = bool(kwargs.get("supply_broadcast", False))

        # Observation dimensions:
        # [self_status(5) + friendly_base(3) + enemy_base(3) + teammates(2x4=8) +
        #  enemies(3x4=12) + resources(3x3=9) + global(3) + storm(3) = 46 local]
        # + hidden_info(3 weak_lane + 3 wave_flank = 6) = 52 local
        # + received_messages(2 * msg_tokens) + role_id(3 or 0)
        self.local_obs_dim = 52
        self.msg_obs_dim = 2 * self.msg_tokens  # messages from 2 teammates
        self.role_obs_dim = 3 if self.include_role_id else 0
        self.obs_dim = self.local_obs_dim + self.msg_obs_dim + self.role_obs_dim

        self._obs_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Dict({
                "action": spaces.Discrete(NUM_ACTIONS),
                "message": spaces.Discrete(self.msg_tokens),
            })
            for agent in self.possible_agents
        }

        self.reward_calculator = RewardCalculator()
        if kwargs.get("enable_team_cohesion", False):
            self.reward_calculator.enable_team_cohesion = True
        self.coordinator = TacticalCoordinator(coordination_mode=coordination_mode)
        self.state: Optional[GameState] = None

        # Seeded RNG — ALL randomness flows through this
        self.rng: np.random.Generator = np.random.default_rng(42)

        # Message buffers: outgoing messages are delivered on the NEXT step
        self._pending_messages: Dict[str, int] = {}  # agent_id -> token sent this step
        self._delivered_messages: Dict[str, Dict[str, int]] = {}  # recipient -> {sender: token}

        # Node ID counter for spawned resource nodes
        self._next_node_id = 10

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Dict:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state with per-episode randomization."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "opponent_strategy" in options:
            self.opponent_strategy = options["opponent_strategy"]
        if options and "coordination_mode" in options:
            self.coordination_mode = options["coordination_mode"]
            self.coordinator.coordination_mode = self.coordination_mode
        if options and "enable_communication" in options:
            self.enable_communication = options["enable_communication"]
        if options and "hq_regen" in options:
            self.hq_regen = bool(options["hq_regen"])
        if options and "supply_broadcast" in options:
            self.supply_broadcast = bool(options["supply_broadcast"])

        self.agents = list(self.possible_agents)
        self.coordinator.reset()

        # Randomization config
        rand_cfg = self.cfg.get("randomization", {})
        spawn_jitter = float(rand_cfg.get("spawn_jitter", 30))
        stat_range = rand_cfg.get("stat_multiplier_range", [0.85, 1.15])
        res_jitter = float(rand_cfg.get("resource_pos_jitter", 50))
        res_amount_range = rand_cfg.get("resource_amount_range", [300, 700])
        res_regen_range = rand_cfg.get("resource_regen_range", [0.5, 2.0])

        # Initialize Bases (fixed positions — bases don't move)
        fb_cfg = self.cfg["bases"]["friendly"]
        eb_cfg = self.cfg["bases"]["enemy"]
        bases = {
            "friendly": Base(team="friendly", x=float(fb_cfg["x"]), y=float(fb_cfg["y"]), hp=float(fb_cfg["hp"])),
            "enemy": Base(team="enemy", x=float(eb_cfg["x"]), y=float(eb_cfg["y"]), hp=float(eb_cfg["hp"])),
        }

        # Initialize Resource Nodes (with jitter)
        nodes = []
        for n in self.cfg["resources"]["nodes"]:
            jx = float(n["x"]) + self.rng.uniform(-res_jitter, res_jitter)
            jy = float(n["y"]) + self.rng.uniform(-res_jitter, res_jitter)
            jx = float(np.clip(jx, 50, self.map_w - 50))
            jy = float(np.clip(jy, 50, self.map_h - 50))
            amt = float(self.rng.uniform(res_amount_range[0], res_amount_range[1]))
            regen = float(self.rng.uniform(res_regen_range[0], res_regen_range[1]))
            nodes.append(
                ResourceNode(
                    id=int(n["id"]),
                    x=jx, y=jy,
                    amount=amt,
                    max_amount=amt,
                    regen_rate=regen,
                )
            )

        # Initialize Friendly Units (with spawn jitter and stat multipliers)
        friendly_units = {}
        for role in ["commander", "defense", "attack"]:
            u_cfg = self.cfg["units"][role]
            stat_mult = float(self.rng.uniform(stat_range[0], stat_range[1]))
            jx = float(u_cfg["start_pos"][0]) + self.rng.uniform(-spawn_jitter, spawn_jitter)
            jy = float(u_cfg["start_pos"][1]) + self.rng.uniform(-spawn_jitter, spawn_jitter)
            jx = float(np.clip(jx, 20, self.map_w / 2 - 20))
            jy = float(np.clip(jy, 20, self.map_h - 20))
            friendly_units[role] = Unit(
                id=role,
                team="friendly",
                role=role,
                name=u_cfg["name"],
                x=jx, y=jy,
                hp=float(u_cfg["hp"]) * stat_mult,
                max_hp=float(u_cfg["max_hp"]) * stat_mult,
                speed=float(u_cfg["speed"]) * stat_mult,
                damage=float(u_cfg["damage"]) * stat_mult,
                attack_range=float(u_cfg["attack_range"]),
                vision_range=float(u_cfg["vision_range"]),
                cooldown=int(u_cfg["cooldown"]),
            )

        # Initialize Enemy Units (with spawn jitter and stat multipliers)
        enemy_units = {}
        e_defaults = self.cfg["enemy_squad"]["unit_defaults"]
        e_positions = self.cfg["enemy_squad"]["initial_positions"]
        for i, pos in enumerate(e_positions):
            eid = f"enemy_{i+1}"
            stat_mult = float(self.rng.uniform(stat_range[0], stat_range[1]))
            jx = float(pos[0]) + self.rng.uniform(-spawn_jitter, spawn_jitter)
            jy = float(pos[1]) + self.rng.uniform(-spawn_jitter, spawn_jitter)
            jx = float(np.clip(jx, self.map_w / 2 + 20, self.map_w - 20))
            jy = float(np.clip(jy, 20, self.map_h - 20))
            enemy_units[eid] = Unit(
                id=eid,
                team="enemy",
                role="enemy_grunt",
                name=f"Enemy_{i+1}",
                x=jx, y=jy,
                hp=float(e_defaults["hp"]) * stat_mult,
                max_hp=float(e_defaults["max_hp"]) * stat_mult,
                speed=float(e_defaults["speed"]) * stat_mult,
                damage=float(e_defaults["damage"]) * stat_mult,
                attack_range=float(e_defaults["attack_range"]),
                vision_range=float(e_defaults["vision_range"]),
                cooldown=int(e_defaults["cooldown"]),
            )

        # Schedule dynamic events
        dyn = self.cfg.get("dynamic_events", {})

        storm_step = 999999
        if dyn.get("storm", {}).get("enabled", False):
            sr = dyn["storm"]["spawn_step_range"]
            storm_step = int(self.rng.integers(sr[0], sr[1] + 1))

        wave_step = 999999
        if dyn.get("enemy_wave", {}).get("enabled", False):
            wr = dyn["enemy_wave"]["spawn_step_range"]
            wave_step = int(self.rng.integers(wr[0], wr[1] + 1))

        switch_step = 999999
        if dyn.get("enemy_strategy_switch", {}).get("enabled", False):
            ss = dyn["enemy_strategy_switch"]["switch_step_range"]
            switch_step = int(self.rng.integers(ss[0], ss[1] + 1))

        supply_step = 999999
        if dyn.get("supply_drop", {}).get("enabled", False):
            sd = dyn["supply_drop"]["spawn_step_range"]
            supply_step = int(self.rng.integers(sd[0], sd[1] + 1))

        res_spawn_step = 999999
        if dyn.get("resource_respawn", {}).get("enabled", False):
            rr = dyn["resource_respawn"]["interval_range"]
            res_spawn_step = int(self.rng.integers(rr[0], rr[1] + 1))

        # Randomize hidden asymmetric facts
        weak_lane = int(self.rng.integers(0, 3))
        wave_flank = int(self.rng.integers(0, 3))

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
            win_type=None,
            terminated=False,
            truncated=False,
            storm=None,
            supply_drop=None,
            enemy_strategy=self.opponent_strategy,
            storm_spawn_step=storm_step,
            enemy_wave_step=wave_step,
            enemy_strategy_switch_step=switch_step,
            supply_drop_step=supply_step,
            next_resource_spawn_step=res_spawn_step,
            weak_lane=weak_lane,
            wave_flank=wave_flank,
            wave_warning_active=False,
        )

        # Reset message buffers
        self._pending_messages = {}
        self._delivered_messages = {a: {} for a in self.possible_agents}
        self._next_node_id = 10

        obs = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Advance simulation by 1 discrete step.

        actions can be:
          - Dict[str, int]: action index only (legacy, no communication)
          - Dict[str, Dict[str, int]]: {"action": int, "message": int}
        """
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
            "storm_damage": 0.0,
        }

        # Parse actions — support both legacy (int) and new (dict with message) formats
        parsed_actions: Dict[str, int] = {}
        new_messages: Dict[str, int] = {}
        for agent_id, act in actions.items():
            if isinstance(act, dict):
                parsed_actions[agent_id] = int(act.get("action", 0))
                new_messages[agent_id] = int(act.get("message", 0))
            else:
                parsed_actions[agent_id] = int(act)
                new_messages[agent_id] = 0  # silent

        # 0. Deliver pending messages from previous step
        self._delivered_messages = {a: {} for a in self.possible_agents}
        for sender, token in self._pending_messages.items():
            for recipient in self.possible_agents:
                if recipient != sender:
                    self._delivered_messages[recipient][sender] = token

        # Store this step's messages for delivery next step
        if self.enable_communication and self.coordination_mode == "coordinated":
            self._pending_messages = dict(new_messages)
        else:
            self._pending_messages = {}

        # Update coordinator state and history for battlefield visualizer / telemetry
        self.coordinator.step_decay()
        if self.coordination_mode == "coordinated":
            for sender, token in new_messages.items():
                if token > 0:
                    self.coordinator.post_message(
                        Message(
                            sender=sender,
                            recipient="all",
                            msg_type=MessageType.REQUEST_SUPPORT if token % 2 == 1 else MessageType.ENEMY_DETECTED,
                            priority=min(5, max(1, token)),
                            step=self.state.step_count,
                            payload={"token": token},
                        )
                    )

        # 1. Update Cooldowns & Regen
        for u in list(self.state.friendly_units.values()) + list(self.state.enemy_units.values()):
            if u.cooldown_timer > 0:
                u.cooldown_timer -= 1
        for node in self.state.resource_nodes:
            if node.amount < node.max_amount:
                node.amount = min(node.max_amount, node.amount + node.regen_rate)

        # 2. Execute Friendly Actions
        for agent_id, action_idx in parsed_actions.items():
            if agent_id in self.state.friendly_units:
                unit = self.state.friendly_units[agent_id]
                if unit.is_alive:
                    unit.current_action = action_idx
                    self._apply_unit_action(unit, action_idx)

        # 3. Dynamic Events
        self._process_dynamic_events(step_events)

        # 4. Execute Opponent Strategy AI
        self._execute_opponent_behavior()

        # 5. Resolve Combat Engagements
        combat_events = self._resolve_combat()
        for k, v in combat_events.items():
            step_events[k] = step_events.get(k, 0.0) + v

        # 6. Resolve Resource Harvesting
        harvest_val = self._resolve_harvesting()
        step_events["resources_harvested"] = harvest_val

        # 7. Check Game Over
        is_over = self.state.is_game_over()

        # 8. Compute Rewards
        rewards = self.reward_calculator.compute_rewards(self.state, self.state, step_events)
        self.last_step_events = dict(step_events)

        # Apply message cost
        if self.msg_cost > 0:
            for agent_id, token in new_messages.items():
                if token > 0:  # non-silent
                    rewards[agent_id] -= self.msg_cost

        terminations = {agent: self.state.terminated for agent in self.possible_agents}
        truncations = {agent: self.state.truncated for agent in self.possible_agents}

        obs = {agent: self._get_observation(agent) for agent in self.possible_agents}
        infos = {
            agent: {
                "step": self.state.step_count,
                "winner": self.state.winner,
                "win_type": self.state.win_type,
                "friendly_base_hp": self.state.bases["friendly"].hp,
                "enemy_base_hp": self.state.bases["enemy"].hp,
                "friendly_resources": self.state.friendly_resources,
                "enemy_strategy": self.state.enemy_strategy,
                "storm_active": self.state.storm is not None and self.state.storm.active,
                "supply_drop_broadcast": (
                    (self.state.supply_drop.x, self.state.supply_drop.y)
                    if self.supply_broadcast and self.state.supply_drop and not self.state.supply_drop.collected
                    else None
                ),
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
            unit.move_towards(700.0, 300.0, self.map_w, self.map_h)

        elif action == ActionType.PUSH_TOP_LANE:
            if unit.x < 300.0 and unit.y > 160.0:
                unit.move_towards(220.0, 120.0, self.map_w, self.map_h)
            else:
                unit.move_towards(700.0, 120.0, self.map_w, self.map_h)

        elif action == ActionType.PUSH_BOT_LANE:
            if unit.x < 300.0 and unit.y < 440.0:
                unit.move_towards(220.0, 480.0, self.map_w, self.map_h)
            else:
                unit.move_towards(700.0, 480.0, self.map_w, self.map_h)

        elif action == ActionType.MOVE_RESOURCE:
            # Move to closest non-empty resource node that is visible
            active_nodes = [n for n in self.state.resource_nodes
                          if n.amount > 10.0 and unit.can_see(n.x, n.y)]
            if not active_nodes:
                # Fallback: move to any known node position
                active_nodes = [n for n in self.state.resource_nodes if n.amount > 10.0]
            if active_nodes:
                closest = min(active_nodes, key=lambda n: unit.distance_to(n.x, n.y))
                unit.move_towards(closest.x, closest.y, self.map_w, self.map_h)

        elif action == ActionType.ATTACK_NEAREST_ENEMY:
            # Only attack enemies that are visible
            alive_enemies = [e for e in self.state.enemy_units.values()
                           if e.is_alive and unit.can_see(e.x, e.y)]
            if alive_enemies:
                closest_enemy = min(alive_enemies, key=lambda e: unit.distance_to(e.x, e.y))
                if unit.distance_to(closest_enemy.x, closest_enemy.y) > unit.attack_range * 0.8:
                    unit.move_towards(closest_enemy.x, closest_enemy.y, self.map_w, self.map_h)

        elif action == ActionType.DEFEND_PERIMETER:
            target_x = fb.x + 120.0
            target_y = fb.y + math.sin(self.state.step_count * 0.1) * 100.0
            unit.move_towards(target_x, target_y, self.map_w, self.map_h)

        elif action == ActionType.TACTICAL_RETREAT:
            alive_enemies = [e for e in self.state.enemy_units.values()
                           if e.is_alive and unit.can_see(e.x, e.y)]
            if alive_enemies:
                closest_enemy = min(alive_enemies, key=lambda e: unit.distance_to(e.x, e.y))
                unit.move_away_from(closest_enemy.x, closest_enemy.y, self.map_w, self.map_h)
            else:
                unit.move_towards(fb.x, fb.y, self.map_w, self.map_h)

    def _process_dynamic_events(self, step_events: Dict[str, Any]) -> None:
        """Handle all dynamic events: storm, waves, strategy switch, supply drops, resource respawn."""
        s = self.state
        step = s.step_count
        dyn = self.cfg.get("dynamic_events", {})

        # --- Storm zone ---
        if step == s.storm_spawn_step and s.storm is None:
            storm_cfg = dyn.get("storm", {})
            dur_range = storm_cfg.get("duration_range", [60, 150])
            duration = int(self.rng.integers(dur_range[0], dur_range[1] + 1))
            # Spawn from a random edge
            edge = int(self.rng.integers(0, 4))  # 0=top, 1=right, 2=bottom, 3=left
            radius = float(storm_cfg.get("radius", 60.0))
            if edge == 0:
                sx, sy = float(self.rng.uniform(100, self.map_w - 100)), radius
            elif edge == 1:
                sx, sy = self.map_w - radius, float(self.rng.uniform(100, self.map_h - 100))
            elif edge == 2:
                sx, sy = float(self.rng.uniform(100, self.map_w - 100)), self.map_h - radius
            else:
                sx, sy = radius, float(self.rng.uniform(100, self.map_h - 100))

            drift_speed = float(storm_cfg.get("drift_speed", 1.5))
            angle = float(self.rng.uniform(0, 2 * math.pi))
            s.storm = StormZone(
                x=sx, y=sy, radius=radius,
                damage_per_step=float(storm_cfg.get("damage_per_step", 5.0)),
                drift_dx=drift_speed * math.cos(angle),
                drift_dy=drift_speed * math.sin(angle),
                spawn_step=step,
                end_step=step + duration,
            )

        if s.storm and s.storm.active:
            if step >= s.storm.end_step:
                s.storm.active = False
            else:
                s.storm.advance(self.map_w, self.map_h)
                # Damage all units inside storm
                for u in list(s.friendly_units.values()) + list(s.enemy_units.values()):
                    if u.is_alive and s.storm.contains(u.x, u.y):
                        u.hp -= s.storm.damage_per_step
                        if u.team == "friendly":
                            step_events["friendly_unit_damage"] += s.storm.damage_per_step
                        else:
                            step_events["enemy_unit_damage"] += s.storm.damage_per_step
                        if u.hp <= 0:
                            u.is_alive = False
                            if u.team == "friendly":
                                step_events["friendly_units_lost"] += 1
                            else:
                                step_events["enemy_units_killed"] += 1

        # Wave warning sensor (Defense receives early alert 30 steps before arrival)
        warning_lead = 30
        if (s.enemy_wave_step - warning_lead <= step < s.enemy_wave_step + 10) and s.enemy_wave_step < 999999:
            s.wave_warning_active = True
        else:
            s.wave_warning_active = False

        # --- Enemy reinforcement wave ---
        if step == s.enemy_wave_step:
            wave_cfg = dyn.get("enemy_wave", {})
            num_range = wave_cfg.get("num_units_range", [1, 2])
            num_new = int(self.rng.integers(num_range[0], num_range[1] + 1))
            flanks = ["top", "middle", "bottom"]
            flank = flanks[s.wave_flank]
            flank_y_center = 90.0 if flank == "top" else (300.0 if flank == "middle" else 510.0)

            # Check if friendly units are guarding this flank
            guard_present = any(
                f.is_alive and abs(f.y - flank_y_center) <= 120.0 and f.x <= 550.0
                for f in s.friendly_units.values()
            )

            e_defaults = self.cfg["enemy_squad"]["unit_defaults"]
            rand_cfg = self.cfg.get("randomization", {})
            stat_range = rand_cfg.get("stat_multiplier_range", [0.85, 1.15])

            for j in range(num_new):
                eid = f"enemy_wave_{j+1}"
                stat_mult = float(self.rng.uniform(stat_range[0], stat_range[1]))
                if flank == "top":
                    ey = float(self.rng.uniform(30, 150))
                elif flank == "bottom":
                    ey = float(self.rng.uniform(self.map_h - 150, self.map_h - 30))
                else:
                    ey = float(self.rng.uniform(200, 400))
                ex = float(self.rng.uniform(self.map_w - 80, self.map_w - 20))

                spawn_hp = float(e_defaults["hp"]) * stat_mult
                if guard_present:
                    # Friendly guard ambushes the incoming wave!
                    spawn_hp = max(10.0, spawn_hp - 40.0)
                    step_events["enemy_unit_damage"] += 40.0

                s.enemy_units[eid] = Unit(
                    id=eid, team="enemy", role="enemy_grunt", name=f"Wave_{j+1}",
                    x=ex, y=ey,
                    hp=spawn_hp,
                    max_hp=float(e_defaults["max_hp"]) * stat_mult,
                    speed=float(e_defaults["speed"]) * stat_mult,
                    damage=float(e_defaults["damage"]) * stat_mult,
                    attack_range=float(e_defaults["attack_range"]),
                    vision_range=float(e_defaults["vision_range"]),
                    cooldown=int(e_defaults["cooldown"]),
                )

            if not guard_present:
                # Unguarded flank: sneak attack inflicts surprise damage to friendly base
                fb = s.bases["friendly"]
                fb.hp = max(0.0, fb.hp - 100.0)
                step_events["friendly_base_damage"] += 100.0

        # --- Enemy strategy switch ---
        if step == s.enemy_strategy_switch_step:
            switch_cfg = dyn.get("enemy_strategy_switch", {})
            strategies = switch_cfg.get("strategies", ["aggressive", "defensive", "resource_focused"])
            available = [st for st in strategies if st != s.enemy_strategy]
            if available:
                s.enemy_strategy = available[int(self.rng.integers(0, len(available)))]

        # --- Supply drop ---
        if step == s.supply_drop_step and s.supply_drop is None:
            sd_cfg = dyn.get("supply_drop", {})
            sx = float(self.rng.uniform(100, self.map_w - 100))
            sy = float(self.rng.uniform(80, self.map_h - 80))
            s.supply_drop = SupplyDrop(
                x=sx, y=sy,
                heal_fraction=float(sd_cfg.get("heal_amount", 0.5)),
                spawn_step=step,
            )

        # Check supply drop pickup (visible units only)
        if s.supply_drop and not s.supply_drop.collected:
            sd = s.supply_drop
            for u in s.friendly_units.values():
                if u.is_alive and u.distance_to(sd.x, sd.y) <= sd.pickup_range:
                    # Heal all friendly units
                    for fu in s.friendly_units.values():
                        if fu.is_alive:
                            fu.hp = min(fu.max_hp, fu.hp + fu.max_hp * sd.heal_fraction)
                    sd.collected = True
                    break

        # --- HQ Regeneration (Environment Fix 1, default False) ---
        if self.hq_regen:
            fbase = s.bases.get("friendly")
            if fbase and fbase.is_alive:
                for fu in s.friendly_units.values():
                    if fu.is_alive and fu.distance_to(fbase.x, fbase.y) <= 60.0:
                        fu.hp = min(fu.max_hp, fu.hp + 0.5)
            ebase = s.bases.get("enemy")
            if ebase and ebase.is_alive:
                for eu in s.enemy_units.values():
                    if eu.is_alive and eu.distance_to(ebase.x, ebase.y) <= 60.0:
                        eu.hp = min(eu.max_hp, eu.hp + 0.5)


        # --- Resource node respawn ---
        if step == s.next_resource_spawn_step:
            res_cfg = dyn.get("resource_respawn", {})
            max_nodes = int(res_cfg.get("max_total_nodes", 5))
            active_nodes = [n for n in s.resource_nodes if n.amount > 0]
            if len(active_nodes) < max_nodes:
                rand_cfg_r = self.cfg.get("randomization", {})
                res_amount_range = rand_cfg_r.get("resource_amount_range", [300, 700])
                res_regen_range = rand_cfg_r.get("resource_regen_range", [0.5, 2.0])
                nx = float(self.rng.uniform(150, self.map_w - 150))
                ny = float(self.rng.uniform(80, self.map_h - 80))
                amt = float(self.rng.uniform(res_amount_range[0], res_amount_range[1]))
                regen = float(self.rng.uniform(res_regen_range[0], res_regen_range[1]))
                s.resource_nodes.append(ResourceNode(
                    id=self._next_node_id, x=nx, y=ny,
                    amount=amt, max_amount=amt, regen_rate=regen,
                ))
                self._next_node_id += 1
            # Schedule next respawn
            interval_range = res_cfg.get("interval_range", [50, 120])
            s.next_resource_spawn_step = step + int(self.rng.integers(
                interval_range[0], interval_range[1] + 1))

    def _execute_opponent_behavior(self) -> None:
        """Configurable Opponent Strategy AI (uses current strategy from state, supports mid-game switch)."""
        alive_enemies = [e for e in self.state.enemy_units.values() if e.is_alive]
        alive_friendlies = [f for f in self.state.friendly_units.values() if f.is_alive]
        fb = self.state.bases["friendly"]
        eb = self.state.bases["enemy"]
        strategy = self.state.enemy_strategy

        for enemy in alive_enemies:
            if strategy == "aggressive":
                in_range_friendlies = [f for f in alive_friendlies if enemy.distance_to(f.x, f.y) <= enemy.attack_range]
                if in_range_friendlies:
                    pass  # Stay and attack
                else:
                    enemy.move_towards(fb.x + 30.0, fb.y, self.map_w, self.map_h)

            elif strategy == "defensive":
                intruders = [f for f in alive_friendlies if f.x > 500.0]
                if intruders:
                    closest_intruder = min(intruders, key=lambda f: enemy.distance_to(f.x, f.y))
                    enemy.move_towards(closest_intruder.x, closest_intruder.y, self.map_w, self.map_h)
                else:
                    patrol_x = eb.x - 120.0
                    enemy_idx = int(enemy.id.split('_')[-1]) if '_' in enemy.id else 0
                    patrol_y = eb.y + math.sin(self.state.step_count * 0.08 + enemy_idx) * 80.0
                    enemy.move_towards(patrol_x, patrol_y, self.map_w, self.map_h)

            elif strategy == "resource_focused":
                if self.state.enemy_resources < 100.0:
                    active_nodes = [n for n in self.state.resource_nodes if n.amount > 10.0]
                    if active_nodes:
                        closest_node = min(active_nodes, key=lambda n: enemy.distance_to(n.x, n.y))
                        enemy.move_towards(closest_node.x, closest_node.y, self.map_w, self.map_h)
                    else:
                        enemy.move_towards(fb.x, fb.y, self.map_w, self.map_h)
                else:
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
                targets = [e for e in alive_enemies if f.distance_to(e.x, e.y) <= f.attack_range]
                if targets:
                    target = min(targets, key=lambda e: e.hp)
                    target.hp -= f.damage
                    f.cooldown_timer = f.cooldown
                    f.damage_dealt += f.damage
                    events["enemy_unit_damage"] += f.damage

                    other_friendlies = [other for other in alive_friendlies if other.id != f.id]
                    if any(other.distance_to(target.x, target.y) <= other.attack_range for other in other_friendlies):
                        events["coordinated_strikes"] += 1

                    if target.hp <= 0:
                        target.is_alive = False
                        f.kills += 1
                        events["enemy_units_killed"] += 1

                elif f.distance_to(eb.x, eb.y) <= f.attack_range + 30.0 or min(
                    math.hypot(f.x - 700.0, f.y - 120.0),
                    math.hypot(f.x - 700.0, f.y - 300.0),
                    math.hypot(f.x - 700.0, f.y - 480.0),
                ) <= f.attack_range + 30.0:
                    if f.y < 210.0:
                        attacked_lane = 0
                    elif f.y < 390.0:
                        attacked_lane = 1
                    else:
                        attacked_lane = 2

                    if attacked_lane == self.state.weak_lane:
                        multiplier = 4.0  # Vulnerable weak generator / undefended flank
                    else:
                        multiplier = 0.1  # Heavy energy shield absorbs attack

                    eff_damage = f.damage * multiplier
                    eb.hp = max(0.0, eb.hp - eff_damage)
                    f.cooldown_timer = f.cooldown
                    f.damage_dealt += eff_damage
                    events["enemy_base_damage"] += eff_damage

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

            for f in self.state.friendly_units.values():
                if f.is_alive and f.distance_to(node.x, node.y) <= gather_range:
                    take = min(node.amount, gather_rate)
                    node.amount -= take
                    self.state.friendly_resources += take
                    harvested += take

            for e in self.state.enemy_units.values():
                if e.is_alive and e.distance_to(node.x, node.y) <= gather_range:
                    take = min(node.amount, gather_rate)
                    node.amount -= take
                    self.state.enemy_resources += take

        return harvested

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Construct the partially-observable observation vector for the agent.

        Structure:
          [0:5]   Self status (x, y, hp, cooldown, alive)
          [5:8]   Friendly base (x, y, hp_frac)
          [8:11]  Enemy base (x, y, hp_frac) - only if visible
          [11:19] Teammates (2 * 4 values: rel_x, rel_y, hp_frac, alive)
          [19:31] Visible enemy units (3 * 4 values, zeroed if not visible)
          [31:40] Visible resource nodes (3 * 3 values, zeroed if not visible)
          [40:43] Global state (friendly_resources_frac, enemy_resources_frac, time_frac)
          [43:46] Storm info (rel_x, rel_y, active_flag) if visible
          [46:46+2*msg_tokens] Received messages (one-hot)
          [46+2*msg_tokens:] Role ID one-hot (3 dims)
        """
        assert self.state is not None
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        unit = self.state.friendly_units.get(agent_id)
        fb = self.state.bases["friendly"]
        eb = self.state.bases["enemy"]

        ref_x = unit.x if (unit and unit.is_alive) else fb.x
        ref_y = unit.y if (unit and unit.is_alive) else fb.y

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

        # 2. Friendly base status (3 values) — always visible (Commander has base telemetry)
        obs[idx:idx+3] = [
            (fb.x / self.map_w) * 2.0 - 1.0,
            (fb.y / self.map_h) * 2.0 - 1.0,
            fb.hp / fb.max_hp,
        ]
        idx += 3

        # 3. Enemy base status (3 values) — only if visible to this unit
        if unit and unit.is_alive and unit.can_see(eb.x, eb.y):
            obs[idx:idx+3] = [
                (eb.x / self.map_w) * 2.0 - 1.0,
                (eb.y / self.map_h) * 2.0 - 1.0,
                eb.hp / eb.max_hp,
            ]
        # else: stays zeros (unknown)
        idx += 3

        # 4. Teammates status (2 other agents * 4 values = 8 values)
        # Commander special: always sees teammate HP. Others: only if in vision.
        other_agents = [a for a in self.possible_agents if a != agent_id]
        for other_id in other_agents:
            other = self.state.friendly_units.get(other_id)
            if other and other.is_alive:
                can_observe = (
                    agent_id == "commander"  # Commander sees all teammate HP
                    or (unit and unit.is_alive and unit.can_see(other.x, other.y))
                )
                if can_observe:
                    rel_x = (other.x - ref_x) / (self.map_w / 2.0)
                    rel_y = (other.y - ref_y) / (self.map_h / 2.0)
                    obs[idx:idx+4] = [
                        np.clip(rel_x, -1.0, 1.0),
                        np.clip(rel_y, -1.0, 1.0),
                        other.hp / other.max_hp,
                        1.0,
                    ]
                elif agent_id == "commander":
                    # Commander sees HP but not accurate position (only rough direction)
                    obs[idx:idx+4] = [0.0, 0.0, other.hp / other.max_hp, 1.0]
            idx += 4

        # 5. Visible enemy units (up to 3 units * 4 values = 12 values)
        # Only enemies within vision radius
        enemies = sorted(self.state.enemy_units.values(), key=lambda e: e.id)
        visible_enemies = []
        if unit and unit.is_alive:
            visible_enemies = [e for e in enemies if e.is_alive and unit.can_see(e.x, e.y)]
        # Sort by distance, take closest 3
        visible_enemies.sort(key=lambda e: unit.distance_to(e.x, e.y) if unit else 999)
        for i in range(3):
            if i < len(visible_enemies):
                e = visible_enemies[i]
                rel_x = (e.x - ref_x) / (self.map_w / 2.0)
                rel_y = (e.y - ref_y) / (self.map_h / 2.0)
                obs[idx:idx+4] = [
                    np.clip(rel_x, -1.0, 1.0),
                    np.clip(rel_y, -1.0, 1.0),
                    e.hp / e.max_hp,
                    1.0,
                ]
            idx += 4

        # 6. Visible resource nodes (up to 3 nodes * 3 values = 9 values)
        visible_nodes = []
        if unit and unit.is_alive:
            visible_nodes = [n for n in self.state.resource_nodes
                           if n.amount > 0 and unit.can_see(n.x, n.y)]
        visible_nodes.sort(key=lambda n: unit.distance_to(n.x, n.y) if unit else 999)
        for i in range(3):
            if i < len(visible_nodes):
                n = visible_nodes[i]
                rel_x = (n.x - ref_x) / (self.map_w / 2.0)
                rel_y = (n.y - ref_y) / (self.map_h / 2.0)
                obs[idx:idx+3] = [
                    np.clip(rel_x, -1.0, 1.0),
                    np.clip(rel_y, -1.0, 1.0),
                    n.amount / n.max_amount,
                ]
            idx += 3

        # 7. Global State (3 values)
        obs[idx:idx+3] = [
            min(1.0, self.state.friendly_resources / 500.0),
            min(1.0, self.state.enemy_resources / 500.0),
            float(self.state.step_count) / float(self.max_steps),
        ]
        idx += 3

        # 8. Storm info (3 values) — only if visible to this unit
        storm = self.state.storm
        if storm and storm.active and unit and unit.is_alive and unit.can_see(storm.x, storm.y):
            rel_sx = (storm.x - ref_x) / (self.map_w / 2.0)
            rel_sy = (storm.y - ref_y) / (self.map_h / 2.0)
            obs[idx:idx+3] = [
                np.clip(rel_sx, -1.0, 1.0),
                np.clip(rel_sy, -1.0, 1.0),
                1.0,  # storm active flag
            ]
        idx += 3

        # 9. Hidden asymmetric information (3 weak_lane dims + 3 wave_flank dims = 6 values)
        # Attack scout sees weak_lane (one-hot)
        if agent_id == "attack" and self.state.weak_lane in (0, 1, 2):
            obs[idx + self.state.weak_lane] = 1.0
        idx += 3

        # Defense guard sees wave_flank only when warning sensor is active (one-hot)
        if agent_id == "defense" and self.state.wave_warning_active and self.state.wave_flank in (0, 1, 2):
            obs[idx + self.state.wave_flank] = 1.0
        idx += 3

        # 10. Received messages (2 teammates * msg_tokens one-hot)
        if self.enable_communication and self.coordination_mode == "coordinated":
            other_ids = [a for a in self.possible_agents if a != agent_id]
            for i_other, other_id in enumerate(other_ids):
                token = self._delivered_messages.get(agent_id, {}).get(other_id, 0)
                if 0 <= token < self.msg_tokens:
                    obs[idx + token] = 1.0
                idx += self.msg_tokens
        else:
            idx += self.msg_obs_dim  # zeros

        # 10. Role One-Hot ID (3 values if enabled)
        if self.include_role_id:
            role_map = {"commander": 0, "defense": 1, "attack": 2}
            role_offset = role_map.get(agent_id, 0)
            obs[idx + role_offset] = 1.0
            idx += 3

        assert idx == self.obs_dim, f"Obs dimension mismatch: {idx} != {self.obs_dim}"
        return obs
