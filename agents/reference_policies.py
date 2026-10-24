"""Oracle and Silent reference policies for evaluating communication value."""

import numpy as np
from environment.actions import ActionType


class OracleMessengerPolicy:
    """Oracle communication policy:
    - Attack scout observes weak_lane and broadcasts token (lane + 1).
    - Defense guard observes wave_flank warning and broadcasts token (flank + 4).
    - Commander receives tokens and coordinates squad attack on the weak lane.
    """

    def __init__(self):
        self.known_weak_lane = {}

    def reset(self):
        self.known_weak_lane = {}

    def select_action_and_message(self, agent: str, obs: np.ndarray) -> tuple[int, int]:
        self_hp = obs[2]
        friendly_base_hp = obs[7]
        enemy_visible = any(obs[19 + i * 4 + 3] > 0.5 for i in range(3))

        msg_token = 0
        action = ActionType.HOLD

        if agent == "attack":
            weak_slice = obs[46:49]
            if np.any(weak_slice > 0.5):
                lane = int(np.argmax(weak_slice))
                self.known_weak_lane["attack"] = lane
                msg_token = lane + 1
            else:
                lane = self.known_weak_lane.get("attack", 1)

            if self_hp < 0.15:
                action = ActionType.TACTICAL_RETREAT
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            else:
                if lane == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif lane == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE

        elif agent == "defense":
            flank_slice = obs[49:52]
            warning_active = np.any(flank_slice > 0.5)
            if warning_active:
                flank = int(np.argmax(flank_slice))
                msg_token = flank + 4
            else:
                flank = None

            slot1 = obs[52:61]
            slot2 = obs[61:70]
            for slot in (slot1, slot2):
                if np.any(slot > 0.5):
                    t = int(np.argmax(slot))
                    if 1 <= t <= 3:
                        self.known_weak_lane["defense"] = t - 1

            weak_lane = self.known_weak_lane.get("defense", None)

            if self_hp < 0.15:
                action = ActionType.TACTICAL_RETREAT
            elif warning_active and flank is not None:
                if flank == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif flank == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            elif weak_lane is not None:
                if weak_lane == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif weak_lane == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE
            else:
                action = ActionType.DEFEND_PERIMETER

        elif agent == "commander":
            slot1 = obs[52:61]
            slot2 = obs[61:70]
            for slot in (slot1, slot2):
                if np.any(slot > 0.5):
                    t = int(np.argmax(slot))
                    if 1 <= t <= 3:
                        self.known_weak_lane["commander"] = t - 1

            weak_lane = self.known_weak_lane.get("commander", None)

            if self_hp < 0.15:
                action = ActionType.TACTICAL_RETREAT
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            elif weak_lane is not None:
                if weak_lane == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif weak_lane == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE
            else:
                action = ActionType.DEFEND_PERIMETER

        return int(action), int(msg_token)


class SilentReferencePolicy:
    """Silent reference policy:
    - Identical combat and tactical reflexes as Oracle, BUT:
    - Messages are completely suppressed (token = 0).
    - Teammates do NOT receive weak lane or wave flank intelligence.
    - Commander and teammates must guess blindly (defaulting to Mid lane or perimeter).
    """

    def select_action_and_message(self, agent: str, obs: np.ndarray) -> tuple[int, int]:
        self_hp = obs[2]
        friendly_base_hp = obs[7]
        enemy_visible = any(obs[19 + i * 4 + 3] > 0.5 for i in range(3))

        action = ActionType.HOLD

        if agent == "attack":
            # Attack knows its own lane, but CANNOT tell teammates
            weak_slice = obs[46:49]
            lane = int(np.argmax(weak_slice)) if np.any(weak_slice > 0.5) else 1

            if self_hp < 0.15:
                action = ActionType.TACTICAL_RETREAT
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            else:
                if lane == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif lane == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE

        elif agent == "defense":
            # Defense sees warning, but cannot coordinate; guards base perimeter
            if self_hp < 0.15:
                action = ActionType.TACTICAL_RETREAT
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            else:
                action = ActionType.DEFEND_PERIMETER

        elif agent == "commander":
            # Commander receives NO messages (silent) -> pushes default Mid lane blindly
            if self_hp < 0.15:
                action = ActionType.TACTICAL_RETREAT
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            else:
                action = ActionType.MOVE_ENEMY_BASE  # Blindly pushes Mid lane (wrong 67% of the time)

        return int(action), 0  # Always silent (token 0)


class OracleV2Policy:
    """Oracle v2 policy with tactical improvements based on Phase 1 failure analysis:
    - Low-HP retreat towards base (ActionType.MOVE_BASE) rather than random flee.
    - Flank wave interception: Defense prioritizes defending base/perimeter when wave alert fires.
    - Coordinated retreat: sole survivor falls back to defend perimeter instead of charging 1v3.
    """

    def __init__(self):
        self.known_weak_lane = {}

    def reset(self):
        self.known_weak_lane = {}

    def select_action_and_message(self, agent: str, obs: np.ndarray) -> tuple[int, int]:
        self_hp = obs[2]
        friendly_base_hp = obs[7]
        enemy_visible = any(obs[19 + i * 4 + 3] > 0.5 for i in range(3))

        teammate1_alive = obs[11] > 0.5
        teammate2_alive = obs[15] > 0.5
        sole_survivor = (not teammate1_alive) and (not teammate2_alive)

        msg_token = 0
        action = ActionType.HOLD

        if agent == "attack":
            weak_slice = obs[46:49]
            if np.any(weak_slice > 0.5):
                lane = int(np.argmax(weak_slice))
                self.known_weak_lane["attack"] = lane
                msg_token = lane + 1
            else:
                lane = self.known_weak_lane.get("attack", 1)

            if self_hp < 0.20:
                action = ActionType.MOVE_BASE
            elif sole_survivor:
                action = ActionType.DEFEND_PERIMETER
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            else:
                if lane == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif lane == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE

        elif agent == "defense":
            flank_slice = obs[49:52]
            warning_active = np.any(flank_slice > 0.5)
            if warning_active:
                flank = int(np.argmax(flank_slice))
                msg_token = flank + 4
            else:
                flank = None

            slot1 = obs[52:61]
            slot2 = obs[61:70]
            for slot in (slot1, slot2):
                if np.any(slot > 0.5):
                    t = int(np.argmax(slot))
                    if 1 <= t <= 3:
                        self.known_weak_lane["defense"] = t - 1

            weak_lane = self.known_weak_lane.get("defense", None)

            if self_hp < 0.20:
                action = ActionType.MOVE_BASE
            elif warning_active:
                if enemy_visible:
                    action = ActionType.ATTACK_NEAREST_ENEMY
                else:
                    action = ActionType.DEFEND_PERIMETER
            elif sole_survivor:
                action = ActionType.DEFEND_PERIMETER
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            elif weak_lane is not None:
                if weak_lane == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif weak_lane == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE
            else:
                action = ActionType.DEFEND_PERIMETER

        elif agent == "commander":
            slot1 = obs[52:61]
            slot2 = obs[61:70]
            for slot in (slot1, slot2):
                if np.any(slot > 0.5):
                    t = int(np.argmax(slot))
                    if 1 <= t <= 3:
                        self.known_weak_lane["commander"] = t - 1

            weak_lane = self.known_weak_lane.get("commander", None)

            if self_hp < 0.20:
                action = ActionType.MOVE_BASE
            elif sole_survivor:
                action = ActionType.DEFEND_PERIMETER
            elif enemy_visible:
                action = ActionType.ATTACK_NEAREST_ENEMY
            elif weak_lane is not None:
                if weak_lane == 0:
                    action = ActionType.PUSH_TOP_LANE
                elif weak_lane == 1:
                    action = ActionType.MOVE_ENEMY_BASE
                else:
                    action = ActionType.PUSH_BOT_LANE
            else:
                action = ActionType.DEFEND_PERIMETER

        return int(action), int(msg_token)

