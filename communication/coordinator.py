"""Tactical Coordinator and Message Bus for STRATOS AI."""

from typing import Any, Dict, List, Optional
import numpy as np
from communication.messages import Message, MessageType


class TacticalCoordinator:
    """Manages tactical message routing, state arbitration, and observation encoding."""

    def __init__(self, coordination_mode: str = "coordinated"):
        """
        Args:
            coordination_mode: 'coordinated' (messages active) or 'independent' (messages blocked/zeroed)
        """
        self.coordination_mode = coordination_mode
        self.message_history: List[Message] = []
        self.inboxes: Dict[str, List[Message]] = {
            "commander": [],
            "defense": [],
            "attack": [],
        }
        self.active_directive: MessageType = MessageType.HOLD_FORMATION
        self.directive_target: Optional[tuple] = None
        self.threat_level: float = 0.0

    def reset(self) -> None:
        """Reset message log and mailboxes."""
        self.message_history.clear()
        for k in self.inboxes:
            self.inboxes[k].clear()
        self.active_directive = MessageType.HOLD_FORMATION
        self.directive_target = None
        self.threat_level = 0.0

    def post_message(self, message: Message) -> None:
        """Post a validated tactical message to the bus."""
        if self.coordination_mode != "coordinated":
            return  # In independent mode, communication is suppressed

        self.message_history.append(message)

        # Route to mailboxes
        for agent_id in self.inboxes:
            if message.is_relevant_for(agent_id):
                self.inboxes[agent_id].append(message)

        # Update high-level coordinator state
        if message.msg_type in (MessageType.DEFEND_BASE, MessageType.ATTACK_TARGET, MessageType.RESUME_OFFENSE, MessageType.RESOURCE_LOW):
            self.active_directive = message.msg_type
            if message.target_x > 0 and message.target_y > 0:
                self.directive_target = (message.target_x, message.target_y)

        if message.msg_type == MessageType.ENEMY_DETECTED:
            self.threat_level = min(1.0, self.threat_level + 0.3)
        elif message.msg_type == MessageType.THREAT_REDUCED:
            self.threat_level = max(0.0, self.threat_level - 0.4)

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Return the most recent messages."""
        return self.message_history[-limit:]

    def get_communication_features(self, agent_id: str, state: Any) -> np.ndarray:
        """
        Extract numerical communication feature vector for observation space (6 dimensions):
        [directive_code, threat_level, rel_target_x, rel_target_y, support_flag, priority]
        """
        if self.coordination_mode != "coordinated":
            # Independent baseline: zeros for communication channels
            return np.zeros(6, dtype=np.float32)

        # Encode directive
        directive_map = {
            MessageType.HOLD_FORMATION: 0.0,
            MessageType.DEFEND_BASE: 0.8,
            MessageType.ATTACK_TARGET: -0.8,
            MessageType.RESUME_OFFENSE: -0.4,
            MessageType.RESOURCE_LOW: 0.3,
        }
        directive_val = directive_map.get(self.active_directive, 0.0)

        # Urgent target offset relative to agent
        agent = state.friendly_units.get(agent_id)
        rel_tx = 0.0
        rel_ty = 0.0
        if self.directive_target and agent:
            rel_tx = (self.directive_target[0] - agent.x) / (state.map_width / 2.0)
            rel_ty = (self.directive_target[1] - agent.y) / (state.map_height / 2.0)
            rel_tx = float(np.clip(rel_tx, -1.0, 1.0))
            rel_ty = float(np.clip(rel_ty, -1.0, 1.0))

        # Check pending messages in inbox with temporal freshness (active within last 15 steps)
        inbox = self.inboxes.get(agent_id, [])
        support_flag = 0.0
        max_prio = 0.0
        if inbox:
            recent = [m for m in inbox if (state.step_count - m.step) <= 15]
            if any(m.msg_type == MessageType.REQUEST_SUPPORT for m in recent):
                support_flag = 1.0
            if recent:
                max_prio = max(m.priority for m in recent) / 5.0

        return np.array([
            directive_val,
            float(self.threat_level),
            rel_tx,
            rel_ty,
            support_flag,
            max_prio,
        ], dtype=np.float32)

    def step_decay(self) -> None:
        """Decay transient alerts per step."""
        if self.threat_level > 0.0:
            self.threat_level = max(0.0, self.threat_level - 0.02)
