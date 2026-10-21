"""Structured tactical communication messages for STRATOS AI."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(str, Enum):
    """Permitted symbolic tactical messages."""
    ENEMY_DETECTED = "ENEMY_DETECTED"
    REQUEST_SUPPORT = "REQUEST_SUPPORT"
    DEFEND_BASE = "DEFEND_BASE"
    ATTACK_TARGET = "ATTACK_TARGET"
    RESOURCE_LOW = "RESOURCE_LOW"
    THREAT_REDUCED = "THREAT_REDUCED"
    RESUME_OFFENSE = "RESUME_OFFENSE"
    HOLD_FORMATION = "HOLD_FORMATION"


@dataclass
class Message:
    """Tactical message exchanged between agents."""
    sender: str
    recipient: str  # "all", "commander", "defense", "attack"
    msg_type: MessageType
    priority: int = 1  # 1 (low) to 5 (urgent)
    target_x: float = 0.0
    target_y: float = 0.0
    step: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def is_relevant_for(self, agent_id: str) -> bool:
        """Check if message is addressed to recipient or squad."""
        return self.recipient in ("all", agent_id)

    def __str__(self) -> str:
        return f"[Step {self.step}] {self.sender.upper()} -> {self.recipient.upper()}: {self.msg_type.value} (P:{self.priority})"
