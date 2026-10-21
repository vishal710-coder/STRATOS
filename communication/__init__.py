"""Communication module for STRATOS AI."""

from communication.messages import Message, MessageType
from communication.coordinator import TacticalCoordinator

__all__ = ["Message", "MessageType", "TacticalCoordinator"]
