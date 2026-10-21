"""Opponent controllers for configurable strategy evaluation."""

from typing import List


class OpponentController:
    """Manages AI behavior for enemy squad."""

    STRATEGIES = ["aggressive", "defensive", "resource_focused"]

    def __init__(self, strategy: str = "aggressive"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown opponent strategy: {strategy}. Must be one of {self.STRATEGIES}")
        self.strategy = strategy

    def set_strategy(self, strategy: str) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown opponent strategy: {strategy}. Must be one of {self.STRATEGIES}")
        self.strategy = strategy

    def get_description(self) -> str:
        descriptions = {
            "aggressive": "Direct assault on friendly base, engaging any units encountered.",
            "defensive": "Perimeter defense around enemy base, counter-attacking only when invaded.",
            "resource_focused": "Secures crystal deposits to build strength before launching coordinated siege.",
        }
        return descriptions.get(self.strategy, "")
