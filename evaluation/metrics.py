"""Metrics calculation and aggregation for STRATOS AI research evaluation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import numpy as np
import pandas as pd


@dataclass
class EpisodeMetric:
    """Individual episode evaluation result."""
    episode_id: int
    scenario_name: str
    coordination_mode: str
    opponent_strategy: str
    winner: str
    duration_steps: int
    team_reward: float
    friendly_base_final_hp: float
    enemy_base_final_hp: float
    friendly_units_survived: int
    enemy_units_eliminated: int
    friendly_resources_collected: float
    messages_exchanged: int
    win_type: str = "unknown"


def aggregate_metrics(episode_records: List[EpisodeMetric]) -> Dict[str, Any]:
    """
    Compute rigorous statistical metrics across a sample of evaluation episodes.
    """
    if not episode_records:
        return {}

    total_episodes = len(episode_records)
    wins = sum(1 for e in episode_records if e.winner == "friendly")
    losses = sum(1 for e in episode_records if e.winner == "enemy")
    draws = sum(1 for e in episode_records if e.winner == "draw")

    durations = [e.duration_steps for e in episode_records]
    rewards = [e.team_reward for e in episode_records]
    survivals = [e.friendly_units_survived for e in episode_records]
    enemy_losses = [e.enemy_units_eliminated for e in episode_records]
    resources = [e.friendly_resources_collected for e in episode_records]
    messages = [e.messages_exchanged for e in episode_records]

    return {
        "total_episodes": total_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / total_episodes, 4),
        "loss_rate": round(losses / total_episodes, 4),
        "draw_rate": round(draws / total_episodes, 4),
        "avg_duration": round(float(np.mean(durations)), 2),
        "avg_team_reward": round(float(np.mean(rewards)), 2),
        "std_team_reward": round(float(np.std(rewards)), 2),
        "avg_friendly_survival": round(float(np.mean(survivals)), 2),
        "friendly_survival_rate": round(float(np.mean(survivals)) / 3.0, 4),
        "avg_enemy_eliminated": round(float(np.mean(enemy_losses)), 2),
        "avg_resources_collected": round(float(np.mean(resources)), 2),
        "avg_messages_exchanged": round(float(np.mean(messages)), 2),
    }


def to_dataframe(episode_records: List[EpisodeMetric]) -> pd.DataFrame:
    """Convert list of EpisodeMetric records to pandas DataFrame."""
    return pd.DataFrame([asdict(e) for e in episode_records])
