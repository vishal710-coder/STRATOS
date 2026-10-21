"""Tests for evaluation metrics and benchmark pipelines."""

import pytest
from evaluation.metrics import EpisodeMetric, aggregate_metrics, to_dataframe
from evaluation.evaluate import evaluate_policy


def test_metrics_aggregation():
    records = [
        EpisodeMetric(1, "test", "coordinated", "aggressive", "friendly", 120, 45.0, 800.0, 0.0, 2, 3, 50.0, 8),
        EpisodeMetric(2, "test", "coordinated", "aggressive", "enemy", 150, -30.0, 0.0, 500.0, 0, 1, 30.0, 6),
    ]
    summary = aggregate_metrics(records)

    assert summary["total_episodes"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["avg_duration"] == 135.0
    assert summary["avg_team_reward"] == 7.5


def test_evaluate_policy_fast():
    records, summary = evaluate_policy(
        checkpoint_path=None,
        num_episodes=2,
        coordination_mode="coordinated",
        opponent_strategy="aggressive",
        seed=42,
    )
    assert len(records) == 2
    assert summary["total_episodes"] == 2
    assert "win_rate" in summary
