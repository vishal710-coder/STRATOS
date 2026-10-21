"""Tests for STRATOS AI Strategy Simulation Environment."""

import numpy as np
import pytest
from environment.strategy_game import StrategyGameEnv
from environment.actions import ActionType


def test_environment_initialization():
    env = StrategyGameEnv()
    assert env.possible_agents == ["commander", "defense", "attack"]
    assert env.obs_dim == 49


def test_environment_reset():
    env = StrategyGameEnv()
    obs, infos = env.reset(seed=42)
    assert len(obs) == 3
    for agent in env.possible_agents:
        assert agent in obs
        assert obs[agent].shape == (49,)
        assert np.all(np.isfinite(obs[agent]))
        assert np.all(obs[agent] >= -1.0) and np.all(obs[agent] <= 1.0)


def test_environment_step():
    env = StrategyGameEnv()
    obs, _ = env.reset(seed=42)
    actions = {
        "commander": ActionType.MOVE_BASE,
        "defense": ActionType.DEFEND_PERIMETER,
        "attack": ActionType.MOVE_ENEMY_BASE,
    }
    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    assert len(next_obs) == 3
    for agent in env.possible_agents:
        assert isinstance(rewards[agent], (int, float))
        assert np.isfinite(rewards[agent])
        assert isinstance(terminations[agent], bool)
        assert isinstance(truncations[agent], bool)


def test_environment_reproducibility():
    """Verify deterministic dynamics under fixed random seeds."""
    env1 = StrategyGameEnv()
    env2 = StrategyGameEnv()

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    for a in env1.possible_agents:
        np.testing.assert_allclose(obs1[a], obs2[a], atol=1e-5)

    actions = {"commander": 1, "defense": 5, "attack": 2}
    o1, r1, t1, tr1, _ = env1.step(actions)
    o2, r2, t2, tr2, _ = env2.step(actions)

    for a in env1.possible_agents:
        np.testing.assert_allclose(o1[a], o2[a], atol=1e-5)
        assert r1[a] == r2[a]
        assert t1[a] == t2[a]


def test_environment_role_conditioning():
    """Verify 52-dim observation space and distinct 3-dim role badges."""
    env = StrategyGameEnv(include_role_id=True)
    assert env.obs_dim == 52
    obs, _ = env.reset(seed=42)

    assert obs["commander"].shape == (52,)
    assert obs["defense"].shape == (52,)
    assert obs["attack"].shape == (52,)

    # Check 3-dim one-hot badges
    np.testing.assert_allclose(obs["commander"][-3:], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["defense"][-3:], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(obs["attack"][-3:], [0.0, 0.0, 1.0])

