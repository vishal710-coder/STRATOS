"""Tests for STRATOS AI Strategy Simulation Environment (v2 — Dynamic & Stochastic)."""

import numpy as np
import pytest
from environment.strategy_game import StrategyGameEnv
from environment.actions import ActionType


def test_environment_initialization():
    env = StrategyGameEnv()
    assert env.possible_agents == ["commander", "defense", "attack"]
    assert env.obs_dim == 67  # 46 local + 18 comm + 3 role


def test_environment_reset():
    env = StrategyGameEnv()
    obs, infos = env.reset(seed=42)
    assert len(obs) == 3
    for agent in env.possible_agents:
        assert agent in obs
        assert obs[agent].shape == (67,)
        assert np.all(np.isfinite(obs[agent]))
        assert np.all(obs[agent] >= -1.0) and np.all(obs[agent] <= 1.0)


def test_environment_step():
    env = StrategyGameEnv()
    obs, _ = env.reset(seed=42)
    actions = {
        "commander": {"action": ActionType.MOVE_BASE.value, "message": 1},
        "defense": {"action": ActionType.DEFEND_PERIMETER.value, "message": 0},
        "attack": {"action": ActionType.MOVE_ENEMY_BASE.value, "message": 2},
    }
    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    assert len(next_obs) == 3
    for agent in env.possible_agents:
        assert isinstance(rewards[agent], (int, float))
        assert np.isfinite(rewards[agent])
        assert isinstance(terminations[agent], bool)
        assert isinstance(truncations[agent], bool)
        assert next_obs[agent].shape == (67,)


def test_environment_reproducibility():
    """Verify deterministic dynamics under identical random seeds."""
    env1 = StrategyGameEnv()
    env2 = StrategyGameEnv()

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    for a in env1.possible_agents:
        np.testing.assert_allclose(obs1[a], obs2[a], atol=1e-5)

    actions = {"commander": 1, "defense": 5, "attack": 2}
    for _ in range(5):
        o1, r1, t1, tr1, _ = env1.step(actions)
        o2, r2, t2, tr2, _ = env2.step(actions)

        for a in env1.possible_agents:
            np.testing.assert_allclose(o1[a], o2[a], atol=1e-5)
            assert r1[a] == r2[a]
            assert t1[a] == t2[a]
            assert tr1[a] == tr2[a]


def test_environment_seed_diversity():
    """Verify different seeds produce distinct environments and observations."""
    env1 = StrategyGameEnv()
    env2 = StrategyGameEnv()

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=999)

    # Different seeds should produce different unit positions or stats
    diffs = [np.max(np.abs(obs1[a] - obs2[a])) for a in env1.possible_agents]
    assert max(diffs) > 0.01, "Different seeds produced identical observations!"


def test_environment_role_conditioning():
    """Verify 67-dim vs 64-dim observation space and distinct 3-dim role badges."""
    env_with = StrategyGameEnv(include_role_id=True)
    assert env_with.obs_dim == 67
    obs_with, _ = env_with.reset(seed=42)

    assert obs_with["commander"].shape == (67,)
    assert obs_with["defense"].shape == (67,)
    assert obs_with["attack"].shape == (67,)

    # Check 3-dim one-hot badges
    np.testing.assert_allclose(obs_with["commander"][-3:], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs_with["defense"][-3:], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(obs_with["attack"][-3:], [0.0, 0.0, 1.0])

    env_without = StrategyGameEnv(include_role_id=False)
    assert env_without.obs_dim == 64
    obs_without, _ = env_without.reset(seed=42)
    assert obs_without["commander"].shape == (64,)


def test_timeout_is_draw():
    """Verify timeout triggers a draw with win_type='timeout_draw' and truncated=True."""
    env = StrategyGameEnv()
    env.reset(seed=42)
    env.state.step_count = env.max_steps - 1

    actions = {"commander": 0, "defense": 0, "attack": 0}
    _, _, terminations, truncations, infos = env.step(actions)

    assert env.state.truncated is True
    assert env.state.winner == "draw"
    assert env.state.win_type == "timeout_draw"
    for a in env.possible_agents:
        assert truncations[a] is True


def test_no_resource_hoarding_win():
    """Verify that having overwhelmingly high resources does NOT result in a win upon timeout."""
    env = StrategyGameEnv()
    env.reset(seed=42)
    env.state.friendly_resources = 50000.0
    env.state.enemy_resources = 0.0
    env.state.step_count = env.max_steps - 1

    actions = {"commander": 0, "defense": 0, "attack": 0}
    env.step(actions)

    assert env.state.winner == "draw"
    assert env.state.win_type == "timeout_draw"
    assert env.state.winner != "friendly"


def test_message_delivery_delay():
    """Verify 1-step message delivery delay between teammates."""
    env = StrategyGameEnv()
    obs, _ = env.reset(seed=42)

    # Step 1: Commander sends token 3 (meaningful), attack sends token 0 (silent)
    actions_1 = {
        "commander": {"action": 0, "message": 3},
        "defense": {"action": 0, "message": 0},
        "attack": {"action": 0, "message": 0},
    }
    obs_1, _, _, _, _ = env.step(actions_1)

    # At step 1, messages from step 1 have not been delivered yet (1-step delay)
    def_msg_step1 = obs_1["defense"][46:46+9]
    assert def_msg_step1[0] == 1.0  # Still default silent (0)

    # Step 2: Now the message from step 1 should be delivered
    actions_2 = {
        "commander": {"action": 0, "message": 0},
        "defense": {"action": 0, "message": 0},
        "attack": {"action": 0, "message": 0},
    }
    obs_2, _, _, _, _ = env.step(actions_2)

    # Defense should receive token 3 from Commander in slot 0 (commander is first other agent)
    def_msg_step2 = obs_2["defense"][46:46+9]
    assert def_msg_step2[3] == 1.0, f"Expected token 3 received by defense at step 2, got {def_msg_step2}"

    # Attack should also receive token 3 from Commander
    atk_msg_step2 = obs_2["attack"][46:46+9]
    assert atk_msg_step2[3] == 1.0, f"Expected token 3 received by attack at step 2, got {atk_msg_step2}"


def test_partial_observability():
    """Verify enemies outside unit's vision radius are zeroed out in local observation."""
    env = StrategyGameEnv()
    obs, _ = env.reset(seed=42)

    # Place all enemies very far away from friendly units
    for e in env.state.enemy_units.values():
        e.x = 750.0
        e.y = 550.0

    defense = env.state.friendly_units["defense"]
    defense.x = 80.0
    defense.y = 300.0

    obs_def = env._get_observation("defense")
    # Enemy slice is [19:31] (3 enemies * 4 features)
    enemy_slice = obs_def[19:31]
    assert np.all(enemy_slice == 0.0), f"Defense should not see distant enemies: {enemy_slice}"


def test_sync_vec_env():
    """Verify batched synchronous multi-environment stepping."""
    from environment.vec_env import SyncVecEnv

    vec_env = SyncVecEnv(num_envs=3, coordination_mode="coordinated")
    assert vec_env.num_envs == 3
    assert vec_env.obs_dim == 67

    all_obs, all_infos = vec_env.reset(seed=100)
    assert len(all_obs) == 3
    for obs in all_obs:
        assert len(obs) == 3
        for agent in vec_env.possible_agents:
            assert obs[agent].shape == (67,)

    actions_list = [
        {"commander": {"action": 0, "message": 1}, "defense": {"action": 1, "message": 0}, "attack": {"action": 2, "message": 0}},
        {"commander": {"action": 1, "message": 0}, "defense": {"action": 2, "message": 1}, "attack": {"action": 0, "message": 0}},
        {"commander": {"action": 2, "message": 0}, "defense": {"action": 0, "message": 0}, "attack": {"action": 1, "message": 1}},
    ]
    next_obs, rews, terms, truncs, infos = vec_env.step(actions_list)

    assert len(next_obs) == 3
    assert len(rews) == 3
    for i in range(3):
        for agent in vec_env.possible_agents:
            assert isinstance(rews[i][agent], (int, float))


def test_reward_ordering_acceptance():
    """Acceptance test: Random policy must score below Scripted Heuristic, and Heuristic below Targeted."""
    from scripts.audit_and_evaluate_v2 import RandomPolicy, ScriptedHeuristicPolicy
    from agents.targeted_policy import TargetedCombatPolicy

    rand_bot = RandomPolicy()
    heur_bot = ScriptedHeuristicPolicy()
    good_bot = TargetedCombatPolicy()

    seeds = list(range(5000, 5015))
    opponents = ["aggressive", "resource_focused"]

    results = {"random": [], "heuristic": [], "targeted": []}

    for opp in opponents:
        env = StrategyGameEnv(coordination_mode="independent", opponent_strategy=opp)
        for s in seeds:
            # 1. Random
            obs, _ = env.reset(seed=s)
            r_tot = 0.0
            rng = np.random.default_rng(s)
            while True:
                act = {a: rand_bot.select_action(a, obs[a], rng) for a in env.possible_agents}
                obs, r, term, trunc, _ = env.step(act)
                r_tot += sum(r.values())
                if any(term.values()) or any(trunc.values()):
                    break
            results["random"].append(r_tot)

            # 2. Heuristic
            obs, _ = env.reset(seed=s)
            r_tot = 0.0
            while True:
                act = {a: heur_bot.select_action(a, obs[a], rng) for a in env.possible_agents}
                obs, r, term, trunc, _ = env.step(act)
                r_tot += sum(r.values())
                if any(term.values()) or any(trunc.values()):
                    break
            results["heuristic"].append(r_tot)

            # 3. Targeted Good Policy
            obs, _ = env.reset(seed=s)
            r_tot = 0.0
            while True:
                act = {a: good_bot.select_action(a, obs[a], rng) for a in env.possible_agents}
                obs, r, term, trunc, _ = env.step(act)
                r_tot += sum(r.values())
                if any(term.values()) or any(trunc.values()):
                    break
            results["targeted"].append(r_tot)

    mean_rand = np.mean(results["random"])
    mean_heur = np.mean(results["heuristic"])
    mean_good = np.mean(results["targeted"])

    assert mean_rand < mean_heur, f"Expected Random ({mean_rand:.2f}) < Heuristic ({mean_heur:.2f})"
    assert mean_heur < mean_good, f"Expected Heuristic ({mean_heur:.2f}) < Targeted ({mean_good:.2f})"


