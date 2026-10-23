"""Tests for STRATOS AI agents and neural policies."""

import numpy as np
import torch
import pytest

from agents.policies import ActorCriticPolicy
from agents.commander import CommanderAgent
from agents.defense import DefenseAgent
from agents.attack import AttackAgent
from agents.opponent import OpponentController
from environment.actions import NUM_ACTIONS


def test_actor_critic_policy_shapes():
    policy = ActorCriticPolicy(obs_dim=49, act_dim=7, hidden_dim=128)
    dummy_obs = torch.randn(4, 49)

    logits, values = policy(dummy_obs)
    assert logits.shape == (4, 7)
    assert values.shape == (4, 1)

    actions, logprobs, entropy, v = policy.get_action_and_value(dummy_obs)
    assert actions.shape == (4,)
    assert logprobs.shape == (4,)
    assert entropy.shape == (4,)
    assert v.shape == (4, 1)


def test_agents_selection():
    policy = ActorCriticPolicy(obs_dim=49, act_dim=7)

    commander = CommanderAgent(policy=policy)
    defense = DefenseAgent(policy=policy)
    attack = AttackAgent(policy=policy)

    obs = np.random.uniform(-1.0, 1.0, size=(49,)).astype(np.float32)

    c_act = commander.select_action(obs)
    d_act = defense.select_action(obs)
    a_act = attack.select_action(obs)

    assert 0 <= c_act < NUM_ACTIONS
    assert 0 <= d_act < NUM_ACTIONS
    assert 0 <= a_act < NUM_ACTIONS


def test_opponent_controller():
    opp = OpponentController(strategy="aggressive")
    assert opp.strategy == "aggressive"
    opp.set_strategy("defensive")
    assert opp.strategy == "defensive"

    with pytest.raises(ValueError):
        OpponentController(strategy="invalid_strat")


def test_dual_head_policy_shapes():
    """Verify dual-head ActorCriticPolicy (Action + Learned Communication Token)."""
    from agents.policies import CentralizedCritic

    policy = ActorCriticPolicy(obs_dim=73, act_dim=9, msg_dim=9, hidden_dim=128, has_comm_head=True)
    obs = torch.randn(5, 73)

    actions, msgs, act_logp, msg_logp, entropy, values = policy.get_actions_and_values(obs)

    assert actions.shape == (5,)
    assert msgs.shape == (5,)
    assert act_logp.shape == (5,)
    assert msg_logp.shape == (5,)
    assert entropy.shape == (5,)
    assert values.shape == (5, 1)

    # Check discrete ranges
    assert torch.all(actions >= 0) and torch.all(actions < 9)
    assert torch.all(msgs >= 0) and torch.all(msgs < 9)


def test_centralized_critic():
    """Verify MAPPO centralized critic evaluating joint state."""
    from agents.policies import CentralizedCritic

    critic = CentralizedCritic(state_dim=67 * 3, hidden_dim=128)
    joint_state = torch.randn(4, 67 * 3)
    val = critic(joint_state)
    assert val.shape == (4, 1)

