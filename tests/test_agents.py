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
