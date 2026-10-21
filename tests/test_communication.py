"""Tests for tactical message bus and agent communication channel."""

import numpy as np
import pytest

from communication.messages import Message, MessageType
from communication.coordinator import TacticalCoordinator
from environment.game_state import GameState, Unit, Base


def test_message_creation():
    msg = Message(
        sender="defense",
        recipient="all",
        msg_type=MessageType.ENEMY_DETECTED,
        priority=4,
        target_x=250.0,
        target_y=300.0,
    )
    assert msg.is_relevant_for("commander") is True
    assert msg.is_relevant_for("attack") is True
    assert msg.priority == 4


def test_coordinator_routing():
    coord = TacticalCoordinator(coordination_mode="coordinated")
    msg = Message(
        sender="defense",
        recipient="attack",
        msg_type=MessageType.REQUEST_SUPPORT,
        priority=5,
    )
    coord.post_message(msg)

    assert len(coord.inboxes["attack"]) == 1
    assert len(coord.inboxes["commander"]) == 0
    assert len(coord.message_history) == 1


def test_independent_mode_zeroes_communication():
    coord_indep = TacticalCoordinator(coordination_mode="independent")
    state = GameState()
    state.friendly_units["defense"] = Unit("defense", "friendly", "defense", "Defense", 100, 100, 100, 100, 3, 10, 50, 100)

    # In independent mode, posted messages are dropped
    msg = Message(sender="defense", recipient="all", msg_type=MessageType.ENEMY_DETECTED, priority=4)
    coord_indep.post_message(msg)

    feats = coord_indep.get_communication_features("defense", state)
    assert np.all(feats == 0.0)


def test_message_freshness_decay():
    """Verify that transient support requests expire after 15 steps."""
    coord = TacticalCoordinator(coordination_mode="coordinated")
    state = GameState()
    state.friendly_units["attack"] = Unit("attack", "friendly", "attack", "Attack", 100, 100, 100, 100, 3, 10, 50, 100)

    # Post support request at step 10
    msg = Message(sender="defense", recipient="attack", msg_type=MessageType.REQUEST_SUPPORT, priority=5, step=10)
    coord.post_message(msg)

    # At step 15 (within 15 steps), support flag should be active
    state.step_count = 15
    feats_15 = coord.get_communication_features("attack", state)
    assert feats_15[4] == 1.0, f"Expected support_flag 1.0, got {feats_15[4]}"
    assert feats_15[5] == 1.0, f"Expected priority 1.0 (5/5), got {feats_15[5]}"

    # At step 30 (20 steps later), support flag should expire to 0.0
    state.step_count = 30
    feats_30 = coord.get_communication_features("attack", state)
    assert feats_30[4] == 0.0, f"Expected support_flag 0.0 after expiry, got {feats_30[4]}"


def test_communication_dataflow_to_policy():
    """Verify that communication features directly shift policy logits and state value."""
    import torch
    from agents.policies import ActorCriticPolicy

    policy = ActorCriticPolicy(obs_dim=49, act_dim=7)
    base_obs = np.random.uniform(-0.5, 0.5, size=(43,)).astype(np.float32)

    indep_obs = np.concatenate([base_obs, np.zeros(6, dtype=np.float32)])
    coord_obs = np.concatenate([base_obs, np.array([0.8, 0.9, 0.2, -0.1, 1.0, 1.0], dtype=np.float32)])

    t_indep = torch.as_tensor(indep_obs).unsqueeze(0)
    t_coord = torch.as_tensor(coord_obs).unsqueeze(0)

    with torch.no_grad():
        logits_indep, val_indep = policy(t_indep)
        logits_coord, val_coord = policy(t_coord)

    logit_diff = torch.abs(logits_coord - logits_indep).max().item()
    val_diff = torch.abs(val_coord - val_indep).item()

    assert logit_diff > 0.0, "Communication features produced 0 change in policy logits!"
    assert val_diff > 0.0, "Communication features produced 0 change in critic value!"


def test_autonomous_communication_event_triggers():
    """Verify that environmental events populate communication features."""
    from environment.strategy_game import StrategyGameEnv

    env = StrategyGameEnv(coordination_mode="coordinated", opponent_strategy="aggressive")
    obs, _ = env.reset(seed=42)

    # Move enemies close to base to trigger threat detection
    env.state.enemy_units["enemy_1"].x = 200.0
    env.state.enemy_units["enemy_1"].y = 300.0

    # Step environment
    obs, _, _, _, _ = env.step({"commander": 0, "defense": 0, "attack": 0})

    # Coordinator must have messages and non-zero communication features
    assert len(env.coordinator.message_history) > 0
    attack_comm = obs["attack"][43:49]
    assert np.any(attack_comm != 0.0), f"Expected non-zero comm features, got {attack_comm}"

