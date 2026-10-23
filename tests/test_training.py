"""Tests for PPO training components and checkpoint persistence."""

import tempfile
from pathlib import Path
import torch
import pytest

from agents.policies import ActorCriticPolicy
from training.rollout import MultiAgentRolloutBuffer
from training.ppo import PPOTrainer
from training.checkpoint import save_checkpoint, load_checkpoint


def test_rollout_buffer_insertion_and_gae():
    device = torch.device("cpu")
    buffer = MultiAgentRolloutBuffer(buffer_size=10, num_agents=3, obs_dim=49, device=device)

    for _ in range(10):
        obs = torch.randn(3, 49)
        act = torch.zeros(3, dtype=torch.long)
        logp = torch.zeros(3)
        rew = torch.ones(3)
        done = torch.zeros(3)
        val = torch.ones(3) * 0.5
        buffer.insert(obs, act, logp, rew, done, val)

    next_val = torch.ones(3) * 0.5
    next_done = torch.zeros(3)
    buffer.compute_returns_and_advantages(next_val, next_done)

    assert buffer.advantages.shape == (10, 3)
    assert buffer.returns.shape == (10, 3)


def test_ppo_trainer_update():
    device = torch.device("cpu")
    policy = ActorCriticPolicy(obs_dim=49, act_dim=7)
    trainer = PPOTrainer(policy=policy, device="cpu")
    buffer = MultiAgentRolloutBuffer(buffer_size=10, num_agents=3, obs_dim=49, device=device)

    for _ in range(10):
        obs = torch.randn(3, 49)
        act = torch.zeros(3, dtype=torch.long)
        logp = torch.zeros(3)
        rew = torch.ones(3)
        done = torch.zeros(3)
        val = torch.ones(3) * 0.5
        buffer.insert(obs, act, logp, rew, done, val)

    buffer.compute_returns_and_advantages(torch.ones(3) * 0.5, torch.zeros(3))
    metrics = trainer.update(buffer, batch_size=15, ppo_epochs=2)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_checkpoint_roundtrip():
    policy1 = ActorCriticPolicy(obs_dim=49, act_dim=7)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_checkpoint(tmp_path, policy1, total_steps=500, episode=10)
        policy2, meta = load_checkpoint(tmp_path)

        assert meta["total_steps"] == 500
        assert meta["episode"] == 10

        # Verify identical parameters
        for p1, p2 in zip(policy1.parameters(), policy2.parameters()):
            assert torch.equal(p1, p2)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_train_pipeline_resumption():
    from training.train import train_stratos

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        init_ckpt = tmp_p / "init.pt"
        from environment.actions import NUM_ACTIONS
        policy = ActorCriticPolicy(obs_dim=73, act_dim=NUM_ACTIONS)
        save_checkpoint(str(init_ckpt), policy, total_steps=100, episode=2)

        out_ckpt = tmp_p / "resumed.pt"
        hist_csv = tmp_p / "history.csv"
        hist_json = tmp_p / "history.json"

        results = train_stratos(
            total_timesteps=30,
            seed=999,
            coordination_mode="coordinated",
            opponent_strategy="mixed",
            resume_checkpoint=str(init_ckpt),
            output_checkpoint_path=str(out_ckpt),
            history_csv_path=str(hist_csv),
            history_json_path=str(hist_json),
            verbose=False,
        )

        assert results["total_steps"] >= 130
        assert out_ckpt.exists()
        reloaded_policy, meta = load_checkpoint(str(out_ckpt))
        assert meta["total_steps"] >= 130


def test_training_aids():
    """Verify message entropy bonus, auxiliary fact prediction loss, and message cost."""
    device = torch.device("cpu")
    policy = ActorCriticPolicy(
        obs_dim=73,
        act_dim=9,
        msg_dim=9,
        has_comm_head=True,
        has_aux_head=True,
    )
    trainer = PPOTrainer(
        policy=policy,
        entropy_coef=0.01,
        comm_entropy_coef=0.03,
        aux_loss_coef=0.1,
        device="cpu",
    )
    buffer = MultiAgentRolloutBuffer(buffer_size=10, num_agents=3, obs_dim=73, device=device)

    for _ in range(10):
        obs = torch.randn(3, 73)
        act = torch.zeros(3, dtype=torch.long)
        msg = torch.ones(3, dtype=torch.long)
        logp = torch.zeros(3)
        rew = torch.ones(3)
        done = torch.zeros(3)
        val = torch.ones(3) * 0.5
        fact = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.long)
        buffer.insert(obs, act, logp, rew, done, val, message=msg, hidden_fact=fact)

    buffer.compute_returns_and_advantages(torch.ones(3) * 0.5, torch.zeros(3))
    metrics = trainer.update(buffer, batch_size=15, ppo_epochs=2)

    assert "comm_entropy" in metrics
    assert "aux_loss" in metrics
    assert metrics["comm_entropy"] >= 0.0
    assert metrics["aux_loss"] > 0.0

    # Verify per-message cost in StrategyGameEnv
    from environment.strategy_game import StrategyGameEnv
    env = StrategyGameEnv(message_cost=0.05)
    env.reset(seed=42)
    # Step with non-silent message (token 1)
    _, r_msg, _, _, _ = env.step({
        "commander": {"action": 0, "message": 1},
        "defense": {"action": 0, "message": 0},
        "attack": {"action": 0, "message": 0},
    })
    env.reset(seed=42)
    # Step with silent message (token 0)
    _, r_silent, _, _, _ = env.step({
        "commander": {"action": 0, "message": 0},
        "defense": {"action": 0, "message": 0},
        "attack": {"action": 0, "message": 0},
    })
    # Commander should have received a 0.05 penalty for transmitting message token 1
    assert r_msg["commander"] < r_silent["commander"]
    assert pytest.approx(r_silent["commander"] - r_msg["commander"], abs=1e-4) == 0.05

