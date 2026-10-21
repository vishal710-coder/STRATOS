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
        policy = ActorCriticPolicy(obs_dim=49, act_dim=7)
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
