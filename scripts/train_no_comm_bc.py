"""Train NO_COMM BC Student Policy from SilentReferencePolicy demonstrations.

This provides an honest, fair baseline for NO_COMM:
- Same architecture (ActorCriticPolicy, obs_dim=73, act_dim=9, has_comm_head=False)
- Supervised demonstrations from SilentReferencePolicy where communication is disabled
- Prevents the confounding factor where NO_COMM inherits a COMM prior expecting scout tokens.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from environment.strategy_game import StrategyGameEnv
from agents.policies import ActorCriticPolicy
from agents.reference_policies import SilentReferencePolicy
from training.checkpoint import save_checkpoint



def collect_silent_trajectories(
    num_episodes_per_opponent: int = 50,
    seed_start: int = 150000,
) -> Dict[str, np.ndarray]:
    opponents = ["aggressive", "defensive", "resource_focused"]
    silent = SilentReferencePolicy()
    role_map = {"commander": 0, "defense": 1, "attack": 2}
    current_seed = seed_start

    t0 = time.time()
    all_obs: List[np.ndarray] = []
    all_acts: List[int] = []
    all_returns: List[float] = []

    for opp in opponents:
        for _ in range(num_episodes_per_opponent):
            env = StrategyGameEnv(
                opponent_strategy=opp,
                coordination_mode="independent",
                enable_communication=False,
                include_role_id=True,
            )
            obs, _ = env.reset(seed=current_seed)
            current_seed += 1

            ep_obs: List[np.ndarray] = []
            ep_acts: List[int] = []
            ep_rewards: List[float] = []
            done = False

            while not done:
                acts_dict = {}
                for ag in env.possible_agents:
                    act, _ = silent.select_action_and_message(ag, obs[ag])
                    acts_dict[ag] = act
                    ep_obs.append(obs[ag].copy())
                    ep_acts.append(act)

                obs, rewards, term, trunc, _ = env.step(acts_dict)
                ep_rewards.append(float(sum(rewards.values()) / len(rewards)))
                done = any(term.values()) or any(trunc.values())

            gamma = 0.99
            step_returns = []
            g = 0.0
            for r in reversed(ep_rewards):
                g = r + gamma * g
                step_returns.append(g)
            step_returns.reverse()

            for i in range(len(ep_obs)):
                step_idx = i // len(env.possible_agents)
                ret = step_returns[step_idx] if step_idx < len(step_returns) else 0.0
                all_obs.append(ep_obs[i])
                all_acts.append(ep_acts[i])
                all_returns.append(ret)

    dt = time.time() - t0
    print(f"[NO_COMM BC] Collected {len(all_obs)} samples across {num_episodes_per_opponent*3} eps in {dt:.2f}s.")
    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "acts": np.array(all_acts, dtype=np.int64),
        "returns": np.array(all_returns, dtype=np.float32),
    }


def train_no_comm_bc():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cpu"

    dataset = collect_silent_trajectories(num_episodes_per_opponent=50, seed_start=150000)

    policy = ActorCriticPolicy(
        obs_dim=73,
        act_dim=9,
        msg_dim=9,
        has_comm_head=False,
        has_aux_head=False,
    ).to(device)

    obs_t = torch.as_tensor(dataset["obs"], dtype=torch.float32)
    acts_t = torch.as_tensor(dataset["acts"], dtype=torch.int64)
    rets_t = torch.as_tensor(dataset["returns"], dtype=torch.float32)

    train_ds = TensorDataset(obs_t, acts_t, rets_t)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    val_loss_fn = nn.MSELoss()

    policy.train()
    print("[NO_COMM BC] Training policy for 12 epochs...")
    for epoch in range(12):
        total_loss = 0.0
        correct = 0
        total = 0
        for b_obs, b_acts, b_rets in loader:
            optimizer.zero_grad()
            hidden = policy.network(b_obs)
            logits = policy.actor(hidden)
            val = policy.critic(hidden).squeeze(-1)
            act_loss = loss_fn(logits, b_acts)
            val_loss = val_loss_fn(val, b_rets) * 0.01
            loss = act_loss + val_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_acts)
            pred = logits.argmax(dim=-1)
            correct += (pred == b_acts).sum().item()
            total += len(b_acts)

        acc = correct / total * 100.0
        print(f"  Epoch {epoch+1:02d}/12: Loss = {total_loss/total:.4f}, Action Acc = {acc:.2f}%")

    ckpt_path = "artifacts/checkpoints/no_comm_bc_student.pt"
    save_checkpoint(
        path=ckpt_path,
        policy=policy,
        optimizer=optimizer,
        total_steps=len(dataset["obs"]),
        episode=150,
        metrics={"action_accuracy": acc},
    )
    print(f"[NO_COMM BC] Checkpoint saved successfully to: {ckpt_path}")


if __name__ == "__main__":
    train_no_comm_bc()
