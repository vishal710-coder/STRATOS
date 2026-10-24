"""Phase 1: Imitate Oracle via Supervised BC + DAgger rounds.

Generates Oracle trajectories on seeds 100000+, trains decentralized policy
with weighted scout token loss and silence enforcement for commander/defense,
runs DAgger rounds, evaluates on validation seeds 1000-1019, and saves
artifacts/checkpoints/comm_bc_student.pt immediately.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from agents.policies import ActorCriticPolicy
from agents.reference_policies import OracleMessengerPolicy
from training.checkpoint import save_checkpoint, load_checkpoint


def collect_oracle_trajectories(
    num_episodes_per_opponent: int = 80,
    seed_start: int = 100000,
) -> Dict[str, np.ndarray]:
    """Collect expert trajectories by running OracleMessengerPolicy in StrategyGameEnv."""
    opponents = ["aggressive", "defensive", "resource_focused"]
    oracle = OracleMessengerPolicy()

    all_obs: List[np.ndarray] = []
    all_acts: List[int] = []
    all_msgs: List[int] = []
    all_roles: List[int] = []
    all_hidden: List[Tuple[int, int]] = []
    all_returns: List[float] = []

    role_map = {"commander": 0, "defense": 1, "attack": 2}
    current_seed = seed_start

    print(f"[Phase 1] Collecting Oracle expert trajectories ({num_episodes_per_opponent * len(opponents)} episodes)...")
    t0 = time.time()

    for opp in opponents:
        for _ in range(num_episodes_per_opponent):
            env = StrategyGameEnv(
                opponent_strategy=opp,
                enable_communication=True,
                include_role_id=True,
            )
            obs, _ = env.reset(seed=current_seed)
            current_seed += 1
            oracle.reset()

            ep_transitions: List[Tuple[np.ndarray, int, int, int, Tuple[int, int]]] = []
            ep_rewards: List[float] = []
            done = False

            while not done:
                # Expert action & message selection
                acts_dict = {}
                step_targets = {}
                for ag in env.possible_agents:
                    act, msg = oracle.select_action_and_message(ag, obs[ag])
                    acts_dict[ag] = {"action": act, "message": msg}
                    step_targets[ag] = (act, msg)

                # Record per-agent observations and expert decisions
                weak_l = env.state.weak_lane if env.state else 0
                wave_f = env.state.wave_flank if env.state else 0
                for ag in env.possible_agents:
                    ep_transitions.append((
                        obs[ag].copy(),
                        step_targets[ag][0],
                        step_targets[ag][1],
                        role_map[ag],
                        (weak_l, wave_f),
                    ))

                obs, rewards, term, trunc, _ = env.step(acts_dict)
                ep_rewards.append(float(sum(rewards.values()) / len(rewards)))
                done = any(term.values()) or any(trunc.values())

            # Compute discounted returns for critic initialization (gamma=0.99)
            gamma = 0.99
            step_returns = []
            g = 0.0
            for r in reversed(ep_rewards):
                g = r + gamma * g
                step_returns.append(g)
            step_returns.reverse()

            # Assign returns to per-agent transitions (3 transitions per step)
            for i, trans in enumerate(ep_transitions):
                step_idx = i // len(env.possible_agents)
                ret = step_returns[step_idx] if step_idx < len(step_returns) else 0.0
                all_obs.append(trans[0])
                all_acts.append(trans[1])
                all_msgs.append(trans[2])
                all_roles.append(trans[3])
                all_hidden.append(trans[4])
                all_returns.append(ret)

    dt = time.time() - t0
    print(f"[Phase 1] Collected {len(all_obs)} agent samples in {dt:.2f}s ({len(all_obs)/dt:.1f} samples/s).")

    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "acts": np.array(all_acts, dtype=np.int64),
        "msgs": np.array(all_msgs, dtype=np.int64),
        "roles": np.array(all_roles, dtype=np.int64),
        "hidden": np.array(all_hidden, dtype=np.int64),
        "returns": np.array(all_returns, dtype=np.float32),
    }


def collect_dagger_trajectories(
    policy: ActorCriticPolicy,
    num_episodes_per_opponent: int = 20,
    seed_start: int = 110000,
    device: str = "cpu",
    epsilon: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Roll out student policy, query Oracle on visited states, and return aggregated labels."""
    opponents = ["aggressive", "defensive", "resource_focused"]
    oracle = OracleMessengerPolicy()
    role_map = {"commander": 0, "defense": 1, "attack": 2}
    current_seed = seed_start

    policy.eval()
    all_obs: List[np.ndarray] = []
    all_acts: List[int] = []
    all_msgs: List[int] = []
    all_roles: List[int] = []
    all_hidden: List[Tuple[int, int]] = []
    all_returns: List[float] = []

    for opp in opponents:
        for _ in range(num_episodes_per_opponent):
            env = StrategyGameEnv(
                opponent_strategy=opp,
                enable_communication=True,
                include_role_id=True,
            )
            obs, _ = env.reset(seed=current_seed)
            current_seed += 1
            oracle.reset()

            ep_transitions: List[Tuple[np.ndarray, int, int, int, Tuple[int, int]]] = []
            ep_rewards: List[float] = []
            done = False

            while not done:
                # Student decides actions and messages to step the environment
                obs_tensor = torch.as_tensor(
                    np.array([obs[ag] for ag in env.possible_agents]),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_tensor, deterministic=(np.random.rand() > epsilon))

                student_acts = {
                    ag: {"action": int(a_t[i].item()), "message": int(m_t[i].item())}
                    for i, ag in enumerate(env.possible_agents)
                }

                # Query Oracle on visited states for ground-truth expert labels
                weak_l = env.state.weak_lane if env.state else 0
                wave_f = env.state.wave_flank if env.state else 0
                for ag in env.possible_agents:
                    exp_act, exp_msg = oracle.select_action_and_message(ag, obs[ag])
                    ep_transitions.append((
                        obs[ag].copy(),
                        exp_act,
                        exp_msg,
                        role_map[ag],
                        (weak_l, wave_f),
                    ))

                obs, rewards, term, trunc, _ = env.step(student_acts)
                ep_rewards.append(float(sum(rewards.values()) / len(rewards)))
                done = any(term.values()) or any(trunc.values())

            # Returns
            gamma = 0.99
            step_returns = []
            g = 0.0
            for r in reversed(ep_rewards):
                g = r + gamma * g
                step_returns.append(g)
            step_returns.reverse()

            for i, trans in enumerate(ep_transitions):
                step_idx = i // len(env.possible_agents)
                ret = step_returns[step_idx] if step_idx < len(step_returns) else 0.0
                all_obs.append(trans[0])
                all_acts.append(trans[1])
                all_msgs.append(trans[2])
                all_roles.append(trans[3])
                all_hidden.append(trans[4])
                all_returns.append(ret)

    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "acts": np.array(all_acts, dtype=np.int64),
        "msgs": np.array(all_msgs, dtype=np.int64),
        "roles": np.array(all_roles, dtype=np.int64),
        "hidden": np.array(all_hidden, dtype=np.int64),
        "returns": np.array(all_returns, dtype=np.float32),
    }


def train_student_bc(
    policy: ActorCriticPolicy,
    dataset_dict: Dict[str, np.ndarray],
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    scout_msg_weight: float = 4.0,
    others_msg_weight: float = 1.0,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train policy with supervised cross-entropy on actions and tokens."""
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    t_obs = torch.as_tensor(dataset_dict["obs"], dtype=torch.float32)
    t_acts = torch.as_tensor(dataset_dict["acts"], dtype=torch.long)
    t_msgs = torch.as_tensor(dataset_dict["msgs"], dtype=torch.long)
    t_roles = torch.as_tensor(dataset_dict["roles"], dtype=torch.long)
    t_hidden = torch.as_tensor(dataset_dict["hidden"], dtype=torch.long)
    t_returns = torch.as_tensor(dataset_dict["returns"], dtype=torch.float32)

    dataset = TensorDataset(t_obs, t_acts, t_msgs, t_roles, t_hidden, t_returns)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    metrics = {}
    for epoch in range(1, epochs + 1):
        total_loss_accum = 0.0
        act_loss_accum = 0.0
        scout_msg_loss_accum = 0.0
        others_msg_loss_accum = 0.0
        val_loss_accum = 0.0
        aux_loss_accum = 0.0

        for b_obs, b_acts, b_msgs, b_roles, b_hidden, b_ret in loader:
            b_obs = b_obs.to(device)
            b_acts = b_acts.to(device)
            b_msgs = b_msgs.to(device)
            b_roles = b_roles.to(device)
            b_hidden = b_hidden.to(device)
            b_ret = b_ret.to(device)

            features = policy.network(b_obs)
            act_logits = policy.actor(features)
            msg_logits = policy.comm_actor(features)
            values = policy.critic(features).squeeze(-1)

            loss_act = F.cross_entropy(act_logits, b_acts)

            # Role mask: Attack is role == 2
            is_scout = (b_roles == 2)
            if is_scout.any():
                loss_scout_msg = F.cross_entropy(msg_logits[is_scout], b_msgs[is_scout])
            else:
                loss_scout_msg = torch.tensor(0.0, device=device)

            is_others = ~is_scout
            if is_others.any():
                loss_others_msg = F.cross_entropy(msg_logits[is_others], b_msgs[is_others])
            else:
                loss_others_msg = torch.tensor(0.0, device=device)

            loss_val = 0.5 * F.mse_loss(values, b_ret)

            # Auxiliary loss: predict weak_lane and wave_flank from representation
            if policy.has_aux_head and policy.aux_head is not None:
                aux_logits = policy.aux_head(features)
                loss_aux = (
                    F.cross_entropy(aux_logits[:, :3], b_hidden[:, 0])
                    + F.cross_entropy(aux_logits[:, 3:6], b_hidden[:, 1])
                )
            else:
                loss_aux = torch.tensor(0.0, device=device)

            total_loss = (
                loss_act
                + scout_msg_weight * loss_scout_msg
                + others_msg_weight * loss_others_msg
                + 0.1 * loss_aux
                + 0.01 * loss_val
            )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss_accum += total_loss.item()
            act_loss_accum += loss_act.item()
            scout_msg_loss_accum += loss_scout_msg.item()
            others_msg_loss_accum += loss_others_msg.item()
            val_loss_accum += loss_val.item()
            aux_loss_accum += loss_aux.item()

        scheduler.step()
        num_batches = len(loader)
        metrics = {
            "total_loss": total_loss_accum / num_batches,
            "act_loss": act_loss_accum / num_batches,
            "scout_msg_loss": scout_msg_loss_accum / num_batches,
            "others_msg_loss": others_msg_loss_accum / num_batches,
            "val_loss": val_loss_accum / num_batches,
            "aux_loss": aux_loss_accum / num_batches,
            "lr": scheduler.get_last_lr()[0],
        }
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {metrics['total_loss']:.4f} "
                f"(Act: {metrics['act_loss']:.4f}, ScoutMsg: {metrics['scout_msg_loss']:.4f}, "
                f"OthersMsg: {metrics['others_msg_loss']:.4f}, Aux: {metrics['aux_loss']:.4f}) | LR: {metrics['lr']:.5f}"
            )

    return metrics


def evaluate_policy_on_validation(
    policy: ActorCriticPolicy,
    seed_start: int = 1000,
    num_seeds: int = 20,
    deterministic: bool = True,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate policy on validation seeds (1000-1019) across all 3 opponents."""
    policy.eval()
    opponents = ["aggressive", "defensive", "resource_focused"]
    results = []

    for opp in opponents:
        for s_idx in range(num_seeds):
            seed = seed_start + s_idx
            env = StrategyGameEnv(
                opponent_strategy=opp,
                enable_communication=True,
                include_role_id=True,
            )
            obs, _ = env.reset(seed=seed)
            done = False
            ep_reward = 0.0

            while not done:
                obs_tensor = torch.as_tensor(
                    np.array([obs[ag] for ag in env.possible_agents]),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    a_t, m_t, _, _, _, _ = policy.get_actions_and_values(
                        obs_tensor, deterministic=deterministic
                    )

                actions_dict = {
                    ag: {"action": int(a_t[i].item()), "message": int(m_t[i].item())}
                    for i, ag in enumerate(env.possible_agents)
                }
                obs, rewards, term, trunc, _ = env.step(actions_dict)
                ep_reward += sum(rewards.values()) / len(rewards)
                done = any(term.values()) or any(trunc.values())

            winner = env.state.winner if env.state else "draw"
            results.append({
                "win": winner == "friendly",
                "loss": winner == "enemy",
                "draw": winner not in ("friendly", "enemy"),
                "reward": ep_reward,
                "steps": env.state.step_count if env.state else 300,
            })

    win_rate = np.mean([r["win"] for r in results]) * 100.0
    draw_rate = np.mean([r["draw"] for r in results]) * 100.0
    loss_rate = np.mean([r["loss"] for r in results]) * 100.0
    mean_reward = float(np.mean([r["reward"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))

    return {
        "num_episodes": len(results),
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
    }


def main():
    print("=" * 65)
    print("  STRATOS AI — PHASE 1: ORACLE IMITATION & DAGGER")
    print("=" * 65)
    t_start = time.time()
    device = "cpu"

    # Step 1: Collect initial Oracle dataset D_0
    dataset = collect_oracle_trajectories(num_episodes_per_opponent=80, seed_start=100000)

    # Step 2: Initialize student ActorCriticPolicy
    policy = ActorCriticPolicy(
        obs_dim=73,
        act_dim=9,
        msg_dim=9,
        hidden_dim=256,
        has_comm_head=True,
        has_aux_head=True,
    ).to(device)

    print("\n[Phase 1] Training Initial Student via Supervised BC (15 epochs)...")
    train_student_bc(
        policy=policy,
        dataset_dict=dataset,
        epochs=15,
        batch_size=256,
        lr=1e-3,
        scout_msg_weight=4.0,
        others_msg_weight=1.0,
        device=device,
    )

    # Step 3: DAgger Round 1
    print("\n[Phase 1] Running DAgger Round 1 (rolling out student, querying Oracle)...")
    dagger1 = collect_dagger_trajectories(
        policy=policy,
        num_episodes_per_opponent=20,
        seed_start=110000,
        device=device,
        epsilon=0.1,
    )
    for k in dataset:
        dataset[k] = np.concatenate([dataset[k], dagger1[k]], axis=0)
    print(f"[Phase 1] Dataset aggregated: {len(dataset['obs'])} total samples. Retraining (10 epochs)...")
    train_student_bc(
        policy=policy,
        dataset_dict=dataset,
        epochs=10,
        batch_size=256,
        lr=5e-4,
        scout_msg_weight=4.0,
        others_msg_weight=1.0,
        device=device,
    )

    # Step 4: DAgger Round 2
    print("\n[Phase 1] Running DAgger Round 2...")
    dagger2 = collect_dagger_trajectories(
        policy=policy,
        num_episodes_per_opponent=20,
        seed_start=120000,
        device=device,
        epsilon=0.05,
    )
    for k in dataset:
        dataset[k] = np.concatenate([dataset[k], dagger2[k]], axis=0)
    print(f"[Phase 1] Dataset aggregated: {len(dataset['obs'])} total samples. Retraining (10 epochs)...")
    train_student_bc(
        policy=policy,
        dataset_dict=dataset,
        epochs=10,
        batch_size=256,
        lr=3e-4,
        scout_msg_weight=4.0,
        others_msg_weight=1.0,
        device=device,
    )

    # Step 5: Evaluate on validation seeds 1000-1019
    print("\n[Phase 1] Evaluating Student Policy on Validation Seeds (1000-1019, 60 eps)...")
    val_det = evaluate_policy_on_validation(policy, seed_start=1000, num_seeds=20, deterministic=True, device=device)
    val_samp = evaluate_policy_on_validation(policy, seed_start=1000, num_seeds=20, deterministic=False, device=device)

    print(f"Validation (Greedy): Win = {val_det['win_rate']:.1f}%, Draw = {val_det['draw_rate']:.1f}%, "
          f"Loss = {val_det['loss_rate']:.1f}%, Mean Reward = {val_det['mean_reward']:.2f}, Steps = {val_det['mean_steps']:.1f}")
    print(f"Validation (Sampled): Win = {val_samp['win_rate']:.1f}%, Draw = {val_samp['draw_rate']:.1f}%, "
          f"Loss = {val_samp['loss_rate']:.1f}%, Mean Reward = {val_samp['mean_reward']:.2f}")

    # Step 6: Save checkpoint IMMEDIATELY as artifacts/checkpoints/comm_bc_student.pt
    ckpt_path = "artifacts/checkpoints/comm_bc_student.pt"
    save_checkpoint(
        path=ckpt_path,
        policy=policy,
        optimizer=None,
        total_steps=len(dataset["obs"]),
        episode=240 + 60 + 60,
        metrics={"val_greedy": val_det, "val_sampled": val_samp},
    )
    print(f"\n[Phase 1 Complete] Saved fallback checkpoint IMMEDIATELY to: {ckpt_path}")
    print(f"Phase 1 Duration: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
