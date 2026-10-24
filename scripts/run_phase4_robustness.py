"""Phase 4: Robustness Testing Suite on Validation Seeds 1000-1049.

Evaluates Best COMM, Oracle, and NO_COMM under communication disruptions:
1. Baseline: intact (1-step delay, 0% drop, 0% noise)
2. Message Loss: drop probability p in {0.10, 0.30, 0.50} (dropped token -> 0)
3. Message Delay: 2 steps delay instead of 1 step
4. Token Noise: 10% random token corruption (uniform over 0..8)
5. Forced Scout Death @ Step 50: Attack scout killed at step 50
6. Forced Scout Death @ Step 100: Attack scout killed at step 100

Each condition tested across 50 validation seeds (1000-1049) x 3 opponents = 150 episodes.
"""

from collections import deque
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from agents.reference_policies import OracleMessengerPolicy
from training.checkpoint import load_checkpoint


def run_single_episode(args: Tuple) -> Dict[str, Any]:
    torch.set_num_threads(1)
    (
        model_name,
        ckpt_path,
        opponent,
        seed,
        drop_prob,
        delay_steps,
        noise_prob,
        forced_kill_step,
    ) = args

    # Initialize policy
    if model_name == "Oracle":
        policy = OracleMessengerPolicy()
        has_comm = True
    elif model_name == "Best_COMM":
        policy, _ = load_checkpoint(ckpt_path, device="cpu")
        policy.eval()
        has_comm = True
    elif model_name == "NO_COMM":
        policy, _ = load_checkpoint(ckpt_path, device="cpu")
        policy.eval()
        has_comm = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Seed local RNG for perturbations
    rng = np.random.default_rng(seed * 1000 + int(drop_prob * 100) + int(noise_prob * 100) + delay_steps)

    env = StrategyGameEnv(
        opponent_strategy=opponent,
        coordination_mode="coordinated" if has_comm else "independent",
        enable_communication=has_comm,
        include_role_id=True,
        enable_team_cohesion=True,
    )
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    # FIFO message queue for delay testing
    # queue holds dict of {agent_id: token}
    # delay_steps=1 is standard default
    msg_queue = deque()
    for _ in range(delay_steps - 1):
        msg_queue.append({a: 0 for a in env.possible_agents})

    done = False
    step = 0
    ep_reward = 0.0

    while not done:
        step += 1

        # Check forced scout death perturbation
        if forced_kill_step is not None and step >= forced_kill_step:
            scout = env.state.friendly_units.get("attack")
            if scout and scout.is_alive:
                scout.is_alive = False
                scout.hp = 0.0

        # Action and message selection
        if model_name == "Oracle":
            acts = {}
            raw_msgs = {}
            for a in env.possible_agents:
                act, msg = policy.select_action_and_message(a, obs[a])
                acts[a] = act
                raw_msgs[a] = msg
        elif model_name == "Best_COMM":
            obs_t = torch.as_tensor(np.array([obs[a] for a in env.possible_agents]), dtype=torch.float32)
            with torch.no_grad():
                a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=True)
            acts = {a: int(a_t[i].item()) for i, a in enumerate(env.possible_agents)}
            raw_msgs = {a: int(m_t[i].item()) for i, a in enumerate(env.possible_agents)}
        else:  # NO_COMM
            obs_t = torch.as_tensor(np.array([obs[a] for a in env.possible_agents]), dtype=torch.float32)
            with torch.no_grad():
                a_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
            acts = {a: int(a_t[i].item()) for i, a in enumerate(env.possible_agents)}
            raw_msgs = {a: 0 for a in env.possible_agents}

        # Apply message perturbations (noise, loss, delay)
        perturbed_msgs = {}
        for a, tok in raw_msgs.items():
            cur_tok = tok
            # Token noise
            if noise_prob > 0.0 and rng.random() < noise_prob:
                cur_tok = int(rng.integers(0, 9))
            # Message loss
            if drop_prob > 0.0 and rng.random() < drop_prob:
                cur_tok = 0
            perturbed_msgs[a] = cur_tok

        # Message delay queue
        msg_queue.append(perturbed_msgs)
        delivered_to_env = msg_queue.popleft()

        # Build action dict for env
        if has_comm:
            action_dict = {
                a: {"action": acts[a], "message": delivered_to_env[a]}
                for a in env.possible_agents
            }
        else:
            action_dict = {a: acts[a] for a in env.possible_agents}

        obs, rewards, term, trunc, _ = env.step(action_dict)
        ep_reward += sum(rewards.values()) / len(rewards)
        done = any(term.values()) or any(trunc.values())

    winner = env.state.winner if env.state else "draw"
    return {
        "model": model_name,
        "opponent": opponent,
        "seed": seed,
        "win": winner == "friendly",
        "loss": winner == "enemy",
        "draw": winner not in ("friendly", "enemy"),
        "reward": ep_reward,
        "steps": env.state.step_count if env.state else 300,
    }


def evaluate_perturbation(
    model_name: str,
    ckpt_path: str,
    seeds: List[int],
    opponents: List[str],
    drop_prob: float = 0.0,
    delay_steps: int = 1,
    noise_prob: float = 0.0,
    forced_kill_step: int = None,
    pool: mp.Pool = None,
) -> Dict[str, Any]:
    tasks = [
        (model_name, ckpt_path, opp, s, drop_prob, delay_steps, noise_prob, forced_kill_step)
        for opp in opponents
        for s in seeds
    ]

    if pool is not None:
        ep_results = pool.map(run_single_episode, tasks)
    else:
        ep_results = [run_single_episode(t) for t in tasks]

    win_rate = float(np.mean([r["win"] for r in ep_results]) * 100.0)
    draw_rate = float(np.mean([r["draw"] for r in ep_results]) * 100.0)
    loss_rate = float(np.mean([r["loss"] for r in ep_results]) * 100.0)
    mean_reward = float(np.mean([r["reward"] for r in ep_results]))

    return {
        "model": model_name,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "mean_reward": mean_reward,
        "num_episodes": len(ep_results),
    }


def main():
    print("=" * 70)
    print("  STRATOS AI v5 — PHASE 4: ROBUSTNESS TESTING SUITE")
    print("=" * 70)
    t_start = time.time()

    seeds = list(range(1000, 1050))
    opponents = ["aggressive", "defensive", "resource_focused"]

    comm_ckpt = "artifacts/checkpoints/comm_best.pt"
    no_comm_ckpt = "artifacts/checkpoints/no_comm_bc_student.pt"

    perturbations = [
        ("intact", {"drop_prob": 0.0, "delay_steps": 1, "noise_prob": 0.0, "forced_kill_step": None}),
        ("loss_0.10", {"drop_prob": 0.10, "delay_steps": 1, "noise_prob": 0.0, "forced_kill_step": None}),
        ("loss_0.30", {"drop_prob": 0.30, "delay_steps": 1, "noise_prob": 0.0, "forced_kill_step": None}),
        ("loss_0.50", {"drop_prob": 0.50, "delay_steps": 1, "noise_prob": 0.0, "forced_kill_step": None}),
        ("delay_2steps", {"drop_prob": 0.0, "delay_steps": 2, "noise_prob": 0.0, "forced_kill_step": None}),
        ("token_noise_0.10", {"drop_prob": 0.0, "delay_steps": 1, "noise_prob": 0.10, "forced_kill_step": None}),
        ("kill_scout_step50", {"drop_prob": 0.0, "delay_steps": 1, "noise_prob": 0.0, "forced_kill_step": 50}),
        ("kill_scout_step100", {"drop_prob": 0.0, "delay_steps": 1, "noise_prob": 0.0, "forced_kill_step": 100}),
    ]

    models = [
        ("Best_COMM", comm_ckpt),
        ("Oracle", None),
        ("NO_COMM", no_comm_ckpt),
    ]

    all_results = {}

    # Run with multiprocessing pool
    pool_size = min(mp.cpu_count(), 6)
    with mp.Pool(processes=pool_size) as pool:
        for m_name, m_path in models:
            all_results[m_name] = {}
            print(f"\n[Phase 4] Testing model '{m_name}' across {len(perturbations)} perturbations...")
            for p_name, p_params in perturbations:
                t0 = time.time()
                res = evaluate_perturbation(
                    model_name=m_name,
                    ckpt_path=m_path,
                    seeds=seeds,
                    opponents=opponents,
                    pool=pool,
                    **p_params,
                )
                dt = time.time() - t0
                all_results[m_name][p_name] = res
                print(f"  [{m_name} | {p_name:18s}] Win={res['win_rate']:5.1f}% | Draw={res['draw_rate']:5.1f}% | "
                      f"Loss={res['loss_rate']:5.1f}% | Reward={res['mean_reward']:+6.1f} ({dt:.1f}s)")

    # Save to JSON
    out_path = Path("artifacts/experiments/phase4_robustness_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[Phase 4 Complete] Total robustness evaluation finished in {total_time:.1f}s ({total_time/60:.2f} mins).")


if __name__ == "__main__":
    main()
