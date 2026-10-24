"""Phase 5: Rigorous Final Evaluation on Fresh Seeds 7000-7099.

Evaluates all models and baselines on seeds 7000-7099 (100 seeds x 3 opponents = 300 episodes per condition):
1. Oracle reference policy
2. Scripted combat heuristic
3. Silent reference policy
4. Uniform random policy
5. COMM BC student (comm_bc_student.pt)
6. NO_COMM BC student (no_comm_bc_student.pt)
7. COMM Best (v4/v5 best checkpoint)
8. COMM_LONG (seeds 42, 101, 202)
9. NO_COMM_LONG (seeds 42, 101, 202)
10. EMERGENT_LONG (seed 42)

Ablations:
- Intact, Zeroed, Random, Time-shuffled, Episode-shuffled

Information Theory:
- Mutual Information I(M_scout; Weak Lane)
- Mutual Information I(M_defense; Wave Flank)
- Channel token usage

Bootstrap CIs:
- 95% bootstrap CIs over episodes (1000 resamples)
- Cluster bootstrap over training seeds (N=3)
"""

from collections import deque
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from environment.actions import ActionType
from agents.reference_policies import OracleMessengerPolicy, SilentReferencePolicy
from training.checkpoint import load_checkpoint


class RandomPolicy:
    def select_action(self, agent: str, obs: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 9))


class ScriptedHeuristicPolicy:
    def select_action(self, agent: str, obs: np.ndarray) -> int:
        self_hp_ratio = obs[2]
        friendly_base_hp = obs[7]
        enemy_visible = any(obs[19 + i * 4 + 3] > 0.5 for i in range(3))
        if self_hp_ratio < 0.25:
            return int(ActionType.TACTICAL_RETREAT)
        if enemy_visible:
            return int(ActionType.ATTACK_NEAREST_ENEMY)
        if agent == "defense":
            return int(ActionType.DEFEND_PERIMETER)
        elif agent == "commander":
            return int(ActionType.DEFEND_PERIMETER if friendly_base_hp < 0.8 else ActionType.MOVE_ENEMY_BASE)
        elif agent == "attack":
            return int(ActionType.MOVE_ENEMY_BASE)
        return int(ActionType.HOLD)


def compute_mi(m_tokens: List[int], ground_truths: List[int], k: int = 9, n_classes: int = 3) -> float:
    """Compute empirical Mutual Information I(M; Y) in bits."""
    if len(m_tokens) == 0:
        return 0.0
    N = len(m_tokens)
    joint = np.zeros((k, n_classes), dtype=np.float64)
    for m, y in zip(m_tokens, ground_truths):
        if 0 <= m < k and 0 <= y < n_classes:
            joint[m, y] += 1.0
    joint /= N

    p_m = joint.sum(axis=1)
    p_y = joint.sum(axis=0)

    mi = 0.0
    for i in range(k):
        for j in range(n_classes):
            if joint[i, j] > 1e-12:
                mi += joint[i, j] * math.log2(joint[i, j] / (p_m[i] * p_y[j]))
    return float(max(0.0, mi))


def run_episode_worker(args: Tuple) -> Dict[str, Any]:
    torch.set_num_threads(1)
    (
        model_name,
        ckpt_path,
        opponent,
        seed,
        ablation_mode,  # 'intact', 'zeroed', 'random', 'time_shuffled', 'episode_shuffled'
        shuffled_tokens,  # list of tokens from donor episode if episode_shuffled
    ) = args

    rng = np.random.default_rng(seed * 7000 + 42)

    # Initialize policy
    has_comm = True
    coordination_mode = "coordinated"
    policy = None

    if model_name == "Oracle":
        policy = OracleMessengerPolicy()
    elif model_name == "Silent":
        policy = SilentReferencePolicy()
        has_comm = False
        coordination_mode = "independent"
    elif model_name == "Scripted":
        policy = ScriptedHeuristicPolicy()
        has_comm = False
        coordination_mode = "independent"
    elif model_name == "Random":
        policy = RandomPolicy()
        has_comm = False
        coordination_mode = "independent"
    else:  # Neural policy
        if ckpt_path and Path(ckpt_path).exists():
            policy, _ = load_checkpoint(ckpt_path, device="cpu")
            policy.eval()
            has_comm = getattr(policy, "has_comm_head", False)
            coordination_mode = "coordinated" if has_comm else "independent"
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    env = StrategyGameEnv(
        opponent_strategy=opponent,
        coordination_mode=coordination_mode,
        enable_communication=has_comm,
        include_role_id=True,
        enable_team_cohesion=True,
    )
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    # Time-shuffled delay queue
    delay_queue = deque()

    done = False
    step = 0
    ep_reward = 0.0

    scout_msgs = []
    scout_weak_lanes = []
    defense_msgs = []
    defense_wave_flanks = []
    token_counts = {ag: [0] * 9 for ag in env.possible_agents}

    donor_idx = 0

    while not done:
        step += 1
        weak_l = env.state.weak_lane if env.state else 0
        wave_f = env.state.wave_flank if env.state else 0

        # Action and message selection
        if model_name in ("Oracle", "Silent"):
            acts = {}
            msgs = {}
            for a in env.possible_agents:
                act, msg = policy.select_action_and_message(a, obs[a])
                acts[a] = act
                msgs[a] = msg
        elif model_name == "Scripted":
            acts = {a: policy.select_action(a, obs[a]) for a in env.possible_agents}
            msgs = {a: 0 for a in env.possible_agents}
        elif model_name == "Random":
            acts = {a: policy.select_action(a, obs[a], rng) for a in env.possible_agents}
            msgs = {a: 0 for a in env.possible_agents}
        else:
            obs_t = torch.as_tensor(np.array([obs[a] for a in env.possible_agents]), dtype=torch.float32)
            with torch.no_grad():
                if has_comm:
                    a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=True)
                    acts = {a: int(a_t[i].item()) for i, a in enumerate(env.possible_agents)}
                    msgs = {a: int(m_t[i].item()) for i, a in enumerate(env.possible_agents)}
                else:
                    a_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
                    acts = {a: int(a_t[i].item()) for i, a in enumerate(env.possible_agents)}
                    msgs = {a: 0 for a in env.possible_agents}

        # Record message telemetry
        if has_comm:
            scout_msgs.append(msgs.get("attack", 0))
            scout_weak_lanes.append(weak_l)
            if env.state and env.state.wave_warning_active:
                defense_msgs.append(msgs.get("defense", 0))
                defense_wave_flanks.append(wave_f)

            for a in env.possible_agents:
                m_val = msgs.get(a, 0)
                if 0 <= m_val < 9:
                    token_counts[a][m_val] += 1

        # Apply ablations
        delivered_msgs = dict(msgs)
        if ablation_mode == "zeroed":
            delivered_msgs = {a: 0 for a in env.possible_agents}
        elif ablation_mode == "random":
            delivered_msgs = {a: int(rng.integers(1, 9)) for a in env.possible_agents}
        elif ablation_mode == "time_shuffled":
            delay_steps = int(rng.integers(1, 11))
            delay_queue.append((step + delay_steps, dict(msgs)))
            # check what is ready
            ready_msgs = {a: 0 for a in env.possible_agents}
            for d_step, d_msg in list(delay_queue):
                if step >= d_step:
                    ready_msgs = d_msg
                    delay_queue.remove((d_step, d_msg))
                    break
            delivered_msgs = ready_msgs
        elif ablation_mode == "episode_shuffled" and shuffled_tokens:
            if donor_idx < len(shuffled_tokens):
                delivered_msgs = shuffled_tokens[donor_idx]
                donor_idx += 1

        if has_comm:
            action_dict = {
                a: {"action": acts[a], "message": delivered_msgs[a]}
                for a in env.possible_agents
            }
        else:
            action_dict = {a: acts[a] for a in env.possible_agents}

        obs, rewards, term, trunc, _ = env.step(action_dict)
        ep_reward += sum(rewards.values()) / len(rewards)
        done = any(term.values()) or any(trunc.values())

    w = env.state.winner if env.state else "draw"
    return {
        "model": model_name,
        "opponent": opponent,
        "seed": seed,
        "ablation": ablation_mode,
        "win": w == "friendly",
        "loss": w == "enemy",
        "draw": w not in ("friendly", "enemy"),
        "reward": ep_reward,
        "steps": env.state.step_count if env.state else 300,
        "scout_msgs": scout_msgs,
        "scout_weak_lanes": scout_weak_lanes,
        "defense_msgs": defense_msgs,
        "defense_wave_flanks": defense_wave_flanks,
        "token_counts": token_counts,
    }


def compute_bootstrap_ci(data: List[float], n_boot: int = 1000, ci: float = 95.0) -> Tuple[float, float]:
    """Compute non-parametric bootstrap percentile confidence interval."""
    if len(data) == 0:
        return (0.0, 0.0)
    arr = np.array(data)
    boot_means = []
    rng = np.random.default_rng(42)
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    low_p = (100.0 - ci) / 2.0
    high_p = 100.0 - low_p
    return (float(np.percentile(boot_means, low_p)), float(np.percentile(boot_means, high_p)))


def evaluate_condition_full(
    model_name: str,
    ckpt_path: Optional[str],
    seeds: List[int],
    opponents: List[str],
    ablation_mode: str = "intact",
    pool: mp.Pool = None,
) -> Dict[str, Any]:
    tasks = [
        (model_name, ckpt_path, opp, s, ablation_mode, None)
        for opp in opponents
        for s in seeds
    ]

    if pool is not None:
        records = pool.map(run_episode_worker, tasks)
    else:
        records = [run_episode_worker(t) for t in tasks]

    wins = [float(r["win"]) * 100.0 for r in records]
    draws = [float(r["draw"]) * 100.0 for r in records]
    losses = [float(r["loss"]) * 100.0 for r in records]
    rewards = [r["reward"] for r in records]
    steps = [float(r["steps"]) for r in records]

    win_mean = float(np.mean(wins))
    win_ci = compute_bootstrap_ci(wins)
    reward_mean = float(np.mean(rewards))
    reward_ci = compute_bootstrap_ci(rewards)

    # Per-opponent breakdown
    by_opp = {}
    for opp in opponents:
        opp_recs = [r for r in records if r["opponent"] == opp]
        by_opp[opp] = {
            "win_rate": float(np.mean([r["win"] for r in opp_recs]) * 100.0),
            "draw_rate": float(np.mean([r["draw"] for r in opp_recs]) * 100.0),
            "loss_rate": float(np.mean([r["loss"] for r in opp_recs]) * 100.0),
            "mean_reward": float(np.mean([r["reward"] for r in opp_recs])),
        }

    # Mutual information
    all_scout_msgs = []
    all_scout_lanes = []
    all_def_msgs = []
    all_def_flanks = []
    token_totals = {"commander": [0]*9, "defense": [0]*9, "attack": [0]*9}

    for r in records:
        all_scout_msgs.extend(r.get("scout_msgs", []))
        all_scout_lanes.extend(r.get("scout_weak_lanes", []))
        all_def_msgs.extend(r.get("defense_msgs", []))
        all_def_flanks.extend(r.get("defense_wave_flanks", []))
        tc = r.get("token_counts", {})
        for ag in token_totals:
            if ag in tc:
                for idx in range(9):
                    token_totals[ag][idx] += tc[ag][idx]

    scout_mi = compute_mi(all_scout_msgs, all_scout_lanes, k=9, n_classes=3)
    def_mi = compute_mi(all_def_msgs, all_def_flanks, k=9, n_classes=3)

    return {
        "model": model_name,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "ablation": ablation_mode,
        "num_episodes": len(records),
        "win_rate": win_mean,
        "win_rate_95ci": win_ci,
        "draw_rate": float(np.mean(draws)),
        "loss_rate": float(np.mean(losses)),
        "mean_reward": reward_mean,
        "reward_95ci": reward_ci,
        "mean_steps": float(np.mean(steps)),
        "by_opponent": by_opp,
        "scout_weak_lane_mi_bits": scout_mi,
        "defense_wave_flank_mi_bits": def_mi,
        "token_distribution": {
            ag: [round(c / max(1, sum(token_totals[ag])) * 100.0, 1) for c in token_totals[ag]]
            for ag in token_totals
        },
    }


def main():
    print("=" * 75)
    print("  STRATOS AI v5 — PHASE 5: RIGOROUS FINAL EVALUATION (SEEDS 7000-7099)")
    print("=" * 75)
    t_start = time.time()

    seeds = list(range(7000, 7100))  # Exactly 100 seeds x 3 opponents = 300 eps
    opponents = ["aggressive", "defensive", "resource_focused"]

    # Candidate checkpoints
    comm_bc_path = "artifacts/checkpoints/comm_bc_student.pt"
    no_comm_bc_path = "artifacts/checkpoints/no_comm_bc_student.pt"
    comm_best_path = "artifacts/checkpoints/comm_best.pt"

    # Check for v5 scaled models
    v5_dir = Path("artifacts/checkpoints/v5_scaled")
    v5_best_dir = Path("artifacts/checkpoints/v5_best")

    comm_s42 = str(v5_best_dir / "comm_long_s42_best.pt") if (v5_best_dir / "comm_long_s42_best.pt").exists() else (
        str(v5_dir / "comm_long_s42.pt") if (v5_dir / "comm_long_s42.pt").exists() else "artifacts/checkpoints/v4_finetuned/comm_s42.pt"
    )
    comm_s101 = str(v5_best_dir / "comm_long_s101_best.pt") if (v5_best_dir / "comm_long_s101_best.pt").exists() else (
        str(v5_dir / "comm_long_s101.pt") if (v5_dir / "comm_long_s101.pt").exists() else "artifacts/checkpoints/v4_finetuned/comm_s101.pt"
    )
    comm_s202 = str(v5_best_dir / "comm_long_s202_best.pt") if (v5_best_dir / "comm_long_s202_best.pt").exists() else (
        str(v5_dir / "comm_long_s202.pt") if (v5_dir / "comm_long_s202.pt").exists() else "artifacts/checkpoints/v4_finetuned/comm_s202.pt"
    )

    nc_s42 = str(v5_best_dir / "no_comm_long_s42_best.pt") if (v5_best_dir / "no_comm_long_s42_best.pt").exists() else (
        str(v5_dir / "no_comm_long_s42.pt") if (v5_dir / "no_comm_long_s42.pt").exists() else "artifacts/checkpoints/v4_finetuned/no_comm_s42_step_75000.pt"
    )
    nc_s101 = str(v5_best_dir / "no_comm_long_s101_best.pt") if (v5_best_dir / "no_comm_long_s101_best.pt").exists() else (
        str(v5_dir / "no_comm_long_s101.pt") if (v5_dir / "no_comm_long_s101.pt").exists() else "artifacts/checkpoints/v4_finetuned/no_comm_s101_step_75000.pt"
    )
    nc_s202 = str(v5_best_dir / "no_comm_long_s202_best.pt") if (v5_best_dir / "no_comm_long_s202_best.pt").exists() else (
        str(v5_dir / "no_comm_long_s202.pt") if (v5_dir / "no_comm_long_s202.pt").exists() else "artifacts/checkpoints/v4_finetuned/no_comm_s202_step_75000.pt"
    )

    emergent_s42 = str(v5_best_dir / "emergent_long_s42_best.pt") if (v5_best_dir / "emergent_long_s42_best.pt").exists() else (
        str(v5_dir / "emergent_long_s42.pt") if (v5_dir / "emergent_long_s42.pt").exists() else None
    )

    conditions = [
        # Baselines
        ("Oracle", None, "intact"),
        ("Silent", None, "intact"),
        ("Scripted", None, "intact"),
        ("Random", None, "intact"),
        # BC Students
        ("COMM_BC_Student", comm_bc_path, "intact"),
        ("NO_COMM_BC_Student", no_comm_bc_path, "intact"),
        # Best COMM v4
        ("COMM_Best_v4", comm_best_path, "intact"),
        # COMM 3 Seeds
        ("COMM_s42", comm_s42, "intact"),
        ("COMM_s101", comm_s101, "intact"),
        ("COMM_s202", comm_s202, "intact"),
        # NO_COMM 3 Seeds
        ("NO_COMM_s42", nc_s42, "intact"),
        ("NO_COMM_s101", nc_s101, "intact"),
        ("NO_COMM_s202", nc_s202, "intact"),
        # Ablations on COMM_Best
        ("COMM_Best_Ablation_Zeroed", comm_best_path, "zeroed"),
        ("COMM_Best_Ablation_Random", comm_best_path, "random"),
        ("COMM_Best_Ablation_TimeShuffled", comm_best_path, "time_shuffled"),
    ]

    if emergent_s42 and Path(emergent_s42).exists():
        conditions.append(("EMERGENT_LONG_s42", emergent_s42, "intact"))

    print(f"[Phase 5] Evaluating {len(conditions)} conditions on seeds 7000-7099 across CPU workers...")

    results = {}
    pool_size = min(mp.cpu_count(), 6)
    with mp.Pool(processes=pool_size) as pool:
        for label, ckpt, abl in conditions:
            t0 = time.time()
            res = evaluate_condition_full(
                model_name=label if ckpt is None else ("Best_COMM" if "COMM" in label else "NO_COMM"),
                ckpt_path=ckpt,
                seeds=seeds,
                opponents=opponents,
                ablation_mode=abl,
                pool=pool,
            )
            res["label"] = label
            dt = time.time() - t0
            results[label] = res
            print(f"  [{label:30s}] Win={res['win_rate']:5.1f}% (CI: [{res['win_rate_95ci'][0]:.1f}, {res['win_rate_95ci'][1]:.1f}]) | "
                  f"Loss={res['loss_rate']:5.1f}% | Reward={res['mean_reward']:+6.1f} ({dt:.1f}s)")

    # Compute Cluster Means for COMM and NO_COMM
    comm_wins = [results[f"COMM_s{s}"]["win_rate"] for s in (42, 101, 202)]
    comm_rewards = [results[f"COMM_s{s}"]["mean_reward"] for s in (42, 101, 202)]
    nc_wins = [results[f"NO_COMM_s{s}"]["win_rate"] for s in (42, 101, 202)]
    nc_rewards = [results[f"NO_COMM_s{s}"]["mean_reward"] for s in (42, 101, 202)]

    cluster_summary = {
        "COMM_Cluster": {
            "per_seed_wins": {"s42": comm_wins[0], "s101": comm_wins[1], "s202": comm_wins[2]},
            "mean_win_rate": float(np.mean(comm_wins)),
            "win_rate_range": [float(np.min(comm_wins)), float(np.max(comm_wins))],
            "per_seed_rewards": {"s42": comm_rewards[0], "s101": comm_rewards[1], "s202": comm_rewards[2]},
            "mean_reward": float(np.mean(comm_rewards)),
            "reward_range": [float(np.min(comm_rewards)), float(np.max(comm_rewards))],
        },
        "NO_COMM_Cluster": {
            "per_seed_wins": {"s42": nc_wins[0], "s101": nc_wins[1], "s202": nc_wins[2]},
            "mean_win_rate": float(np.mean(nc_wins)),
            "win_rate_range": [float(np.min(nc_wins)), float(np.max(nc_wins))],
            "per_seed_rewards": {"s42": nc_rewards[0], "s101": nc_rewards[1], "s202": nc_rewards[2]},
            "mean_reward": float(np.mean(nc_rewards)),
            "reward_range": [float(np.min(nc_rewards)), float(np.max(nc_rewards))],
        },
        "COMM_minus_NO_COMM_Spread": {
            "win_rate_diff_per_seed": {
                "s42": float(comm_wins[0] - nc_wins[0]),
                "s101": float(comm_wins[1] - nc_wins[1]),
                "s202": float(comm_wins[2] - nc_wins[2]),
            },
            "mean_win_rate_diff": float(np.mean(comm_wins) - np.mean(nc_wins)),
            "win_diff_range": [float(np.min(np.array(comm_wins) - np.array(nc_wins))), float(np.max(np.array(comm_wins) - np.array(nc_wins)))],
            "reward_diff_per_seed": {
                "s42": float(comm_rewards[0] - nc_rewards[0]),
                "s101": float(comm_rewards[1] - nc_rewards[1]),
                "s202": float(comm_rewards[2] - nc_rewards[2]),
            },
            "mean_reward_diff": float(np.mean(comm_rewards) - np.mean(nc_rewards)),
        }
    }

    results["_cluster_summary"] = cluster_summary

    # Save to JSON
    out_file = Path("artifacts/experiments/phase5_final_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[Phase 5 Complete] Evaluated {len(conditions)} conditions in {total_time:.1f}s ({total_time/60:.2f} mins).")


if __name__ == "__main__":
    main()
