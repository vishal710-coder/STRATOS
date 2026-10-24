"""Phase 3: Held-Out Evaluation, Behavior Metrics, Message Analysis, and Ablation Study.

Evaluates on held-out seeds 5000-5049 across 3 opponents (150 episodes per condition).
- BC student (greedy & sampled)
- COMM (BC+PPO) (seeds 42, 101, 202 - greedy & sampled)
- NO_COMM (seeds 42, 101, 202 - greedy & sampled)
- Baselines: Oracle, Silent, Scripted, Random, earlier emergent COMM
- Behavior metrics: HQ camping, steps to engagement, units alive at 300, group arrival distance
- Message analysis: token vocabulary, role usage, MI(M_attack; Weak Lane), MI(M_defense; Wave Flank)
- Message ablation study: Intact vs Zeroed vs Random vs Shuffled
"""

from collections import defaultdict
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from agents.policies import ActorCriticPolicy
from agents.reference_policies import OracleMessengerPolicy, SilentReferencePolicy
from agents.targeted_policy import TargetedCombatPolicy
from training.checkpoint import load_checkpoint


def calc_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def calc_mi(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    joint = defaultdict(float)
    p_x = defaultdict(float)
    p_y = defaultdict(float)
    for xi, yi in zip(x, y):
        joint[(xi, yi)] += 1.0 / n
        p_x[xi] += 1.0 / n
        p_y[yi] += 1.0 / n
    mi = 0.0
    for (xi, yi), pxy in joint.items():
        if pxy > 0:
            mi += pxy * math.log2(pxy / (p_x[xi] * p_y[yi]))
    return max(0.0, float(mi))


def evaluate_episode(
    policy: Any,
    policy_type: str,  # 'neural_comm', 'neural_no_comm', 'oracle', 'silent', 'scripted', 'random'
    seed: int,
    opponent: str,
    deterministic: bool = True,
    ablation: str = "none",  # 'none', 'zeroed', 'random', 'shuffled'
) -> Dict[str, Any]:
    """Run a single episode and record outcomes, behavior metrics, and messages."""
    has_comm = (policy_type in ("neural_comm", "oracle"))
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

    done = False
    ep_reward = 0.0

    # Behavior tracking
    hq_steps = {"commander": 0, "defense": 0, "attack": 0}
    alive_steps = {"commander": 0, "defense": 0, "attack": 0}
    first_engagement_step = None
    together_distances = []

    # Message tracking
    sent_messages = []  # dict of role, token, weak_lane, wave_flank, step

    while not done:
        step_idx = env.state.step_count if env.state else 0
        fb = env.state.bases["friendly"]

        # Track HQ distance for each alive unit
        for ag in env.possible_agents:
            u = env.state.friendly_units.get(ag)
            if u and u.is_alive:
                alive_steps[ag] += 1
                dist_hq = ((u.x - fb.x)**2 + (u.y - fb.y)**2)**0.5
                if dist_hq <= 120.0:
                    hq_steps[ag] += 1

        # Check combat engagement & group cohesion
        alive_units = [u for u in env.state.friendly_units.values() if u.is_alive]
        alive_enemies = [e for e in env.state.enemy_units.values() if e.is_alive]
        in_combat = any(any(u.distance_to(e.x, e.y) <= u.attack_range for e in alive_enemies) for u in alive_units)
        if in_combat and len(alive_units) >= 2:
            if first_engagement_step is None:
                first_engagement_step = step_idx
            # Max pairwise distance among engaging units
            max_d = 0.0
            for i in range(len(alive_units)):
                for j in range(i + 1, len(alive_units)):
                    d = alive_units[i].distance_to(alive_units[j].x, alive_units[j].y)
                    if d > max_d:
                        max_d = d
            together_distances.append(max_d)

        # Decide actions
        actions_dict = {}
        if policy_type == "neural_comm":
            obs_tensor = torch.as_tensor(
                np.array([obs[ag] for ag in env.possible_agents]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_tensor, deterministic=deterministic)

            for i, ag in enumerate(env.possible_agents):
                act = int(a_t[i].item())
                tok = int(m_t[i].item())

                # Apply message ablation if active
                if ablation == "zeroed":
                    tok = 0
                elif ablation == "random":
                    tok = int(np.random.randint(1, 9))
                # (shuffled is handled below across step)

                actions_dict[ag] = {"action": act, "message": tok}
                sent_messages.append({
                    "agent": ag,
                    "token": tok,
                    "weak_lane": env.state.weak_lane,
                    "wave_flank": env.state.wave_flank,
                    "step": step_idx,
                })

            if ablation == "shuffled":
                # Randomly permute messages across agents
                toks = [actions_dict[ag]["message"] for ag in env.possible_agents]
                np.random.shuffle(toks)
                for i, ag in enumerate(env.possible_agents):
                    actions_dict[ag]["message"] = toks[i]

        elif policy_type == "neural_no_comm":
            obs_tensor = torch.as_tensor(
                np.array([obs[ag] for ag in env.possible_agents]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                a_t, _, _, _ = policy.get_action_and_value(obs_tensor, deterministic=deterministic)
            for i, ag in enumerate(env.possible_agents):
                actions_dict[ag] = int(a_t[i].item())

        elif policy_type == "oracle":
            for ag in env.possible_agents:
                act, tok = policy.select_action_and_message(ag, obs[ag])
                actions_dict[ag] = {"action": act, "message": tok}
                sent_messages.append({
                    "agent": ag,
                    "token": tok,
                    "weak_lane": env.state.weak_lane,
                    "wave_flank": env.state.wave_flank,
                    "step": step_idx,
                })

        elif policy_type == "silent":
            for ag in env.possible_agents:
                act, tok = policy.select_action_and_message(ag, obs[ag])
                actions_dict[ag] = {"action": act, "message": tok}

        elif policy_type == "scripted":
            for ag in env.possible_agents:
                act = policy.select_action(ag, obs[ag])
                actions_dict[ag] = act

        elif policy_type == "random":
            for ag in env.possible_agents:
                actions_dict[ag] = int(np.random.randint(0, 9))

        obs, rewards, term, trunc, _ = env.step(actions_dict)
        ep_reward += sum(rewards.values()) / len(rewards)
        done = any(term.values()) or any(trunc.values())

    winner = env.state.winner if env.state else "draw"
    units_alive_300 = sum(1 for u in env.state.friendly_units.values() if u.is_alive) if env.state else 0
    hq_fracs = {
        ag: (hq_steps[ag] / max(1, alive_steps[ag])) for ag in env.possible_agents
    }
    mean_group_dist = float(np.mean(together_distances)) if together_distances else 0.0

    return {
        "seed": seed,
        "opponent": opponent,
        "winner": winner,
        "win": winner == "friendly",
        "loss": winner == "enemy",
        "draw": winner not in ("friendly", "enemy"),
        "reward": float(ep_reward),
        "steps": env.state.step_count if env.state else 300,
        "hq_fracs": hq_fracs,
        "first_engagement": first_engagement_step if first_engagement_step is not None else 300,
        "units_alive_300": units_alive_300,
        "mean_group_dist": mean_group_dist,
        "messages": sent_messages,
    }


def evaluate_condition_worker(args: Tuple[str, str, str, bool, str, List[int], List[str]]) -> Dict[str, Any]:
    """Worker evaluating a full condition across 50 seeds x 3 opponents."""
    torch.set_num_threads(1)
    condition_id, ckpt_path, policy_type, deterministic, ablation, seeds, opponents = args

    policy = None
    if policy_type.startswith("neural"):
        policy, _ = load_checkpoint(ckpt_path, device="cpu")
        policy.eval()
    elif policy_type == "oracle":
        policy = OracleMessengerPolicy()
    elif policy_type == "silent":
        policy = SilentReferencePolicy()
    elif policy_type == "scripted":
        policy = TargetedCombatPolicy()

    ep_results = []
    all_messages = []

    for opp in opponents:
        for s in seeds:
            res = evaluate_episode(
                policy=policy,
                policy_type=policy_type,
                seed=s,
                opponent=opp,
                deterministic=deterministic,
                ablation=ablation,
            )
            all_messages.extend(res.pop("messages"))
            ep_results.append(res)

    win_rate = float(np.mean([r["win"] for r in ep_results]) * 100.0)
    draw_rate = float(np.mean([r["draw"] for r in ep_results]) * 100.0)
    loss_rate = float(np.mean([r["loss"] for r in ep_results]) * 100.0)
    mean_reward = float(np.mean([r["reward"] for r in ep_results]))
    std_reward = float(np.std([r["reward"] for r in ep_results]))

    # Behavior metrics
    hq_cmd = float(np.mean([r["hq_fracs"]["commander"] for r in ep_results]) * 100.0)
    hq_def = float(np.mean([r["hq_fracs"]["defense"] for r in ep_results]) * 100.0)
    hq_atk = float(np.mean([r["hq_fracs"]["attack"] for r in ep_results]) * 100.0)
    eng_step = float(np.mean([r["first_engagement"] for r in ep_results]))
    alive_300 = float(np.mean([r["units_alive_300"] for r in ep_results]))
    group_dist = float(np.mean([r["mean_group_dist"] for r in ep_results]))

    # Message analysis
    msg_analysis = {}
    if all_messages:
        df_msg = pd.DataFrame(all_messages)
        total_tokens = len(df_msg)
        token_counts = df_msg["token"].value_counts().to_dict()
        used_tokens = sum(1 for t, c in token_counts.items() if (c / max(1, total_tokens)) >= 0.01)

        # Per role tokens
        role_breakdown = {}
        for role in ["commander", "defense", "attack"]:
            sub = df_msg[df_msg["agent"] == role]
            r_tot = max(1, len(sub))
            counts = [int((sub["token"] == t).sum()) for t in range(9)]
            freqs = [round(c / r_tot, 4) for c in counts]
            role_breakdown[role] = freqs

        # Mutual Information
        sub_atk = df_msg[df_msg["agent"] == "attack"]
        mi_weak = calc_mi(sub_atk["token"].to_numpy(), sub_atk["weak_lane"].to_numpy()) if len(sub_atk) > 0 else 0.0

        sub_def = df_msg[df_msg["agent"] == "defense"]
        mi_flank = calc_mi(sub_def["token"].to_numpy(), sub_def["wave_flank"].to_numpy()) if len(sub_def) > 0 else 0.0

        msg_analysis = {
            "tokens_used_out_of_9": used_tokens,
            "role_token_frequencies": role_breakdown,
            "mi_attack_weak_lane": round(mi_weak, 4),
            "mi_defense_wave_flank": round(mi_flank, 4),
        }

    return {
        "condition_id": condition_id,
        "checkpoint": str(ckpt_path),
        "policy_type": policy_type,
        "deterministic": deterministic,
        "ablation": ablation,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "behavior": {
            "hq_commander_pct": round(hq_cmd, 1),
            "hq_defense_pct": round(hq_def, 1),
            "hq_attack_pct": round(hq_atk, 1),
            "steps_to_first_engagement": round(eng_step, 1),
            "units_alive_step_300": round(alive_300, 2),
            "mean_combat_group_dist": round(group_dist, 1),
        },
        "message_analysis": msg_analysis,
        "episodes": ep_results,
    }


def main():
    print("=" * 65)
    print("  STRATOS AI — PHASE 3: EVALUATION & REPORT GENERATION")
    print("=" * 65)
    t_start = time.time()

    seeds = list(range(5000, 5050))  # 5000 to 5049
    opponents = ["aggressive", "defensive", "resource_focused"]
    out_dir = Path("artifacts/experiments/v4_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 2 selection summary to get selected checkpoints per seed
    p2_summary_file = Path("artifacts/experiments/v4_finetuned/phase2_validation_selection.json")
    if p2_summary_file.exists():
        with open(p2_summary_file, "r") as f:
            p2_summary = json.load(f)
            selected = p2_summary.get("selected_checkpoints", {})
            comm_s42_ckpt = selected.get("comm_s42", {}).get("checkpoint", "artifacts/checkpoints/v4_finetuned/comm_s42.pt")
            comm_s101_ckpt = selected.get("comm_s101", {}).get("checkpoint", "artifacts/checkpoints/v4_finetuned/comm_s101.pt")
            comm_s202_ckpt = selected.get("comm_s202", {}).get("checkpoint", "artifacts/checkpoints/v4_finetuned/comm_s202_step_75000.pt")
            no_comm_s42_ckpt = selected.get("no_comm_s42", {}).get("checkpoint", "artifacts/checkpoints/v4_finetuned/no_comm_s42_step_75000.pt")
            no_comm_s101_ckpt = selected.get("no_comm_s101", {}).get("checkpoint", "artifacts/checkpoints/v4_finetuned/no_comm_s101_step_75000.pt")
            no_comm_s202_ckpt = selected.get("no_comm_s202", {}).get("checkpoint", "artifacts/checkpoints/v4_finetuned/no_comm_s202_step_75000.pt")
    else:
        comm_s42_ckpt = "artifacts/checkpoints/v4_finetuned/comm_s42.pt"
        comm_s101_ckpt = "artifacts/checkpoints/v4_finetuned/comm_s101.pt"
        comm_s202_ckpt = "artifacts/checkpoints/v4_finetuned/comm_s202_step_75000.pt"
        no_comm_s42_ckpt = "artifacts/checkpoints/v4_finetuned/no_comm_s42_step_75000.pt"
        no_comm_s101_ckpt = "artifacts/checkpoints/v4_finetuned/no_comm_s101_step_75000.pt"
        no_comm_s202_ckpt = "artifacts/checkpoints/v4_finetuned/no_comm_s202_step_75000.pt"

    bc_ckpt = "artifacts/checkpoints/comm_bc_student.pt"
    old_comm_ckpt = "artifacts/checkpoints/v3_resumed/comm_s101.pt"

    eval_tasks = [
        # BC Student (greedy & sampled)
        ("bc_student_greedy", bc_ckpt, "neural_comm", True, "none", seeds, opponents),
        ("bc_student_sampled", bc_ckpt, "neural_comm", False, "none", seeds, opponents),

        # COMM (BC+PPO) (3 seeds: greedy & sampled)
        ("comm_ppo_s42_greedy", comm_s42_ckpt, "neural_comm", True, "none", seeds, opponents),
        ("comm_ppo_s42_sampled", comm_s42_ckpt, "neural_comm", False, "none", seeds, opponents),
        ("comm_ppo_s101_greedy", comm_s101_ckpt, "neural_comm", True, "none", seeds, opponents),
        ("comm_ppo_s101_sampled", comm_s101_ckpt, "neural_comm", False, "none", seeds, opponents),
        ("comm_ppo_s202_greedy", comm_s202_ckpt, "neural_comm", True, "none", seeds, opponents),
        ("comm_ppo_s202_sampled", comm_s202_ckpt, "neural_comm", False, "none", seeds, opponents),

        # NO_COMM (3 seeds: greedy & sampled)
        ("no_comm_s42_greedy", no_comm_s42_ckpt, "neural_no_comm", True, "none", seeds, opponents),
        ("no_comm_s42_sampled", no_comm_s42_ckpt, "neural_no_comm", False, "none", seeds, opponents),
        ("no_comm_s101_greedy", no_comm_s101_ckpt, "neural_no_comm", True, "none", seeds, opponents),
        ("no_comm_s101_sampled", no_comm_s101_ckpt, "neural_no_comm", False, "none", seeds, opponents),
        ("no_comm_s202_greedy", no_comm_s202_ckpt, "neural_no_comm", True, "none", seeds, opponents),
        ("no_comm_s202_sampled", no_comm_s202_ckpt, "neural_no_comm", False, "none", seeds, opponents),

        # Reference Baselines on 5000-5049
        ("oracle_reference", "none", "oracle", True, "none", seeds, opponents),
        ("silent_reference", "none", "silent", True, "none", seeds, opponents),
        ("scripted_heuristic", "none", "scripted", True, "none", seeds, opponents),
        ("random_policy", "none", "random", False, "none", seeds, opponents),

        # Old COMM (v3) behavior baseline
        ("old_comm_s101", old_comm_ckpt, "neural_comm", True, "none", seeds, opponents),

        # Message Ablations on BC Student & Best COMM (s101) on the same 150 episodes
        ("bc_student_zeroed", bc_ckpt, "neural_comm", True, "zeroed", seeds, opponents),
        ("bc_student_random", bc_ckpt, "neural_comm", True, "random", seeds, opponents),
        ("bc_student_shuffled", bc_ckpt, "neural_comm", True, "shuffled", seeds, opponents),
        ("comm_ppo_s101_zeroed", comm_s101_ckpt, "neural_comm", True, "zeroed", seeds, opponents),
        ("comm_ppo_s101_random", comm_s101_ckpt, "neural_comm", True, "random", seeds, opponents),
        ("comm_ppo_s101_shuffled", comm_s101_ckpt, "neural_comm", True, "shuffled", seeds, opponents),
    ]

    print(f"[Phase 3] Launching {len(eval_tasks)} evaluation tasks across CPU pool (50 seeds x 3 opps = 150 eps each)...")
    with mp.Pool(processes=min(12, mp.cpu_count())) as pool:
        all_results = pool.map(evaluate_condition_worker, eval_tasks)

    print(f"[Phase 3] All evaluation tasks completed in {time.time() - t_start:.1f}s.")

    # Save complete evaluation output
    res_dict = {r["condition_id"]: r for r in all_results}
    summary_save_path = out_dir / "heldout_evaluation_summary.json"
    with open(summary_save_path, "w") as f:
        # Exclude raw episode dumps from JSON summary for compact reading
        clean_summary = {
            cid: {k: v for k, v in r.items() if k != "episodes"}
            for cid, r in res_dict.items()
        }
        json.dump(clean_summary, f, indent=2)

    # Print Full Scoreboard
    print("\n" + "=" * 90)
    print("  STRATOS AI — HELD-OUT TEST SCOREBOARD (SEEDS 5000-5049, 150 EPISODES PER CONDITION)")
    print("=" * 90)
    print(f"{'Condition ID':25s} | {'Win %':7s} | {'Draw %':7s} | {'Loss %':7s} | {'Mean Reward (Std)':22s} | {'HQ Def %':9s}")
    print("-" * 90)
    for cid, r in clean_summary.items():
        hq_d = r.get("behavior", {}).get("hq_defense_pct", 0.0)
        print(f"{cid:25s} | {r['win_rate']:6.1f}% | {r['draw_rate']:6.1f}% | {r['loss_rate']:6.1f}% | {r['mean_reward']:+6.2f} (±{r['std_reward']:5.1f}) | {hq_d:6.1f}%")

    print(f"\nSaved evaluation summary to: {summary_save_path}")
    print(f"Phase 3 Complete. Total Duration: {time.time() - t_start:.1f}s ({(time.time() - t_start)/60:.2f} mins).")


if __name__ == "__main__":
    main()
