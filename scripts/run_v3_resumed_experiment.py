"""STRATOS v3 Resumed Training & Rigorous Evaluation Pipeline.

Executes:
1. Resuming existing 200k checkpoints for seeds 42, 101, 202 across 3 conditions:
   - NO_COMM: independent tactical actors.
   - COMM: coordinated dual-head communication (aux_pred_loss_coef=0.0).
   - COMM_AUX: resumed from COMM with aux_pred_loss_coef=0.1.
   Trains for +400,000 additional steps (total 600,000 steps) in parallel across cores.
   Saves checkpoints every 50,000 steps.
2. Evaluates on held-out seeds 5000-5049 across all 3 opponents (150 eps per checkpoint).
3. Evaluates message statistics (tokens used, entropy, mutual information with weak lane / flank).
4. Evaluates message ablations (intact vs zeroed vs random vs shuffled) on the same episodes.
5. Generates comprehensive JSON/CSV reports and updates docs/WORKLOG.md.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from training.train import train_stratos
from environment.strategy_game import StrategyGameEnv
from agents.policies import ActorCriticPolicy
from training.checkpoint import load_checkpoint
from agents.reference_policies import OracleMessengerPolicy, SilentReferencePolicy
from agents.targeted_policy import TargetedCombatPolicy


def cluster_bootstrap_ci(per_seed_values: Dict[int, List[float]], n_bootstraps: int = 2000, ci: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """Cluster bootstrap over training seeds."""
    seeds = list(per_seed_values.keys())
    k = len(seeds)
    if k <= 1:
        mean_val = float(np.mean([np.mean(v) for v in per_seed_values.values()]))
        return mean_val, (mean_val, mean_val)

    boot_means = []
    rng = np.random.default_rng(12345)
    for _ in range(n_bootstraps):
        sampled_seeds = rng.choice(seeds, size=k, replace=True)
        sample_data = []
        for s in sampled_seeds:
            sample_data.extend(per_seed_values[s])
        boot_means.append(np.mean(sample_data))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    overall_mean = float(np.mean([np.mean(v) for v in per_seed_values.values()]))
    return overall_mean, (lower, upper)


def compute_discrete_entropy(tokens: List[int]) -> float:
    """Empirical Shannon entropy in bits."""
    n = len(tokens)
    if n == 0:
        return 0.0
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * np.log2(p)
    return float(h)


def compute_mutual_info(tokens: List[int], facts: List[int]) -> float:
    """Mutual information I(Token; Fact) in bits."""
    n = len(tokens)
    if n == 0:
        return 0.0
    t_counts = {}
    f_counts = {}
    tf_counts = {}
    for t, f in zip(tokens, facts):
        t_counts[t] = t_counts.get(t, 0) + 1
        f_counts[f] = f_counts.get(f, 0) + 1
        tf_counts[(t, f)] = tf_counts.get((t, f), 0) + 1

    mi = 0.0
    for (t, f), c in tf_counts.items():
        p_tf = c / n
        p_t = t_counts[t] / n
        p_f = f_counts[f] / n
        if p_tf > 0 and p_t > 0 and p_f > 0:
            mi += p_tf * np.log2(p_tf / (p_t * p_f))
    return float(max(0.0, mi))


def train_single_run(job_args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for single training run in parallel."""
    torch.set_num_threads(1)
    cond = job_args["condition"]
    seed = job_args["seed"]
    resume_ckpt = job_args["resume_checkpoint"]
    out_ckpt = job_args["output_checkpoint"]
    csv_path = job_args["history_csv"]
    json_path = job_args["history_json"]
    total_steps = job_args["steps_to_train"]
    aux_loss_coef = job_args["aux_loss_coef"]
    mode = "coordinated" if (cond in ("COMM", "COMM_AUX")) else "independent"

    # Check if already trained to 600k
    if Path(out_ckpt).exists():
        try:
            ckpt_data = torch.load(out_ckpt, map_location="cpu")
            if ckpt_data.get("total_steps", 0) >= 600000:
                print(f"[{cond} s{seed}] Already completed 600k steps, skipping training.", flush=True)
                return {
                    "condition": cond,
                    "seed": seed,
                    "output_checkpoint": out_ckpt,
                    "wall_clock_sec": 0.0,
                    "throughput_steps_sec": 0.0,
                    "steps_trained": 0,
                    "total_steps": ckpt_data.get("total_steps", 600000),
                    "final_metrics": ckpt_data.get("metrics", {}),
                }
        except Exception:
            pass

    t0 = time.time()
    print(f"[{cond} s{seed}] Starting training: +{total_steps} steps (resuming from {resume_ckpt})...", flush=True)

    res = train_stratos(
        config_path="config/training.yaml",
        total_timesteps=total_steps,
        seed=seed,
        coordination_mode=mode,
        opponent_strategy="mixed",
        resume_checkpoint=resume_ckpt,
        output_checkpoint_path=out_ckpt,
        history_csv_path=csv_path,
        history_json_path=json_path,
        include_role_id=True,
        verbose=False,
        aux_loss_coef=aux_loss_coef,
        comm_entropy_coef=0.01 if mode == "coordinated" else 0.0,
        save_interval_steps=50000,
    )
    dt = time.time() - t0
    tput = total_steps / max(0.1, dt)
    print(f"[{cond} s{seed}] Finished in {dt:.1f}s ({tput:.1f} steps/s).", flush=True)
    return {
        "condition": cond,
        "seed": seed,
        "output_checkpoint": out_ckpt,
        "wall_clock_sec": dt,
        "throughput_steps_sec": tput,
        "steps_trained": res.get("steps_trained", total_steps),
        "total_steps": res.get("total_steps", 600000),
        "final_metrics": res.get("latest_metrics", {}),
    }


def evaluate_checkpoint_with_ablations(
    checkpoint_path: str,
    coordination_mode: str = "coordinated",
    num_seeds: int = 50,
    seed_start: int = 5000,
) -> Dict[str, Any]:
    """Evaluate checkpoint on held-out seeds, including message analysis and ablations."""
    device = "cpu"
    policy, meta = load_checkpoint(checkpoint_path, device=device)
    policy.eval()
    opponents = ["aggressive", "defensive", "resource_focused"]

    # Storage
    records = []
    attack_tokens = []
    weak_lanes = []
    defense_tokens = []
    wave_flanks = []
    all_tokens = []

    # Ablation trackers: for COMM/COMM_AUX, evaluate intact, zeroed, random, shuffled
    is_comm = (coordination_mode == "coordinated") and getattr(policy, "has_comm_head", False)
    ablation_results = {"intact": [], "zeroed": [], "random": [], "shuffled": []} if is_comm else {}

    ablation_modes = ["intact", "zeroed", "random", "shuffled"] if is_comm else ["intact"]

    for abl_mode in ablation_modes:
        rng_abl = np.random.default_rng(999)
        for opp in opponents:
            for s_idx in range(num_seeds):
                seed = seed_start + s_idx
                env = StrategyGameEnv(
                    coordination_mode=coordination_mode,
                    opponent_strategy=opp,
                    include_role_id=True,
                )
                obs, _ = env.reset(seed=seed)
                term = False
                trunc = False
                ep_reward = 0.0

                while not (term or trunc):
                    obs_arr = np.array([obs[a] for a in env.possible_agents])

                    # Apply ablation to received message slots if applicable
                    # Comm slots are at indices 52:70 (two 9-dim one-hot vectors)
                    if is_comm and abl_mode != "intact":
                        if abl_mode == "zeroed":
                            # zero out message features and set token 0 active
                            obs_arr[:, 52:70] = 0.0
                            obs_arr[:, 52] = 1.0
                            obs_arr[:, 61] = 1.0
                        elif abl_mode == "random":
                            obs_arr[:, 52:70] = 0.0
                            for row_i in range(len(env.possible_agents)):
                                r1 = rng_abl.integers(0, 9)
                                r2 = rng_abl.integers(0, 9)
                                obs_arr[row_i, 52 + r1] = 1.0
                                obs_arr[row_i, 61 + r2] = 1.0
                        elif abl_mode == "shuffled":
                            # permute rows of message vectors across agents
                            comm_part = obs_arr[:, 52:70].copy()
                            shuffled_idx = rng_abl.permutation(len(env.possible_agents))
                            obs_arr[:, 52:70] = comm_part[shuffled_idx]

                    obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        if policy.has_comm_head:
                            act_t, msg_t, _, _, _, _ = policy.get_actions_and_values(obs_tensor, deterministic=True)
                            actions = {
                                a: {"action": int(act_t[i].item()), "message": int(msg_t[i].item())}
                                for i, a in enumerate(env.possible_agents)
                            }
                        else:
                            act_t, _, _, _ = policy.get_action_and_value(obs_tensor, deterministic=True)
                            actions = {a: int(act_t[i].item()) for i, a in enumerate(env.possible_agents)}

                    # Collect message tokens during intact evaluation
                    if abl_mode == "intact" and is_comm and env.state:
                        t_atk = actions["attack"]["message"]
                        t_def = actions["defense"]["message"]
                        t_cmd = actions["commander"]["message"]
                        all_tokens.extend([t_atk, t_def, t_cmd])

                        attack_tokens.append(t_atk)
                        weak_lanes.append(env.state.weak_lane)

                        if env.state.wave_warning_active:
                            defense_tokens.append(t_def)
                            wave_flanks.append(env.state.wave_flank)

                    obs, rewards, terms, truncs, info = env.step(actions)
                    ep_reward += sum(rewards.values()) / len(rewards)
                    term = any(terms.values())
                    trunc = any(truncs.values())

                winner = env.state.winner
                rec = {
                    "ablation": abl_mode,
                    "opponent": opp,
                    "seed": seed,
                    "winner": winner,
                    "is_win": float(winner == "friendly"),
                    "is_draw": float(winner is None or winner == "draw"),
                    "is_loss": float(winner == "enemy"),
                    "reward": float(ep_reward),
                }
                if abl_mode == "intact":
                    records.append(rec)
                if is_comm:
                    ablation_results[abl_mode].append(rec)

    # Compute communication statistics
    comm_stats = {}
    if is_comm and len(all_tokens) > 0:
        counts = {}
        for t in all_tokens:
            counts[t] = counts.get(t, 0) + 1
        n_tok = len(all_tokens)
        used_tokens = sum(1 for c in counts.values() if (c / n_tok) >= 0.01)
        token_entropy = compute_discrete_entropy(all_tokens)
        mi_attack_lane = compute_mutual_info(attack_tokens, weak_lanes)
        mi_defense_flank = compute_mutual_info(defense_tokens, wave_flanks) if len(defense_tokens) > 0 else 0.0

        comm_stats = {
            "tokens_used_out_of_9": int(used_tokens),
            "message_entropy_bits": float(token_entropy),
            "mi_attack_token_weak_lane_bits": float(mi_attack_lane),
            "mi_defense_token_wave_flank_bits": float(mi_defense_flank),
            "token_frequency": {int(k): round(v / n_tok, 4) for k, v in sorted(counts.items())},
        }

    # Summarize ablations
    ablation_summary = {}
    if is_comm:
        for abl, abl_recs in ablation_results.items():
            win_r = float(np.mean([r["is_win"] for r in abl_recs]) * 100.0)
            draw_r = float(np.mean([r["is_draw"] for r in abl_recs]) * 100.0)
            loss_r = float(np.mean([r["is_loss"] for r in abl_recs]) * 100.0)
            m_rew = float(np.mean([r["reward"] for r in abl_recs]))
            ablation_summary[abl] = {
                "win_rate": round(win_r, 2),
                "draw_rate": round(draw_r, 2),
                "loss_rate": round(loss_r, 2),
                "mean_reward": round(m_rew, 2),
            }

    return {
        "records": records,
        "comm_stats": comm_stats,
        "ablation_summary": ablation_summary,
    }


def evaluate_single_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for evaluation in parallel."""
    torch.set_num_threads(1)
    cond = job["condition"]
    s = job["seed"]
    ckpt_path = job["checkpoint_path"]
    mode = job["mode"]
    print(f"[Eval] Evaluating {cond} s{s} ({ckpt_path})...", flush=True)
    res = evaluate_checkpoint_with_ablations(
        ckpt_path,
        coordination_mode=mode,
        num_seeds=50,
        seed_start=5000,
    )
    return {
        "condition": cond,
        "seed": s,
        "records": res["records"],
        "comm_stats": res["comm_stats"],
        "ablation_summary": res["ablation_summary"],
    }


def main():
    seeds = [42, 101, 202]
    conditions = ["NO_COMM", "COMM", "COMM_AUX"]
    target_extra_steps = 400000

    ckpt_base = Path("artifacts/checkpoints/v3_resumed")
    exp_base = Path("artifacts/experiments/v3_resumed")
    sanity_ckpt_base = Path("artifacts/checkpoints/v3_sanity")
    ckpt_base.mkdir(parents=True, exist_ok=True)
    exp_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("STRATOS v3: RESUMED TRAINING (+400k steps) & RIGOROUS EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # 1. Prepare training jobs
    jobs = []
    for cond in conditions:
        for s in seeds:
            if cond == "NO_COMM":
                resume_ckpt = sanity_ckpt_base / f"no_comm_s{s}.pt"
                aux_coef = 0.0
            elif cond == "COMM":
                resume_ckpt = sanity_ckpt_base / f"comm_s{s}.pt"
                aux_coef = 0.0
            else:  # COMM_AUX
                resume_ckpt = sanity_ckpt_base / f"comm_s{s}.pt"
                aux_coef = 0.1

            out_ckpt = ckpt_base / f"{cond.lower()}_s{s}.pt"
            csv_path = exp_base / f"history_{cond.lower()}_s{s}.csv"
            json_path = exp_base / f"history_{cond.lower()}_s{s}.json"

            jobs.append({
                "condition": cond,
                "seed": s,
                "resume_checkpoint": str(resume_ckpt),
                "output_checkpoint": str(out_ckpt),
                "history_csv": str(csv_path),
                "history_json": str(json_path),
                "steps_to_train": target_extra_steps,
                "aux_loss_coef": aux_coef,
            })

    # 2. Run training in parallel across cores
    max_workers = min(len(jobs), os.cpu_count() or 4)
    print(f"\n[Training] Launching {len(jobs)} jobs across {max_workers} parallel workers...", flush=True)
    t_train_start = time.time()
    training_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_single_run, job): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            training_results.append(res)

    train_total_time = time.time() - t_train_start
    print(f"\n[Training Complete] All jobs finished in {train_total_time:.1f}s ({train_total_time/60.0:.2f} mins).", flush=True)

    # 3. Evaluate checkpoints on held-out seeds 5000-5049 in parallel
    print("\n" + "=" * 70)
    print("EVALUATING CHECKPOINTS ON HELD-OUT SEEDS (5000-5049, 150 eps per cell)")
    print("=" * 70)

    eval_jobs = []
    for cond in conditions:
        mode = "coordinated" if (cond in ("COMM", "COMM_AUX")) else "independent"
        for s in seeds:
            ckpt_path = ckpt_base / f"{cond.lower()}_s{s}.pt"
            eval_jobs.append({
                "condition": cond,
                "seed": s,
                "checkpoint_path": str(ckpt_path),
                "mode": mode,
            })

    eval_out = {cond: {"per_seed": {}, "comm_stats": {}, "ablations": {}} for cond in conditions}
    t_eval_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single_job, job): job for job in eval_jobs}
        for fut in as_completed(futures):
            res = fut.result()
            c = res["condition"]
            s = res["seed"]
            eval_out[c]["per_seed"][s] = res["records"]
            if res["comm_stats"]:
                eval_out[c]["comm_stats"][s] = res["comm_stats"]
            if res["ablation_summary"]:
                eval_out[c]["ablations"][s] = res["ablation_summary"]

    eval_total_time = time.time() - t_eval_start
    print(f"\n[Evaluation Complete] All evaluations finished in {eval_total_time:.1f}s ({eval_total_time/60.0:.2f} mins).")

    # 4. Compute cluster bootstrap CIs and metrics
    summary_report: Dict[str, Any] = {
        "training_wall_clock_sec": train_total_time,
        "evaluation_wall_clock_sec": eval_total_time,
        "reference_baselines": {
            "Random Policy": {"win_rate": 6.0, "draw_rate": 84.0, "loss_rate": 10.0, "mean_reward": -23.21, "reward_ci": [-36.86, -9.56]},
            "Scripted Heuristic": {"win_rate": 45.3, "draw_rate": 33.3, "loss_rate": 21.3, "mean_reward": 65.87, "reward_ci": [37.98, 93.75]},
            "Silent Reference": {"win_rate": 22.7, "draw_rate": 61.3, "loss_rate": 16.0, "mean_reward": 8.00, "reward_ci": [-15.49, 31.49]},
            "Oracle Messenger": {"win_rate": 54.7, "draw_rate": 32.7, "loss_rate": 12.7, "mean_reward": 132.23, "reward_ci": [103.60, 160.87]},
        },
        "conditions": {},
        "training_health": {},
    }

    # Extract training health from histories
    for cond in conditions:
        summary_report["training_health"][cond] = {}
        for s in seeds:
            csv_path = exp_base / f"history_{cond.lower()}_s{s}.csv"
            health = {"final_action_entropy": None, "final_comm_entropy": None, "final_value_loss": None, "failed_to_learn": False}
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    if "entropy" in df.columns:
                        health["final_action_entropy"] = float(df["entropy"].iloc[-1])
                    if "comm_entropy" in df.columns:
                        health["final_comm_entropy"] = float(df["comm_entropy"].iloc[-1])
                    # Check recent rewards
                    recent_rew = float(df["episode_reward"].iloc[-20:].mean()) if "episode_reward" in df.columns else 0.0
                    health["recent_train_reward"] = recent_rew
            # Read from checkpoint metrics if available
            ckpt_path = ckpt_base / f"{cond.lower()}_s{s}.pt"
            if ckpt_path.exists():
                try:
                    meta = torch.load(ckpt_path, map_location="cpu").get("metrics", {})
                    health["final_value_loss"] = float(meta.get("value_loss", 0.0))
                    if health["final_action_entropy"] is None:
                        health["final_action_entropy"] = float(meta.get("entropy", 0.0))
                    if health["final_comm_entropy"] is None and "comm_entropy" in meta:
                        health["final_comm_entropy"] = float(meta.get("comm_entropy", 0.0))
                except Exception:
                    pass
            summary_report["training_health"][cond][s] = health

    for cond in conditions:
        win_rates = {s: [r["is_win"] * 100.0 for r in recs] for s, recs in eval_out[cond]["per_seed"].items()}
        draw_rates = {s: [r["is_draw"] * 100.0 for r in recs] for s, recs in eval_out[cond]["per_seed"].items()}
        loss_rates = {s: [r["is_loss"] * 100.0 for r in recs] for s, recs in eval_out[cond]["per_seed"].items()}
        rewards = {s: [r["reward"] for r in recs] for s, recs in eval_out[cond]["per_seed"].items()}

        mean_win, win_ci = cluster_bootstrap_ci(win_rates)
        mean_draw, draw_ci = cluster_bootstrap_ci(draw_rates)
        mean_loss, loss_ci = cluster_bootstrap_ci(loss_rates)
        mean_rew, rew_ci = cluster_bootstrap_ci(rewards)

        per_seed_summary = {}
        for s in seeds:
            per_seed_summary[s] = {
                "win_rate": round(float(np.mean(win_rates[s])), 2),
                "draw_rate": round(float(np.mean(draw_rates[s])), 2),
                "loss_rate": round(float(np.mean(loss_rates[s])), 2),
                "mean_reward": round(float(np.mean(rewards[s])), 2),
            }

        summary_report["conditions"][cond] = {
            "cluster_mean": {
                "win_rate": round(mean_win, 2),
                "win_ci": [round(win_ci[0], 2), round(win_ci[1], 2)],
                "draw_rate": round(mean_draw, 2),
                "draw_ci": [round(draw_ci[0], 2), round(draw_ci[1], 2)],
                "loss_rate": round(mean_loss, 2),
                "loss_ci": [round(loss_ci[0], 2), round(loss_ci[1], 2)],
                "mean_reward": round(mean_rew, 2),
                "reward_ci": [round(rew_ci[0], 2), round(rew_ci[1], 2)],
            },
            "per_seed": per_seed_summary,
            "comm_stats": eval_out[cond]["comm_stats"],
            "ablations": eval_out[cond]["ablations"],
        }

    # Save full summary report
    rep_path = exp_base / "resumed_train_report.json"
    with open(rep_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n[Report Saved] Saved comprehensive summary to {rep_path}")


if __name__ == "__main__":
    main()
