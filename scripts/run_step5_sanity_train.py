"""STEP 5: Sanity Training (Small) - 200k steps for COMM and NO_COMM across 3 seeds.

Evaluates on held-out seeds (5000-5049) across 3 opponent archetypes.
Computes win/draw/loss, rewards, entropy, and cluster bootstrap 95% CI.
"""

import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from training.train import train_stratos
from environment.strategy_game import StrategyGameEnv
from agents.policies import ActorCriticPolicy
from training.checkpoint import load_checkpoint
from agents.reference_policies import OracleMessengerPolicy, SilentReferencePolicy
from agents.targeted_policy import TargetedCombatPolicy


def cluster_bootstrap_ci(per_seed_values, n_bootstraps=2000, ci=0.95):
    """Cluster bootstrap over training seeds."""
    seeds = list(per_seed_values.keys())
    k = len(seeds)
    if k <= 1:
        mean_val = float(np.mean([np.mean(v) for v in per_seed_values.values()]))
        return mean_val, (mean_val, mean_val)

    # Each cluster is the set of episode outcomes for one training seed
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


def evaluate_checkpoint_held_out(checkpoint_path, coordination_mode="coordinated", num_seeds=50, seed_start=5000):
    device = "cpu"
    policy, meta = load_checkpoint(checkpoint_path, device=device)
    opponents = ["aggressive", "defensive", "resource_focused"]

    results = []
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
                obs_tensor = torch.as_tensor(
                    np.array([obs[a] for a in env.possible_agents]),
                    dtype=torch.float32,
                    device=device,
                )
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

                obs, rewards, terms, truncs, info = env.step(actions)
                ep_reward += sum(rewards.values()) / len(rewards)
                term = any(terms.values())
                trunc = any(truncs.values())

            winner = env.state.winner
            results.append({
                "opponent": opp,
                "seed": seed,
                "winner": winner,
                "is_win": float(winner == "friendly"),
                "is_draw": float(winner is None or winner == "draw"),
                "is_loss": float(winner == "enemy"),
                "reward": float(ep_reward),
            })
    return results


def evaluate_heuristic_baseline(policy_fn, num_seeds=50, seed_start=5000):
    opponents = ["aggressive", "defensive", "resource_focused"]
    results = []
    for opp in opponents:
        for s_idx in range(num_seeds):
            seed = seed_start + s_idx
            env = StrategyGameEnv(opponent_strategy=opp, include_role_id=True)
            obs, _ = env.reset(seed=seed)
            if hasattr(policy_fn, "reset"):
                policy_fn.reset()
            term = False
            trunc = False
            ep_reward = 0.0

            while not (term or trunc):
                actions = {}
                for a in env.possible_agents:
                    res = policy_fn.select_action_and_message(a, obs[a]) if hasattr(policy_fn, "select_action_and_message") else (policy_fn.select_action(a, obs[a]), 0)
                    actions[a] = {"action": int(res[0]), "message": int(res[1])}
                obs, rewards, terms, truncs, info = env.step(actions)
                ep_reward += sum(rewards.values()) / len(rewards)
                term = any(terms.values())
                trunc = any(truncs.values())

            winner = env.state.winner
            results.append({
                "opponent": opp,
                "seed": seed,
                "winner": winner,
                "is_win": float(winner == "friendly"),
                "is_draw": float(winner is None or winner == "draw"),
                "is_loss": float(winner == "enemy"),
                "reward": float(ep_reward),
            })
    return results


def evaluate_random_baseline(num_seeds=50, seed_start=5000):
    opponents = ["aggressive", "defensive", "resource_focused"]
    results = []
    rng = np.random.default_rng(12345)
    for opp in opponents:
        for s_idx in range(num_seeds):
            seed = seed_start + s_idx
            env = StrategyGameEnv(opponent_strategy=opp, include_role_id=True)
            obs, _ = env.reset(seed=seed)
            term = False
            trunc = False
            ep_reward = 0.0

            while not (term or trunc):
                actions = {
                    a: {"action": int(rng.integers(0, 9)), "message": int(rng.integers(0, 9))}
                    for a in env.possible_agents
                }
                obs, rewards, terms, truncs, info = env.step(actions)
                ep_reward += sum(rewards.values()) / len(rewards)
                term = any(terms.values())
                trunc = any(truncs.values())

            winner = env.state.winner
            results.append({
                "opponent": opp,
                "seed": seed,
                "winner": winner,
                "is_win": float(winner == "friendly"),
                "is_draw": float(winner is None or winner == "draw"),
                "is_loss": float(winner == "enemy"),
                "reward": float(ep_reward),
            })
    return results


def main():
    total_timesteps = 200000
    seeds = [42, 101, 202]
    conditions = ["COMM", "NO_COMM"]

    ckpt_base = Path("artifacts/checkpoints/v3_sanity")
    exp_base = Path("artifacts/experiments/v3_sanity")
    ckpt_base.mkdir(parents=True, exist_ok=True)
    exp_base.mkdir(parents=True, exist_ok=True)

    training_logs = {}

    print(f"=== STRATOS STEP 5: SANITY TRAINING (6 runs x {total_timesteps} steps) ===")

    for cond in conditions:
        is_comm = (cond == "COMM")
        mode = "coordinated" if is_comm else "independent"
        training_logs[cond] = {}

        for s in seeds:
            ckpt_path = ckpt_base / f"{cond.lower()}_s{s}.pt"
            csv_path = exp_base / f"history_{cond.lower()}_s{s}.csv"
            json_path = exp_base / f"history_{cond.lower()}_s{s}.json"

            already_trained = False
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    if ckpt.get("total_steps", 0) >= total_timesteps:
                        already_trained = True
                except Exception:
                    pass

            t0 = time.time()
            if already_trained:
                print(f"\n[Training] {cond} | Seed {s} already completed ({total_timesteps} steps), using saved checkpoint.")
                dt = 0.0
                throughput = 0.0
                mean_ep_rew = 0.0
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    mean_ep_rew = float(df["episode_reward"].iloc[-10:].mean()) if "episode_reward" in df.columns else 0.0
            else:
                print(f"\n[Training] {cond} | Seed {s} | Target: {total_timesteps} steps...")
                res = train_stratos(
                    config_path="config/training.yaml",
                    total_timesteps=total_timesteps,
                    seed=s,
                    coordination_mode=mode,
                    opponent_strategy="mixed",
                    output_checkpoint_path=str(ckpt_path),
                    history_csv_path=str(csv_path),
                    history_json_path=str(json_path),
                    include_role_id=True,
                    verbose=False,
                )
                dt = time.time() - t0
                throughput = total_timesteps / max(0.1, dt)
                mean_ep_rew = float(np.mean([r["episode_reward"] for r in res["history_records"][-10:]])) if res["history_records"] else 0.0
                print(f"  Finished in {dt:.1f}s ({throughput:.0f} steps/s) | Recent Ep Reward: {mean_ep_rew:.2f}")

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                training_logs[cond][s] = {
                    "final_mean_reward": float(mean_ep_rew),
                    "wall_clock_sec": float(dt),
                    "throughput_steps_sec": float(throughput),
                    "final_policy_entropy": float(df["entropy"].iloc[-1]) if "entropy" in df.columns else None,
                    "final_comm_entropy": float(df["comm_entropy"].iloc[-1]) if "comm_entropy" in df.columns else None,
                }

    print("\n=== STEP 5: EVALUATING TRAINED CHECKPOINTS ON HELD-OUT SEEDS (5000-5049) ===")
    eval_results = {"COMM": {}, "NO_COMM": {}}

    for cond in conditions:
        mode = "coordinated" if (cond == "COMM") else "independent"
        for s in seeds:
            ckpt_path = ckpt_base / f"{cond.lower()}_s{s}.pt"
            print(f"Evaluating {cond} Seed {s} on 150 held-out episodes...")
            records = evaluate_checkpoint_held_out(str(ckpt_path), coordination_mode=mode, num_seeds=50, seed_start=5000)
            eval_results[cond][s] = records

    print("\n=== EVALUATING REFERENCE BASELINES ON HELD-OUT SEEDS ===")
    print("Evaluating Random Policy...")
    random_records = evaluate_random_baseline(num_seeds=50, seed_start=5000)

    print("Evaluating Scripted Heuristic...")
    scripted_records = evaluate_heuristic_baseline(TargetedCombatPolicy(), num_seeds=50, seed_start=5000)

    print("Evaluating Oracle Messenger Policy...")
    oracle_records = evaluate_heuristic_baseline(OracleMessengerPolicy(), num_seeds=50, seed_start=5000)

    print("Evaluating Silent Reference Policy...")
    silent_records = evaluate_heuristic_baseline(SilentReferencePolicy(), num_seeds=50, seed_start=5000)

    # Compute Aggregate Metrics & Cluster Bootstrap CIs
    summary_report = {
        "training_logs": training_logs,
        "evaluated_conditions": {},
        "baselines": {},
    }

    print("\n" + "="*80)
    print("STEP 5 SANITY TRAINING RESULTS (Held-Out Seeds 5000-5049, 150 eps per cell)")
    print("="*80)

    # Summary table header
    print(f"{'Condition':<18} | {'Win %':<10} | {'Draw %':<10} | {'Loss %':<10} | {'Mean Reward (95% CI)':<25}")
    print("-" * 80)

    for cond in conditions:
        wins_by_seed = {s: [r["is_win"] * 100.0 for r in eval_results[cond][s]] for s in seeds}
        draws_by_seed = {s: [r["is_draw"] * 100.0 for r in eval_results[cond][s]] for s in seeds}
        losses_by_seed = {s: [r["is_loss"] * 100.0 for r in eval_results[cond][s]] for s in seeds}
        rewards_by_seed = {s: [r["reward"] for r in eval_results[cond][s]] for s in seeds}

        m_win, ci_win = cluster_bootstrap_ci(wins_by_seed)
        m_draw, ci_draw = cluster_bootstrap_ci(draws_by_seed)
        m_loss, ci_loss = cluster_bootstrap_ci(losses_by_seed)
        m_rew, ci_rew = cluster_bootstrap_ci(rewards_by_seed)

        per_seed_summary = {}
        for s in seeds:
            per_seed_summary[s] = {
                "win_rate": float(np.mean(wins_by_seed[s])),
                "draw_rate": float(np.mean(draws_by_seed[s])),
                "loss_rate": float(np.mean(losses_by_seed[s])),
                "mean_reward": float(np.mean(rewards_by_seed[s])),
            }

        summary_report["evaluated_conditions"][cond] = {
            "per_seed": per_seed_summary,
            "mean_win": m_win, "ci_win": list(ci_win),
            "mean_draw": m_draw, "ci_draw": list(ci_draw),
            "mean_loss": m_loss, "ci_loss": list(ci_loss),
            "mean_reward": m_rew, "ci_reward": list(ci_rew),
        }

        ci_str = f"[{ci_rew[0]:.1f}, {ci_rew[1]:.1f}]"
        print(f"{cond:<18} | {m_win:>5.1f}%     | {m_draw:>5.1f}%     | {m_loss:>5.1f}%     | {m_rew:>7.2f} {ci_str:<16}")

    # Baseline rows
    for name, recs in [
        ("Random", random_records),
        ("Scripted Heuristic", scripted_records),
        ("Silent Reference", silent_records),
        ("Oracle Messenger", oracle_records),
    ]:
        w = np.mean([r["is_win"] for r in recs]) * 100.0
        d = np.mean([r["is_draw"] for r in recs]) * 100.0
        l = np.mean([r["is_loss"] for r in recs]) * 100.0
        rews = [r["reward"] for r in recs]
        m_r = np.mean(rews)
        ci_r = 1.96 * (np.std(rews) / np.sqrt(len(rews)))
        summary_report["baselines"][name] = {
            "win_rate": float(w), "draw_rate": float(d), "loss_rate": float(l),
            "mean_reward": float(m_r), "ci_reward": [float(m_r - ci_r), float(m_r + ci_r)],
        }
        ci_str = f"[{m_r - ci_r:.1f}, {m_r + ci_r:.1f}]"
        print(f"{name:<18} | {w:>5.1f}%     | {d:>5.1f}%     | {l:>5.1f}%     | {m_r:>7.2f} {ci_str:<16}")

    print("="*80)

    # Save summary report
    out_file = exp_base / "sanity_train_report.json"
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved sanity report to: {out_file}")


if __name__ == "__main__":
    main()
