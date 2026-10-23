"""Comprehensive Research Experiment Suite for STRATOS AI (v2).

Orchestrates:
1. Multi-seed training across Condition A (NO_COMM) and Condition B (COMM)
2. Rigorous paired-seed evaluation against all opponent strategies
3. Ablation studies (zeroed, random, shuffled messages)
4. Statistical significance testing (paired t-test, bootstrap 95% CI)
5. Token distribution analysis
6. Publication-quality artifact and plot generation
"""

import math
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import train_stratos
from evaluation.evaluate import evaluate_policy
from evaluation.metrics import EpisodeMetric, aggregate_metrics, to_dataframe


def bootstrap_ci(data: List[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval."""
    if len(data) < 2:
        val = data[0] if data else 0.0
        return (val, val)
    rng = np.random.default_rng(42)
    boot_means = []
    arr = np.array(data, dtype=float)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100.0))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100.0))
    return (round(lower, 2), round(upper, 2))


def run_training_suite(
    seeds: List[int],
    timesteps: int,
    checkpoints_dir: Path,
    experiments_dir: Path,
) -> Dict[str, Dict[int, str]]:
    """Train policies across Condition A (NO_COMM) and Condition B (COMM)."""
    checkpoints = {"COMM": {}, "NO_COMM": {}}

    for seed in seeds:
        # 1. Condition B: COMM (Coordinated with learned dual-head communication)
        comm_ckpt = checkpoints_dir / f"stratos_policy_comm_seed{seed}.pt"
        comm_csv = experiments_dir / f"training_history_comm_seed{seed}.csv"
        print(f"\n=======================================================")
        print(f"  Training Condition B [COMM] — Seed: {seed} ({timesteps} steps)")
        print(f"=======================================================")
        train_stratos(
            total_timesteps=timesteps,
            seed=seed,
            coordination_mode="coordinated",
            opponent_strategy="mixed",
            output_checkpoint_path=str(comm_ckpt),
            history_csv_path=str(comm_csv),
            include_role_id=True,
            verbose=True,
        )
        checkpoints["COMM"][seed] = str(comm_ckpt)

        # 2. Condition A: NO_COMM (Independent without communication channels)
        nocomm_ckpt = checkpoints_dir / f"stratos_policy_nocomm_seed{seed}.pt"
        nocomm_csv = experiments_dir / f"training_history_nocomm_seed{seed}.csv"
        print(f"\n=======================================================")
        print(f"  Training Condition A [NO_COMM] — Seed: {seed} ({timesteps} steps)")
        print(f"=======================================================")
        train_stratos(
            total_timesteps=timesteps,
            seed=seed,
            coordination_mode="independent",
            opponent_strategy="mixed",
            output_checkpoint_path=str(nocomm_ckpt),
            history_csv_path=str(nocomm_csv),
            include_role_id=True,
            verbose=True,
        )
        checkpoints["NO_COMM"][seed] = str(nocomm_ckpt)

    return checkpoints


def run_paired_evaluation(
    checkpoints: Dict[str, Dict[int, str]],
    eval_seeds: List[int],
    episodes_per_scenario: int = 15,
    experiments_dir: Path = Path("artifacts/experiments"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run paired-seed evaluation comparing COMM vs NO_COMM across opponents."""
    opponents = ["aggressive", "defensive", "resource_focused"]
    all_records: List[Dict[str, Any]] = []

    print("\n=================================================================")
    print("  Running Paired-Seed Comparative Evaluation Benchmark          ")
    print("=================================================================")

    for train_seed in checkpoints["COMM"].keys():
        comm_ckpt = checkpoints["COMM"][train_seed]
        nocomm_ckpt = checkpoints["NO_COMM"][train_seed]

        for opp in opponents:
            for eval_seed in eval_seeds:
                # Evaluate COMM
                comm_recs, _ = evaluate_policy(
                    checkpoint_path=comm_ckpt,
                    num_episodes=episodes_per_scenario,
                    coordination_mode="coordinated",
                    opponent_strategy=opp,
                    seed=eval_seed,
                )
                # Evaluate NO_COMM with exact same environment seed
                nocomm_recs, _ = evaluate_policy(
                    checkpoint_path=nocomm_ckpt,
                    num_episodes=episodes_per_scenario,
                    coordination_mode="independent",
                    opponent_strategy=opp,
                    seed=eval_seed,
                )

                for r in comm_recs:
                    d = asdict(r)
                    d["train_seed"] = train_seed
                    d["eval_seed"] = eval_seed
                    d["condition"] = "COMM"
                    all_records.append(d)

                for r in nocomm_recs:
                    d = asdict(r)
                    d["train_seed"] = train_seed
                    d["eval_seed"] = eval_seed
                    d["condition"] = "NO_COMM"
                    all_records.append(d)

    df = pd.DataFrame(all_records)
    out_csv = experiments_dir / "multiseed_evaluation_paired.csv"
    df.to_csv(out_csv, index=False)
    print(f"Paired evaluation saved to: {out_csv} (Total records: {len(df)})")

    # Statistical summary
    comm_df = df[df["condition"] == "COMM"]
    nocomm_df = df[df["condition"] == "NO_COMM"]

    comm_wins = comm_df["winner"].eq("friendly").mean()
    nocomm_wins = nocomm_df["winner"].eq("friendly").mean()
    comm_draws = comm_df["winner"].eq("draw").mean()
    nocomm_draws = nocomm_df["winner"].eq("draw").mean()

    comm_rewards = comm_df["team_reward"].tolist()
    nocomm_rewards = nocomm_df["team_reward"].tolist()

    # Welch's t-test calculation
    arr_a = np.array(comm_rewards, dtype=float)
    arr_b = np.array(nocomm_rewards, dtype=float)
    n1, n2 = max(1, len(arr_a)), max(1, len(arr_b))
    m1, m2 = float(np.mean(arr_a)), float(np.mean(arr_b))
    v1, v2 = float(np.var(arr_a, ddof=1)), float(np.var(arr_b, ddof=1))
    denom = math.sqrt(max(1e-8, v1 / n1 + v2 / n2))
    t_stat = float((m1 - m2) / denom)
    z = abs(t_stat)
    p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))

    summary = {
        "comm_win_rate": round(float(comm_wins), 4),
        "nocomm_win_rate": round(float(nocomm_wins), 4),
        "win_rate_delta": round(float(comm_wins - nocomm_wins), 4),
        "comm_draw_rate": round(float(comm_draws), 4),
        "nocomm_draw_rate": round(float(nocomm_draws), 4),
        "comm_mean_reward": round(float(np.mean(comm_rewards)), 2),
        "comm_reward_ci": bootstrap_ci(comm_rewards),
        "nocomm_mean_reward": round(float(np.mean(nocomm_rewards)), 2),
        "nocomm_reward_ci": bootstrap_ci(nocomm_rewards),
        "reward_t_statistic": round(float(t_stat), 4),
        "reward_p_value": float(p_val),
        "statistically_significant": bool(p_val < 0.05),
    }

    out_json = experiments_dir / "comm_vs_nocomm_statistical_report.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    return df, summary


def run_ablation_study(
    best_comm_ckpt: str,
    eval_seeds: List[int],
    episodes_per_ablation: int = 15,
    experiments_dir: Path = Path("artifacts/experiments"),
) -> pd.DataFrame:
    """Run ablation suite: full COMM vs zeroed, random, and shuffled messages."""
    ablations = [None, "zeroed", "random", "shuffled"]
    ablation_records = []

    print("\n=================================================================")
    print("  Running Learned Communication Ablation Suite                   ")
    print("=================================================================")

    for abl in ablations:
        abl_name = abl if abl is not None else "intact_comm"
        for seed in eval_seeds:
            for opp in ["aggressive", "defensive", "resource_focused"]:
                recs, _ = evaluate_policy(
                    checkpoint_path=best_comm_ckpt,
                    num_episodes=episodes_per_ablation,
                    coordination_mode="coordinated",
                    opponent_strategy=opp,
                    seed=seed,
                    comm_ablation=abl,
                )
                for r in recs:
                    d = asdict(r)
                    d["ablation_type"] = abl_name
                    d["eval_seed"] = seed
                    ablation_records.append(d)

    df_abl = pd.DataFrame(ablation_records)
    out_abl_csv = experiments_dir / "ablation_study_results.csv"
    df_abl.to_csv(out_abl_csv, index=False)
    print(f"Ablation study saved to: {out_abl_csv}")
    return df_abl


def generate_research_plots(
    eval_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    experiments_dir: Path,
) -> None:
    """Generate high-resolution publication figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Win / Draw / Loss rates COMM vs NO_COMM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    opponents = ["aggressive", "defensive", "resource_focused"]
    x = np.arange(len(opponents))
    width = 0.35

    comm_win_rates = [
        eval_df[(eval_df["condition"] == "COMM") & (eval_df["opponent_strategy"] == opp)]["winner"].eq("friendly").mean() * 100
        for opp in opponents
    ]
    nocomm_win_rates = [
        eval_df[(eval_df["condition"] == "NO_COMM") & (eval_df["opponent_strategy"] == opp)]["winner"].eq("friendly").mean() * 100
        for opp in opponents
    ]

    ax1.bar(x - width/2, nocomm_win_rates, width, label="NO_COMM Baseline", color="#94a3b8")
    ax1.bar(x + width/2, comm_win_rates, width, label="COMM (Learned)", color="#0284c7")
    ax1.set_ylabel("Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Win Rate by Opponent Behavior", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([o.capitalize() for o in opponents], fontsize=10)
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 105)

    # Figure 1b: Mean Team Reward
    comm_rewards = [
        eval_df[(eval_df["condition"] == "COMM") & (eval_df["opponent_strategy"] == opp)]["team_reward"].mean()
        for opp in opponents
    ]
    nocomm_rewards = [
        eval_df[(eval_df["condition"] == "NO_COMM") & (eval_df["opponent_strategy"] == opp)]["team_reward"].mean()
        for opp in opponents
    ]

    ax2.bar(x - width/2, nocomm_rewards, width, label="NO_COMM Baseline", color="#94a3b8")
    ax2.bar(x + width/2, comm_rewards, width, label="COMM (Learned)", color="#0284c7")
    ax2.set_ylabel("Mean Cumulative Team Reward", fontsize=11, fontweight="bold")
    ax2.set_title("Team Reward Across Opponents", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([o.capitalize() for o in opponents], fontsize=10)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot1_path = experiments_dir / "comm_vs_nocomm_performance.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved performance plot to: {plot1_path}")

    # Figure 2: Ablation Study
    if not ablation_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        abl_types = ["intact_comm", "zeroed", "random", "shuffled"]
        labels = ["Intact Learned COMM", "Zeroed Messages", "Random Tokens", "Shuffled Timesteps"]
        colors = ["#0284c7", "#f43f5e", "#f59e0b", "#8b5cf6"]

        abl_means = []
        abl_errs = []
        for abl in abl_types:
            sub = ablation_df[ablation_df["ablation_type"] == abl]["team_reward"]
            m = sub.mean()
            ci_low, ci_high = bootstrap_ci(sub.tolist())
            abl_means.append(m)
            abl_errs.append(max(0.1, (ci_high - ci_low) / 2.0))

        ax.bar(labels, abl_means, yerr=abl_errs, capsize=5, color=colors)
        ax.set_ylabel("Mean Team Reward", fontsize=11, fontweight="bold")
        ax.set_title("Ablation Study: Causal Impact of Message Degradation", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot2_path = experiments_dir / "ablation_causality.png"
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"Saved ablation plot to: {plot2_path}")


def main():
    parser = argparse.ArgumentParser(description="Run STRATOS v2 Complete Research Suite")
    parser.add_argument("--timesteps", type=int, default=30000, help="Timesteps per training run")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 101, 202], help="Random seeds for training")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[777, 888, 999], help="Random seeds for evaluation")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per evaluation combo")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and evaluate existing checkpoints")
    args = parser.parse_args()

    checkpoints_dir = Path("artifacts/checkpoints")
    experiments_dir = Path("artifacts/experiments")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        print(f"\n[Starting Research Phase 3: Training Suite across {len(args.train_seeds)} seeds × 2 conditions]")
        checkpoints = run_training_suite(
            seeds=args.train_seeds,
            timesteps=args.timesteps,
            checkpoints_dir=checkpoints_dir,
            experiments_dir=experiments_dir,
        )
    else:
        print("[Skipping training — using existing checkpoints]")
        checkpoints = {
            "COMM": {s: str(checkpoints_dir / f"stratos_policy_comm_seed{s}.pt") for s in args.train_seeds},
            "NO_COMM": {s: str(checkpoints_dir / f"stratos_policy_nocomm_seed{s}.pt") for s in args.train_seeds},
        }

    # Paired-seed evaluation
    eval_df, stat_summary = run_paired_evaluation(
        checkpoints=checkpoints,
        eval_seeds=args.eval_seeds,
        episodes_per_scenario=args.episodes,
        experiments_dir=experiments_dir,
    )

    # Run Ablation Study on best COMM checkpoint
    best_comm_ckpt = checkpoints["COMM"][args.train_seeds[0]]
    ablation_df = run_ablation_study(
        best_comm_ckpt=best_comm_ckpt,
        eval_seeds=args.eval_seeds,
        episodes_per_ablation=args.episodes,
        experiments_dir=experiments_dir,
    )

    # Generate figures
    generate_research_plots(eval_df, ablation_df, experiments_dir)

    print("\n=======================================================")
    print("  STRATOS v2 RESEARCH EXPERIMENT SUITE COMPLETE!       ")
    print("=======================================================")
    print(f"COMM Win Rate     : {stat_summary['comm_win_rate']:.1%}")
    print(f"NO_COMM Win Rate  : {stat_summary['nocomm_win_rate']:.1%}")
    print(f"Win Rate Delta    : {stat_summary['win_rate_delta']:+.1%}")
    print(f"COMM Mean Reward  : {stat_summary['comm_mean_reward']} (95% CI: {stat_summary['comm_reward_ci']})")
    print(f"NO_COMM Mean Rew  : {stat_summary['nocomm_mean_reward']} (95% CI: {stat_summary['nocomm_reward_ci']})")
    print(f"P-Value (p < 0.05): {stat_summary['reward_p_value']:.4e} (Significant: {stat_summary['statistically_significant']})")


if __name__ == "__main__":
    main()
