"""CLI script to run rigorous multi-seed evaluation across all opponents and modes."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluate import evaluate_policy
from evaluation.metrics import EpisodeMetric, aggregate_metrics


def run_multiseed_evaluation(
    checkpoint_path: str = "artifacts/checkpoints/stratos_policy_candidate_fix_5000step.pt",
    seeds: List[int] = None,
    episodes_per_seed: int = 5,
    output_csv: str = "artifacts/experiments/multiseed_candidate_evaluation_results.csv",
    output_json: str = "artifacts/experiments/multiseed_candidate_evaluation_summary.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = [42, 101, 202, 303, 404]

    opponents = ["aggressive", "defensive", "resource_focused"]
    modes = ["independent", "coordinated"]

    all_records: List[Dict[str, Any]] = []
    seed_summaries: Dict[str, Any] = {}
    scenario_summaries: Dict[str, Any] = {}

    total_target = len(seeds) * len(opponents) * len(modes) * episodes_per_seed
    if verbose:
        print("\n=================================================================")
        print("  STRATOS AI — Multi-Seed Independent Evaluation Suite          ")
        print("=================================================================")
        print(f"Checkpoint         : {checkpoint_path}")
        print(f"Seeds ({len(seeds)})           : {seeds}")
        print(f"Episodes per Combo : {episodes_per_seed}")
        print(f"Total Target Runs  : {total_target} episodes (5 seeds × 3 opps × 2 modes × {episodes_per_seed} eps)\n")

    global_ep_id = 0
    for seed in seeds:
        seed_key = f"seed_{seed}"
        seed_records = []

        for opp in opponents:
            for mode in modes:
                sc_name = f"{mode}_vs_{opp}"
                records, summary = evaluate_policy(
                    checkpoint_path=checkpoint_path,
                    num_episodes=episodes_per_seed,
                    coordination_mode=mode,
                    opponent_strategy=opp,
                    seed=seed,
                )

                for r in records:
                    global_ep_id += 1
                    r_dict = asdict(r)
                    r_dict["global_episode_id"] = global_ep_id
                    r_dict["evaluation_seed"] = seed
                    all_records.append(r_dict)
                    seed_records.append(r)

                if sc_name not in scenario_summaries:
                    scenario_summaries[sc_name] = []
                scenario_summaries[sc_name].extend(records)

        seed_summaries[seed_key] = aggregate_metrics(seed_records)

    # Save to CSV
    df = pd.DataFrame(all_records)
    out_csv_path = Path(output_csv)
    out_json_path = Path(output_json)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    # Aggregate by scenario across all seeds
    aggregated_scenarios = {}
    for sc_name, recs in scenario_summaries.items():
        aggregated_scenarios[sc_name] = aggregate_metrics(recs)

    # Aggregate by mode
    coord_recs = [r for sc, recs in scenario_summaries.items() if "coordinated" in sc for r in recs]
    indep_recs = [r for sc, recs in scenario_summaries.items() if "independent" in sc for r in recs]
    coord_agg = aggregate_metrics(coord_recs)
    indep_agg = aggregate_metrics(indep_recs)

    # Per-opponent aggregated comparison
    per_opponent = {}
    for opp in opponents:
        c_opp_recs = [r for r in coord_recs if r.opponent_strategy == opp]
        i_opp_recs = [r for r in indep_recs if r.opponent_strategy == opp]
        c_summary = aggregate_metrics(c_opp_recs)
        i_summary = aggregate_metrics(i_opp_recs)
        per_opponent[opp] = {
            "coordinated": c_summary,
            "independent": i_summary,
            "win_rate_delta": round(c_summary["win_rate"] - i_summary["win_rate"], 4),
            "reward_delta": round(c_summary["avg_team_reward"] - i_summary["avg_team_reward"], 2),
        }

    full_summary = {
        "checkpoint": checkpoint_path,
        "seeds_evaluated": seeds,
        "total_episodes": len(all_records),
        "overall_coordinated": coord_agg,
        "overall_independent": indep_agg,
        "overall_win_rate_delta": round(coord_agg["win_rate"] - indep_agg["win_rate"], 4),
        "overall_reward_delta": round(coord_agg["avg_team_reward"] - indep_agg["avg_team_reward"], 2),
        "per_opponent_comparison": per_opponent,
        "scenario_summaries": aggregated_scenarios,
        "seed_summaries": seed_summaries,
    }

    with open(out_json_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    if verbose:
        print("=================================================================")
        print("  MULTI-SEED EVALUATION RESULTS MATRIX (150 EPISODES)           ")
        print("=================================================================")
        print(f"{'Scenario / Opponent':<28} | {'Mode':<6} | {'Win Rate':<10} | {'Avg Reward':<11} | {'Survival':<9} | {'Duration':<8} | {'Messages'}")
        print("-" * 88)
        for opp in opponents:
            for mode in ["independent", "coordinated"]:
                sc_name = f"{mode}_vs_{opp}"
                s = aggregated_scenarios[sc_name]
                print(
                    f"{opp.capitalize():<28} | {mode[:5]:<6} | "
                    f"{s['wins']}/{s['total_episodes']} ({s['win_rate']:.1%}) | "
                    f"{s['avg_team_reward']:>+10.2f} | "
                    f"{s['friendly_survival_rate']:>8.1%} | "
                    f"{s['avg_duration']:>6.1f}s | "
                    f"{s['avg_messages_exchanged']:>6.1f}"
                )

        print("\n=================================================================")
        print("  OVERALL COORDINATED VS INDEPENDENT SUMMARY                    ")
        print("=================================================================")
        print(f"Coordinated Total Win Rate   : {coord_agg['wins']}/{coord_agg['total_episodes']} ({coord_agg['win_rate']:.1%})")
        print(f"Independent Total Win Rate   : {indep_agg['wins']}/{indep_agg['total_episodes']} ({indep_agg['win_rate']:.1%})")
        print(f"Net Communication Win Delta  : {full_summary['overall_win_rate_delta']:+.1%}")
        print(f"Net Team Reward Delta        : {full_summary['overall_reward_delta']:+.2f} points")
        print(f"Mean Messages Exchanged      : {coord_agg['avg_messages_exchanged']:.1f} (Coord) vs {indep_agg['avg_messages_exchanged']:.1f} (Indep)")

        print("\n=================================================================")
        print("  VARIATION ACROSS SEEDS (30 EPISODES PER SEED)                  ")
        print("=================================================================")
        for s_key, s_data in seed_summaries.items():
            print(f"{s_key:<10}: Win Rate = {s_data['wins']}/{s_data['total_episodes']} ({s_data['win_rate']:.1%}) | Avg Reward = {s_data['avg_team_reward']:>+8.2f}")

        print(f"\nArtifacts saved successfully:")
        print(f"CSV  : {out_csv_path}")
        print(f"JSON : {out_json_path}\n")

    return full_summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Evaluation for STRATOS AI Policy")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/stratos_policy_candidate_fix_5000step.pt", help="Path to checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404], help="Random seeds to evaluate")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per combo per seed")
    parser.add_argument("--output-csv", type=str, default="artifacts/experiments/multiseed_candidate_evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--output-json", type=str, default="artifacts/experiments/multiseed_candidate_evaluation_summary.json", help="Output JSON path")
    args = parser.parse_args()

    run_multiseed_evaluation(
        checkpoint_path=args.checkpoint,
        seeds=args.seeds,
        episodes_per_seed=args.episodes,
        output_csv=args.output_csv,
        output_json=args.output_json,
        verbose=True,
    )


if __name__ == "__main__":
    main()
