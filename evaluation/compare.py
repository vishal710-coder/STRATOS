"""Comparative benchmarking experiment comparing Independent vs Coordinated Multi-Agent System."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml

from evaluation.evaluate import evaluate_policy
from evaluation.metrics import EpisodeMetric, aggregate_metrics, to_dataframe


def run_comparative_experiments(
    checkpoint_path: Optional[str] = "artifacts/checkpoints/stratos_policy_latest.pt",
    config_path: str = "config/evaluation.yaml",
    episodes_per_scenario: Optional[int] = None,
    seed: int = 12345,
    verbose: bool = True,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full comparative benchmark across coordination modes and opponent strategies.
    Answers Core Research Question:
    'Does communication between specialized AI agents improve their coordination and performance
    against different opponent strategies compared with agents that operate independently?'
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if episodes_per_scenario is None:
        episodes_per_scenario = int(cfg["evaluation"]["episodes_per_scenario"])

    scenarios = cfg["scenarios"]
    all_episode_records: List[EpisodeMetric] = []
    scenario_summaries: Dict[str, Any] = {}

    if verbose:
        print(f"\n=======================================================")
        print(f"  STRATOS AI - Multi-Agent Research Experiment Suite   ")
        print(f"=======================================================")
        print(f"Episodes per Scenario: {episodes_per_scenario} | Seed: {seed}")
        print(f"Checkpoint: {checkpoint_path}\n")

    for sc in scenarios:
        sc_name = sc["name"]
        coord_mode = sc["coordination_mode"]
        opp_strat = sc["opponent_strategy"]

        if verbose:
            print(f"Running Scenario: [{sc_name}] (Mode: {coord_mode}, Opponent: {opp_strat})...")

        records, summary = evaluate_policy(
            checkpoint_path=checkpoint_path,
            num_episodes=episodes_per_scenario,
            coordination_mode=coord_mode,
            opponent_strategy=opp_strat,
            seed=seed,
        )
        all_episode_records.extend(records)
        scenario_summaries[sc_name] = summary

        if verbose:
            print(
                f"  -> Win Rate: {summary.get('win_rate', 0.0):.1%} | "
                f"Avg Reward: {summary.get('avg_team_reward', 0.0)} | "
                f"Survival Rate: {summary.get('friendly_survival_rate', 0.0):.1%} | "
                f"Avg Messages: {summary.get('avg_messages_exchanged', 0.0)}"
            )

    # Save outputs
    df_all = to_dataframe(all_episode_records)
    out_csv = Path(output_csv) if output_csv else Path(cfg["output"]["results_csv"])
    out_json = Path(output_json) if output_json else Path(cfg["output"]["summary_json"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(scenario_summaries, f, indent=2)

    # Compute overall comparative impact
    coord_wins = df_all[df_all["coordination_mode"] == "coordinated"]["winner"].eq("friendly").mean()
    indep_wins = df_all[df_all["coordination_mode"] == "independent"]["winner"].eq("friendly").mean()
    win_delta = coord_wins - indep_wins

    final_report = {
        "scenario_summaries": scenario_summaries,
        "overall_coordinated_win_rate": round(float(coord_wins), 4),
        "overall_independent_win_rate": round(float(indep_wins), 4),
        "win_rate_delta": round(float(win_delta), 4),
        "results_csv": str(out_csv),
        "summary_json": str(out_json),
    }

    if verbose:
        print(f"\n=======================================================")
        print(f"  EXPERIMENTAL COMPARISON SUMMARY                      ")
        print(f"=======================================================")
        print(f"Coordinated Agents Overall Win Rate : {coord_wins:.1%}")
        print(f"Independent Baseline Overall Win Rate: {indep_wins:.1%}")
        print(f"Communication Performance Delta     : {win_delta:+.1%}")
        print(f"Results successfully saved to: {out_csv}")
        print(f"=======================================================\n")

    return final_report
