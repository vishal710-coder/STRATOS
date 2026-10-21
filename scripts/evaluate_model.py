"""CLI script to evaluate trained model checkpoints."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.compare import run_comparative_experiments
from evaluation.evaluate import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate STRATOS AI Multi-Agent Policy")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/stratos_policy_latest.pt", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=15, help="Number of evaluation episodes per scenario")
    parser.add_argument("--mode", type=str, default="coordinated", choices=["coordinated", "independent"], help="Coordination mode")
    parser.add_argument("--opponent", type=str, default="aggressive", choices=["aggressive", "defensive", "resource_focused"], help="Opponent strategy")
    parser.add_argument("--compare", action="store_true", default=True, help="Run complete comparative suite against all opponents")
    parser.add_argument("--seed", type=int, default=12345, help="Evaluation random seed")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save evaluation CSV")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save evaluation JSON")
    args = parser.parse_args()

    ckpt_path = args.checkpoint if Path(args.checkpoint).exists() else None
    if ckpt_path is None:
        print(f"[Notice] Checkpoint '{args.checkpoint}' not found; evaluating with autonomous baseline.")

    if args.compare:
        run_comparative_experiments(
            checkpoint_path=ckpt_path,
            episodes_per_scenario=args.episodes,
            seed=args.seed,
            verbose=True,
            output_csv=args.output_csv,
            output_json=args.output_json,
        )
    else:
        records, summary = evaluate_policy(
            checkpoint_path=ckpt_path,
            num_episodes=args.episodes,
            coordination_mode=args.mode,
            opponent_strategy=args.opponent,
            seed=args.seed,
        )
        print(f"\n=======================================================")
        print(f"  EVALUATION SUMMARY: {args.mode.upper()} vs {args.opponent.upper()} ")
        print(f"=======================================================")
        print(f"Episodes: {summary['total_episodes']} | Wins: {summary['wins']} ({summary['win_rate']:.1%})")
        print(f"Avg Team Reward     : {summary['avg_team_reward']}")
        print(f"Friendly Survival   : {summary['friendly_survival_rate']:.1%}")
        print(f"Enemy Units Killed  : {summary['avg_enemy_eliminated']} / 3.0")
        print(f"Avg Messages Sent   : {summary['avg_messages_exchanged']}")
        print(f"Avg Episode Duration: {summary['avg_duration']} steps")


if __name__ == "__main__":
    main()
