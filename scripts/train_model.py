"""CLI script to train PPO multi-agent policy."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import train_stratos


def main():
    parser = argparse.ArgumentParser(description="Train STRATOS AI Multi-Agent PPO Policy")
    parser.add_argument("--timesteps", type=int, default=12000, help="Total environment timesteps to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mode", type=str, default="coordinated", choices=["coordinated", "independent"], help="Coordination mode")
    parser.add_argument("--opponent", type=str, default="aggressive", choices=["aggressive", "defensive", "resource_focused", "mixed"], help="Opponent strategy")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--output-checkpoint", type=str, default=None, help="Path to save trained checkpoint")
    parser.add_argument("--history-csv", type=str, default=None, help="Path to save training history CSV")
    parser.add_argument("--history-json", type=str, default=None, help="Path to save training history JSON")
    parser.add_argument("--role-conditioning", action="store_true", default=True, help="Enable 3-dim one-hot agent role identification in observation")
    parser.add_argument("--config", type=str, default="config/training.yaml", help="Path to training config YAML")
    parser.add_argument("--game-config", type=str, default="config/game.yaml", help="Path to game config YAML")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  STRATOS AI — PPO Multi-Agent Training Launcher        ")
    print(f"=======================================================")
    print(f"Timesteps: {args.timesteps} | Seed: {args.seed} | Mode: {args.mode} | Opponent: {args.opponent} | Role Conditioning: {args.role_conditioning}")
    if args.resume_checkpoint:
        print(f"Resuming From: {args.resume_checkpoint}")
    if args.output_checkpoint:
        print(f"Saving Output: {args.output_checkpoint}")

    results = train_stratos(
        config_path=args.config,
        game_config_path=args.game_config,
        total_timesteps=args.timesteps,
        seed=args.seed,
        coordination_mode=args.mode,
        opponent_strategy=args.opponent,
        resume_checkpoint=args.resume_checkpoint,
        output_checkpoint_path=args.output_checkpoint,
        history_csv_path=args.history_csv,
        history_json_path=args.history_json,
        include_role_id=args.role_conditioning,
        verbose=True,
    )

    print(f"\nTraining pipeline completed successfully.")
    print(f"Final Checkpoint: {results['final_checkpoint']}")
    print(f"Total Episodes  : {results['total_episodes']} (Trained: {results['episodes_trained']})")
    print(f"Total Steps     : {results['total_steps']} (Trained: {results['steps_trained']})")
    print(f"Final Win Rate  : {results['final_win_rate']:.1%}")
    print(f"Duration        : {results['training_duration_sec']}s")


if __name__ == "__main__":
    main()
