"""Launch end-to-end STRATOS AI live demonstration."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from visualization.battlefield import BattlefieldRenderer
from training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Launch STRATOS AI Interactive Demonstration")
    parser.add_argument("--mode", type=str, default="coordinated", choices=["coordinated", "independent"])
    parser.add_argument("--opponent", type=str, default="aggressive", choices=["aggressive", "defensive", "resource_focused"])
    # Default to validated candidate fix checkpoint with +33.3% communication win-rate advantage
    default_ckpt = "artifacts/checkpoints/stratos_policy_candidate_fix_5000step.pt"
    if not Path(default_ckpt).exists():
        default_ckpt = "artifacts/checkpoints/stratos_policy_latest.pt"

    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to policy checkpoint")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  STRATOS AI — LIVE RESEARCH DEMONSTRATION")
    print("  Team: NEXUS AI")
    print("  'Collective Intelligence. Coordinated Execution.'")
    print("=" * 65)
    print(f"Mode              : {args.mode.upper()}")
    print(f"Opponent Strategy : {args.opponent.upper()}")
    print("\nControls:")
    print("  [SPACE]      Pause / Resume Simulation (or Next Match on Game Over)")
    print("  [M]          Toggle Coordinated / Independent Mode Live")
    print("  [R]          Reset Simulation Episode")
    print("  [UP/DOWN]    Increase / Decrease Simulation Speed (FPS)")
    print("  [Q / ESC]    Exit Demonstration Window\n")

    policy = None
    include_role_id = False
    if Path(args.checkpoint).exists():
        policy, meta = load_checkpoint(args.checkpoint)
        include_role_id = meta.get("obs_dim", 49) == 52
        print(f"Loaded trained neural policy: {args.checkpoint} (Steps: {meta.get('total_steps', 0)}, Obs Dim: {meta.get('obs_dim', 49)})\n")
    else:
        print("[Notice] Trained checkpoint not detected; demonstration will run autonomous heuristic baseline.\n")

    env = StrategyGameEnv(
        coordination_mode=args.mode,
        opponent_strategy=args.opponent,
        include_role_id=include_role_id,
    )

    try:
        renderer = BattlefieldRenderer(env=env, policy=policy, headless=False)
        renderer.run_episode()
    except Exception as e:
        print(f"Display rendering exception ({e}); running headless demonstration.")
        obs, _ = env.reset()
        for step in range(300):
            obs, rewards, terms, truncs, infos = env.step({"commander": 0, "defense": 5, "attack": 2})
            if any(terms.values()) or any(truncs.values()):
                break
        print(f"Demonstration episode concluded at step {env.state.step_count}. Winner: {env.state.winner}")


if __name__ == "__main__":
    main()
