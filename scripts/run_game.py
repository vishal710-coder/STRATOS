"""Interactive Presentation Runner for STRATOS AI Tactical Simulation.

Supports key toggle or command-line selection across:
- ORACLE Messenger Policy ([1] Key)
- SILENT Reference Policy ([2] Key)
- Learned COMM Policy ([3] Key)
- Learned NO_COMM Policy ([4] Key)
- Replay same seed ([R] Key), next seed ([N] Key), toggle opponent ([O] Key)
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from visualization.battlefield import BattlefieldRenderer, DEMO_MODES
from agents.reference_policies import OracleMessengerPolicy, SilentReferencePolicy
from training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run STRATOS AI Strategy Simulation Game & Expo Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="oracle",
        choices=["oracle", "silent", "comm", "no_comm", "coordinated", "independent"],
        help="Demo mode: oracle, silent, comm, no_comm (or coordinated, independent)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="aggressive",
        choices=["aggressive", "defensive", "resource_focused", "mixed"],
        help="Opponent archetype",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--seed", type=int, default=5000, help="Initial episode seed (default: 5000)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    parser.add_argument("--steps", type=int, default=300, help="Max steps for episode")
    args = parser.parse_args()

    # Map legacy mode names
    mode_map = {
        "coordinated": "oracle",
        "independent": "silent",
        "oracle": "oracle",
        "silent": "silent",
        "comm": "comm",
        "no_comm": "no_comm",
    }
    demo_mode = mode_map.get(args.mode.lower(), "oracle")

    print("\n=======================================================")
    print("  STRATOS AI — Expo Interactive Presentation Runner   ")
    print("=======================================================")
    print(f"Active Mode : {demo_mode.upper()} ([1] Oracle | [2] Silent | [3] Comm | [4] NoComm)")
    print(f"Opponent    : {args.opponent.upper()} ([O] Toggle)")
    print(f"Episode Seed: {args.seed} ([R] Replay | [N] Next)")
    print(f"Display     : {'Headless Console' if args.headless else 'Interactive Pygame HUD'}")
    print("=======================================================\n")

    coord_mode = "coordinated" if demo_mode in ("oracle", "comm") else "independent"

    env = StrategyGameEnv(
        coordination_mode=coord_mode,
        opponent_strategy=args.opponent if args.opponent != "mixed" else "aggressive",
        include_role_id=True,
    )

    if not args.headless:
        try:
            renderer = BattlefieldRenderer(
                env=env,
                headless=False,
                demo_mode=demo_mode,
                seed=args.seed,
                comm_checkpoint=args.checkpoint if demo_mode == "comm" else None,
                nocomm_checkpoint=args.checkpoint if demo_mode == "no_comm" else None,
            )
            renderer.run_episode()
            return
        except Exception as e:
            print(f"Display rendering exception ({e}); switching to headless console simulation.")

    # Headless simulation loop
    obs, _ = env.reset(seed=args.seed)
    policy_obj = OracleMessengerPolicy() if demo_mode == "oracle" else SilentReferencePolicy()

    step = 0
    while step < args.steps:
        step += 1
        actions = {}
        for a in env.possible_agents:
            act, msg = policy_obj.select_action_and_message(a, obs[a])
            actions[a] = {"action": int(act), "message": int(msg)}

        obs, rewards, terms, truncs, infos = env.step(actions)
        if any(terms.values()) or any(truncs.values()):
            break

    state = env.state
    assert state is not None
    print(f"\n--- Episode Finished at Step {state.step_count} ---")
    print(f"Mode              : {demo_mode.upper()}")
    print(f"Winner            : {state.winner} (Type: {state.win_type})")
    print(f"Friendly HQ HP    : {state.bases['friendly'].hp} | Enemy HQ HP: {state.bases['enemy'].hp}")
    print(f"Friendly Resources: {state.friendly_resources}")
    print(f"Weak Lane         : {state.weak_lane} (0: Top, 1: Mid, 2: Bot)")
    print(f"Wave Flank Alert  : {state.wave_flank} (Warning Active: {state.wave_warning_active})")
    print(f"Messages Exchanged: {len(env.coordinator.message_history)}")


if __name__ == "__main__":
    main()
