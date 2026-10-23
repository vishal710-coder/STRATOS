"""CLI script to run interactive game simulation."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from visualization.battlefield import BattlefieldRenderer
from training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run STRATOS AI Strategy Simulation Game")
    parser.add_argument("--mode", type=str, default="coordinated", choices=["coordinated", "independent"], help="Coordination mode")
    parser.add_argument("--opponent", type=str, default="aggressive", choices=["aggressive", "defensive", "resource_focused"], help="Opponent strategy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    parser.add_argument("--steps", type=int, default=300, help="Max steps for episode")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  STRATOS AI — Game Simulation Runner                   ")
    print(f"=======================================================")
    print(f"Mode: {args.mode} | Opponent: {args.opponent} | Headless: {args.headless}")

    policy = None
    include_role_id = False
    if args.checkpoint and Path(args.checkpoint).exists():
        policy, meta = load_checkpoint(args.checkpoint)
        include_role_id = meta.get("obs_dim", 67) in (52, 67) or meta.get("include_role_id", True)
        print(f"Loaded policy checkpoint: {args.checkpoint} (Trained steps: {meta.get('total_steps', 0)})")
    else:
        # Check if default candidate fix exists, else latest
        default_candidate = Path("artifacts/checkpoints/stratos_policy_candidate_fix_5000step.pt")
        default_latest = Path("artifacts/checkpoints/stratos_policy_latest.pt")
        target_default = default_latest if default_latest.exists() else default_candidate
        if target_default.exists():
            policy, meta = load_checkpoint(str(target_default))
            include_role_id = meta.get("obs_dim", 67) in (52, 67) or meta.get("include_role_id", True)
            print(f"Loaded default policy checkpoint: {target_default} (Trained steps: {meta.get('total_steps', 0)})")
        else:
            print("No checkpoint found; running with autonomous heuristics.")

    env = StrategyGameEnv(
        coordination_mode=args.mode,
        opponent_strategy=args.opponent,
        include_role_id=include_role_id,
    )

    if not args.headless:
        try:
            renderer = BattlefieldRenderer(env=env, policy=policy, headless=False)
            renderer.run_episode()
            return
        except Exception as e:
            print(f"Display rendering exception ({e}); switching to headless console simulation.")

    # Headless fallback / execution
    obs, _ = env.reset()
    step = 0
    while step < args.steps:
        step += 1
        actions = {}
        if policy is not None:
            import torch
            import numpy as np
            agent_obs = np.array([obs[a] for a in env.possible_agents])
            obs_t = torch.as_tensor(agent_obs, dtype=torch.float32)
            with torch.no_grad():
                if getattr(policy, "has_comm_head", False):
                    a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=True)
                    for i, a in enumerate(env.possible_agents):
                        actions[a] = {"action": int(a_t[i].item()), "message": int(m_t[i].item())}
                else:
                    a_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
                    for i, a in enumerate(env.possible_agents):
                        actions[a] = int(a_t[i].item())
        else:
            for a in env.possible_agents:
                actions[a] = 0

        obs, rewards, terms, truncs, infos = env.step(actions)
        if any(terms.values()) or any(truncs.values()):
            break

    state = env.state
    assert state is not None
    print(f"\n--- Episode Finished at Step {state.step_count} ---")
    print(f"Winner: {state.winner}")
    print(f"Friendly HQ HP: {state.bases['friendly'].hp} | Enemy HQ HP: {state.bases['enemy'].hp}")
    print(f"Friendly Resources: {state.friendly_resources}")
    print(f"Messages Exchanged: {len(env.coordinator.message_history)}")


if __name__ == "__main__":
    main()
