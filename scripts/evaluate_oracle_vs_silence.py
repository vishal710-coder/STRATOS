"""Benchmark Oracle Communication vs Silence on held-out test seeds (5000-5099)."""

import numpy as np
from environment.strategy_game import StrategyGameEnv
from agents.reference_policies import OracleMessengerPolicy, SilentReferencePolicy


def evaluate_policy(policy, num_seeds: int = 100, seed_start: int = 5000, opponents=None):
    if opponents is None:
        opponents = ["aggressive", "defensive", "resource_focused"]

    results = []
    
    for opp in opponents:
        for s_idx in range(num_seeds):
            seed = seed_start + s_idx
            env = StrategyGameEnv(opponent_strategy=opp, enable_communication=True)
            obs, _ = env.reset(seed=seed)
            if hasattr(policy, "reset"):
                policy.reset()
            terminated = False
            truncated = False
            ep_reward = 0.0

            while not (terminated or truncated):
                actions = {}
                for agent in env.possible_agents:
                    act, msg = policy.select_action_and_message(agent, obs[agent])
                    actions[agent] = {"action": act, "message": msg}
                obs, rewards, terms, truncs, infos = env.step(actions)
                ep_reward += sum(rewards.values()) / len(rewards)
                terminated = any(terms.values())
                truncated = any(truncs.values())

            winner = env.state.winner
            is_win = (winner == "friendly")
            is_loss = (winner == "enemy")
            is_draw = (winner is None or winner == "draw")
            results.append({
                "opponent": opp,
                "seed": seed,
                "winner": winner,
                "is_win": is_win,
                "is_loss": is_loss,
                "is_draw": is_draw,
                "reward": ep_reward,
                "steps": env.state.step_count,
            })

    win_rate = np.mean([r["is_win"] for r in results]) * 100.0
    draw_rate = np.mean([r["is_draw"] for r in results]) * 100.0
    loss_rate = np.mean([r["is_loss"] for r in results]) * 100.0
    rewards = [r["reward"] for r in results]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    ci_reward = 1.96 * (std_reward / np.sqrt(len(rewards)))

    return {
        "num_episodes": len(results),
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "mean_reward": mean_reward,
        "ci_reward": ci_reward,
        "std_reward": std_reward,
    }


def main():
    print("Evaluating Oracle Messenger Policy on held-out seeds (5000-5099)...")
    oracle = OracleMessengerPolicy()
    res_oracle = evaluate_policy(oracle, num_seeds=100, seed_start=5000)
    print(f"Oracle: Win={res_oracle['win_rate']:.1f}%, Draw={res_oracle['draw_rate']:.1f}%, Loss={res_oracle['loss_rate']:.1f}%, Reward={res_oracle['mean_reward']:.2f} +/- {res_oracle['ci_reward']:.2f}")

    print("\nEvaluating Silent Reference Policy on held-out seeds (5000-5099)...")
    silent = SilentReferencePolicy()
    res_silent = evaluate_policy(silent, num_seeds=100, seed_start=5000)
    print(f"Silent: Win={res_silent['win_rate']:.1f}%, Draw={res_silent['draw_rate']:.1f}%, Loss={res_silent['loss_rate']:.1f}%, Reward={res_silent['mean_reward']:.2f} +/- {res_silent['ci_reward']:.2f}")

    gap_win = res_oracle['win_rate'] - res_silent['win_rate']
    gap_reward = res_oracle['mean_reward'] - res_silent['mean_reward']
    print(f"\n==========================================")
    print(f"COMMUNICATION GAP (ORACLE - SILENCE):")
    print(f"Win Rate Gap: +{gap_win:.1f}%")
    print(f"Reward Gap:   +{gap_reward:.2f}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
