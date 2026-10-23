"""Diagnose reward components for STRATOS AI policies.

Breaks down per-episode reward contributions for:
1. Random Action Policy
2. Scripted Heuristic Policy
3. NO_COMM Policy (Seed 42)
4. COMM Policy (Seed 42)
Across held-out seeds (5000..5049) to isolate why passive/random policies score well.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from environment.actions import ActionType
from training.checkpoint import load_checkpoint
from scripts.audit_and_evaluate_v2 import RandomPolicy, ScriptedHeuristicPolicy


def evaluate_with_reward_breakdown(
    env: StrategyGameEnv,
    ep_seed: int,
    policy: Any = None,
    baseline_type: str = None,
    device: str = "cpu",
) -> Dict[str, float]:
    rng = np.random.default_rng(ep_seed)
    obs, _ = env.reset(seed=ep_seed)

    breakdown = {
        "terminal_win": 0.0,
        "terminal_loss": 0.0,
        "terminal_draw": 0.0,
        "step_penalty": 0.0,
        "survival_bonus": 0.0,
        "enemy_unit_dmg": 0.0,
        "friendly_unit_dmg": 0.0,
        "enemy_base_dmg": 0.0,
        "friendly_base_dmg": 0.0,
        "enemy_kills": 0.0,
        "friendly_losses": 0.0,
        "total_reward": 0.0,
        "steps": 0,
        "friendly_survived": 0,
        "enemy_eliminated": 0,
    }

    r_calc = env.reward_calculator
    random_bot = RandomPolicy()
    heuristic_bot = ScriptedHeuristicPolicy()

    while True:
        action_dict = {}
        if baseline_type == "random":
            for a in env.possible_agents:
                action_dict[a] = random_bot.select_action(a, obs[a], rng)
        elif baseline_type == "heuristic":
            for a in env.possible_agents:
                action_dict[a] = heuristic_bot.select_action(a, obs[a], rng)
        elif policy is not None:
            agent_obs = np.array([obs[a] for a in env.possible_agents])
            obs_t = torch.as_tensor(agent_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                if getattr(policy, "has_comm_head", False):
                    action_t, msg_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=True)
                    for i, a in enumerate(env.possible_agents):
                        action_dict[a] = {"action": int(action_t[i].item()), "message": int(msg_t[i].item())}
                else:
                    action_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
                    for i, a in enumerate(env.possible_agents):
                        action_dict[a] = int(action_t[i].item())

        # Step environment
        prev_state = env.state
        next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
        curr_state = env.state
        obs = next_obs
        step_events = getattr(env, "last_step_events", {})

        # Log individual reward components (summed across all 3 agents to match total_team_reward)
        n_agents = len(env.possible_agents)
        breakdown["step_penalty"] += r_calc.step_penalty * n_agents
        alive_count = sum(1 for u in curr_state.friendly_units.values() if u.is_alive)
        breakdown["survival_bonus"] += (alive_count * r_calc.survival_bonus) * n_agents

        e_base_dmg = step_events.get("enemy_base_damage", 0.0)
        f_base_dmg = step_events.get("friendly_base_damage", 0.0)
        e_unit_dmg = step_events.get("enemy_unit_damage", 0.0)
        f_unit_dmg = step_events.get("friendly_unit_damage", 0.0)
        e_kills = step_events.get("enemy_units_killed", 0)
        f_losses = step_events.get("friendly_units_lost", 0)

        breakdown["enemy_base_dmg"] += (e_base_dmg * r_calc.enemy_base_dmg_scale) * n_agents
        breakdown["friendly_base_dmg"] += (f_base_dmg * r_calc.friendly_base_dmg_scale) * n_agents
        breakdown["enemy_unit_dmg"] += (e_unit_dmg * r_calc.enemy_unit_dmg_scale) * n_agents
        breakdown["friendly_unit_dmg"] += (f_unit_dmg * r_calc.friendly_unit_dmg_scale) * n_agents
        breakdown["enemy_kills"] += (e_kills * r_calc.enemy_unit_destroyed_reward) * n_agents
        breakdown["friendly_losses"] += (f_losses * r_calc.friendly_unit_lost_penalty) * n_agents

        if curr_state.terminated or curr_state.truncated:
            if curr_state.winner == "friendly":
                breakdown["terminal_win"] += r_calc.win_reward * n_agents
            elif curr_state.winner == "enemy":
                breakdown["terminal_loss"] += r_calc.loss_penalty * n_agents
            elif curr_state.winner == "draw":
                breakdown["terminal_draw"] += r_calc.draw_reward * n_agents

        breakdown["total_reward"] += sum(rewards.values())

        if any(terminations.values()) or any(truncations.values()):
            break

    breakdown["steps"] = curr_state.step_count
    breakdown["friendly_survived"] = sum(1 for u in curr_state.friendly_units.values() if u.is_alive)
    breakdown["enemy_eliminated"] = sum(1 for u in curr_state.enemy_units.values() if not u.is_alive)
    breakdown["winner"] = curr_state.winner or "draw"
    breakdown["win_type"] = curr_state.win_type or "unknown"

    return breakdown


def main():
    seeds = list(range(5000, 5030))  # 30 seeds per opponent
    opponents = ["aggressive", "defensive", "resource_focused"]

    comm_ckpt = "artifacts/checkpoints/stratos_policy_comm_seed42.pt"
    nocomm_ckpt = "artifacts/checkpoints/stratos_policy_nocomm_seed42.pt"

    comm_pol, _ = load_checkpoint(comm_ckpt) if Path(comm_ckpt).exists() else (None, {})
    nocomm_pol, _ = load_checkpoint(nocomm_ckpt) if Path(nocomm_ckpt).exists() else (None, {})

    conditions = {
        "Random": (None, "random", "independent"),
        "Scripted Heuristic": (None, "heuristic", "independent"),
        "NO_COMM (Seed 42)": (nocomm_pol, None, "independent"),
        "COMM (Seed 42)": (comm_pol, None, "coordinated"),
    }

    all_rows = []

    print("Running reward diagnostic across 30 held-out seeds x 3 opponents...")
    for cond_name, (pol, b_type, mode) in conditions.items():
        for opp in opponents:
            env = StrategyGameEnv(coordination_mode=mode, opponent_strategy=opp, include_role_id=True)
            for s in seeds:
                row = evaluate_with_reward_breakdown(env=env, ep_seed=s, policy=pol, baseline_type=b_type)
                row["condition"] = cond_name
                row["opponent"] = opp
                row["seed"] = s
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    cols = [
        "terminal_win", "terminal_loss", "terminal_draw", "step_penalty",
        "survival_bonus", "enemy_unit_dmg", "friendly_unit_dmg",
        "enemy_base_dmg", "friendly_base_dmg", "enemy_kills", "friendly_losses",
        "total_reward", "steps", "friendly_survived", "enemy_eliminated"
    ]

    summary = df.groupby("condition")[cols].mean()
    print("\n================== REWARD COMPONENT BREAKDOWN (MEANS) ==================")
    print(summary.to_string())

    win_dist = df.groupby(["condition", "winner"]).size().unstack().fillna(0)
    print("\n================== OUTCOME DISTRIBUTION ==================")
    print(win_dist.to_string())

    out_csv = Path("artifacts/experiments/reward_diagnosis.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full per-episode breakdown to: {out_csv}")


if __name__ == "__main__":
    main()
