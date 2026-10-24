"""Evaluate Phase 2 Fixes and OracleV2 on Validation Seeds 1000-1049.

Compares:
1. Oracle v1 (default env)
2. Oracle v2 (default env)
3. Oracle v2 with hq_regen=True
4. Best COMM with default env
5. Best COMM with hq_regen=True

Applies decision rule:
Adopt oracle_v2 ONLY if it improves win rate by >= 5 percentage points across all 3 opponents without increasing draw rate.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from agents.reference_policies import OracleMessengerPolicy, OracleV2Policy
from training.checkpoint import load_checkpoint


def evaluate_policy_condition(
    policy_fn,
    mode: str,
    seeds: List[int],
    opponents: List[str],
    hq_regen: bool = False,
    supply_broadcast: bool = False,
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    results = []

    for opp in opponents:
        for s in seeds:
            env = StrategyGameEnv(
                opponent_strategy=opp,
                coordination_mode="coordinated",
                enable_communication=True,
                include_role_id=True,
                enable_team_cohesion=True,
                hq_regen=hq_regen,
                supply_broadcast=supply_broadcast,
            )
            obs, _ = env.reset(seed=s)
            pol = policy_fn()
            if hasattr(pol, "eval"):
                pol.eval()
            if hasattr(pol, "reset"):
                pol.reset()

            done = False
            ep_reward = 0.0

            while not done:
                if mode in ("oracle", "oracle_v2"):
                    acts = {
                        a: {
                            "action": pol.select_action_and_message(a, obs[a])[0],
                            "message": pol.select_action_and_message(a, obs[a])[1],
                        }
                        for a in env.possible_agents
                    }
                else:
                    obs_t = torch.as_tensor(np.array([obs[a] for a in env.possible_agents]), dtype=torch.float32)
                    with torch.no_grad():
                        a_t, m_t, _, _, _, _ = pol.get_actions_and_values(obs_t, deterministic=True)
                    acts = {
                        a: {"action": int(a_t[i].item()), "message": int(m_t[i].item())}
                        for i, a in enumerate(env.possible_agents)
                    }

                obs, r, term, trunc, _ = env.step(acts)
                ep_reward += sum(r.values()) / len(r)
                done = any(term.values()) or any(trunc.values())

            w = env.state.winner if env.state else "draw"
            results.append({
                "opponent": opp,
                "seed": s,
                "win": w == "friendly",
                "loss": w == "enemy",
                "draw": w not in ("friendly", "enemy"),
                "reward": ep_reward,
                "steps": env.state.step_count if env.state else 300,
            })

    total = len(results)
    by_opp = {}
    for opp in opponents:
        opp_r = [r for r in results if r["opponent"] == opp]
        by_opp[opp] = {
            "win_rate": float(np.mean([r["win"] for r in opp_r]) * 100.0),
            "draw_rate": float(np.mean([r["draw"] for r in opp_r]) * 100.0),
            "loss_rate": float(np.mean([r["loss"] for r in opp_r]) * 100.0),
            "mean_reward": float(np.mean([r["reward"] for r in opp_r])),
        }

    return {
        "overall_win_rate": float(np.mean([r["win"] for r in results]) * 100.0),
        "overall_draw_rate": float(np.mean([r["draw"] for r in results]) * 100.0),
        "overall_loss_rate": float(np.mean([r["loss"] for r in results]) * 100.0),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "by_opponent": by_opp,
    }


def main():
    print("=" * 70)
    print("  STRATOS AI v5 — PHASE 2: EVALUATION OF ENVIRONMENT & STRATEGY FIXES")
    print("=" * 70)

    seeds = list(range(1000, 1050))
    opponents = ["aggressive", "defensive", "resource_focused"]

    # Load comm best once
    comm_pol, _ = load_checkpoint("artifacts/checkpoints/comm_best.pt", device="cpu")
    comm_pol.eval()

    experiments = {
        "Oracle_v1 (Default)": (lambda: OracleMessengerPolicy(), "oracle", False, False),
        "Oracle_v2 (Default)": (lambda: OracleV2Policy(), "oracle_v2", False, False),
        "Oracle_v2 + HQ Regen": (lambda: OracleV2Policy(), "oracle_v2", True, False),
        "Best_COMM (Default)": (lambda: comm_pol, "comm", False, False),
        "Best_COMM + HQ Regen": (lambda: comm_pol, "comm", True, False),
    }

    summary = {}
    for name, (p_fn, mode, hq_r, sp_b) in experiments.items():
        t0 = time.time()
        print(f"\n[Phase 2] Evaluating '{name}' across {len(seeds)*len(opponents)} episodes...")
        res = evaluate_policy_condition(p_fn, mode, seeds, opponents, hq_regen=hq_r, supply_broadcast=sp_b)
        dt = time.time() - t0
        summary[name] = res
        print(f"  Result: Win={res['overall_win_rate']:.1f}% | Draw={res['overall_draw_rate']:.1f}% | "
              f"Loss={res['overall_loss_rate']:.1f}% | Reward={res['mean_reward']:.2f} ({dt:.1f}s)")

    # Print comparative matrix
    print("\n" + "=" * 70)
    print("  PHASE 2 SUMMARY & DECISION AUDIT")
    print("=" * 70)
    print(json.dumps(summary, indent=2))

    # Evaluate decision rule
    v1_win = summary["Oracle_v1 (Default)"]["overall_win_rate"]
    v2_win = summary["Oracle_v2 (Default)"]["overall_win_rate"]
    v1_draw = summary["Oracle_v1 (Default)"]["overall_draw_rate"]
    v2_draw = summary["Oracle_v2 (Default)"]["overall_draw_rate"]

    diff_win = v2_win - v1_win
    diff_draw = v2_draw - v1_draw

    print("\nDecision Rule Check:")
    print(f"  Oracle v1 Win Rate: {v1_win:.1f}% (Draw: {v1_draw:.1f}%)")
    print(f"  Oracle v2 Win Rate: {v2_win:.1f}% (Draw: {v2_draw:.1f}%)")
    print(f"  Win Rate Improvement: {diff_win:+.1f} points (Threshold: >= +5.0 points)")
    print(f"  Draw Rate Change: {diff_draw:+.1f} points (Must not increase)")

    adopt = (diff_win >= 5.0) and (diff_draw <= 0.0)
    print(f"  Adopt Oracle v2 into new BC student? {'YES' if adopt else 'NO (Retain Oracle v1 and comm_bc_student.pt)'}")

    summary["decision_rule"] = {
        "v1_win_rate": v1_win,
        "v2_win_rate": v2_win,
        "win_diff": diff_win,
        "draw_diff": diff_draw,
        "adopted": adopt,
    }

    out_file = Path("artifacts/experiments/phase2_evaluation_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
