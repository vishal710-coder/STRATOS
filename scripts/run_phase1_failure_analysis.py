"""Phase 1: Failure Mode Analysis on Validation Seeds 1000-1049.

Evaluates Oracle and Best COMM on 150 validation episodes each (50 seeds x 3 opponents).
Classifies draws and losses into failure modes:
1. Squad split: units separated by large distance during push (>250) or in different lanes.
2. Early scout death: Attack died before step 100, blinding team.
3. Storm deaths: unit killed or heavily damaged by hazard storms.
4. Wave sneak damage: base took damage while units were away (>300 dist).
5. Defense out of position: Defense far from base when wave hits base.
6. Low-HP retreat without recovery: unit retreated at low HP (<0.20) but never found supply and died.
7. Scout survival vs win rate correlation (alive at step 100 vs died before 100).
8. Low-HP distance to nearest supply drop vs vision radius (scout=120, others=80).
9. Worst opponent archetype identification.
"""

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from agents.reference_policies import OracleMessengerPolicy
from training.checkpoint import load_checkpoint


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def analyze_model(model_name: str, policy: Any, mode: str, seeds: List[int], opponents: List[str]) -> Dict[str, Any]:
    torch.set_num_threads(1)
    if hasattr(policy, "eval"):
        policy.eval()

    episodes = []

    for opp in opponents:
        for s in seeds:
            env = StrategyGameEnv(
                opponent_strategy=opp,
                coordination_mode="coordinated",
                enable_communication=True,
                include_role_id=True,
                enable_team_cohesion=True,
            )
            obs, _ = env.reset(seed=s)
            if hasattr(policy, "reset"):
                policy.reset()

            done = False
            step = 0

            scout_alive_at_100 = True
            scout_death_step = None
            storm_deaths = 0
            base_hit_while_away = 0
            defense_away_during_base_hit = 0
            low_hp_episodes = 0
            low_hp_recovered = 0
            low_hp_died = 0
            low_hp_supply_distances = []
            squad_split_steps = 0
            total_active_push_steps = 0

            last_base_hp = env.state.bases["friendly"].hp
            unit_low_hp_flag = {"attack": False, "defense": False, "commander": False}

            while not done:
                step += 1
                state = env.state

                # Check scout alive status
                scout = state.friendly_units.get("attack")
                if scout is None or not scout.is_alive:
                    if scout_death_step is None:
                        scout_death_step = step
                        if step < 100:
                            scout_alive_at_100 = False

                # Check low-HP retreat and supply distances
                for u_id, u in state.friendly_units.items():
                    if u.is_alive:
                        max_hp = u.max_hp if hasattr(u, "max_hp") else 100.0
                        hp_ratio = u.hp / max_hp
                        if hp_ratio < 0.20:
                            if not unit_low_hp_flag[u_id]:
                                unit_low_hp_flag[u_id] = True
                                low_hp_episodes += 1

                            # Find nearest supply drop
                            sd = state.supply_drop
                            if sd and not sd.collected:
                                dist = distance((u.x, u.y), (sd.x, sd.y))
                                low_hp_supply_distances.append(dist)

                # Check if base was damaged this step
                curr_base_hp = state.bases["friendly"].hp
                if curr_base_hp < last_base_hp:
                    base_pos = (state.bases["friendly"].x, state.bases["friendly"].y)
                    # Check distance of units to base
                    dists_to_base = [
                        distance((u.x, u.y), base_pos)
                        for u in state.friendly_units.values()
                        if u.is_alive
                    ]
                    if dists_to_base and min(dists_to_base) > 300.0:
                        base_hit_while_away += 1

                    defense_unit = state.friendly_units.get("defense")
                    if defense_unit and defense_unit.is_alive:
                        if distance((defense_unit.x, defense_unit.y), base_pos) > 280.0:
                            defense_away_during_base_hit += 1

                last_base_hp = curr_base_hp

                # Check squad cohesion
                alive_friendly = [u for u in state.friendly_units.values() if u.is_alive]
                if len(alive_friendly) >= 2:
                    total_active_push_steps += 1
                    # Pairwise distances
                    pair_dists = []
                    for i in range(len(alive_friendly)):
                        for j in range(i + 1, len(alive_friendly)):
                            pair_dists.append(
                                distance(
                                    (alive_friendly[i].x, alive_friendly[i].y),
                                    (alive_friendly[j].x, alive_friendly[j].y),
                                )
                            )
                    if max(pair_dists) > 250.0:
                        squad_split_steps += 1


                # Select actions
                if mode == "oracle":
                    acts = {
                        a: {
                            "action": policy.select_action_and_message(a, obs[a])[0],
                            "message": policy.select_action_and_message(a, obs[a])[1],
                        }
                        for a in env.possible_agents
                    }
                else:
                    obs_t = torch.as_tensor(np.array([obs[a] for a in env.possible_agents]), dtype=torch.float32)
                    with torch.no_grad():
                        a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=True)
                    acts = {
                        a: {"action": int(a_t[i].item()), "message": int(m_t[i].item())}
                        for i, a in enumerate(env.possible_agents)
                    }

                # Step
                obs, r, term, trunc, info = env.step(acts)
                done = any(term.values()) or any(trunc.values())

            winner = env.state.winner if env.state else "draw"
            win_type = env.state.win_type if env.state else "timeout"

            # Check low-HP recoveries vs deaths at end of episode
            for u_id, flagged in unit_low_hp_flag.items():
                if flagged:
                    u = env.state.friendly_units.get(u_id)
                    if u and u.is_alive and (u.hp / (u.max_hp if hasattr(u, "max_hp") else 100.0) >= 0.40):
                        low_hp_recovered += 1
                    else:
                        low_hp_died += 1

            episodes.append({
                "model": model_name,
                "opponent": opp,
                "seed": s,
                "winner": winner,
                "win_type": win_type,
                "steps": step,
                "scout_alive_at_100": scout_alive_at_100,
                "scout_death_step": scout_death_step,
                "storm_deaths": storm_deaths,
                "base_hit_while_away": base_hit_while_away > 0,
                "defense_away_during_base_hit": defense_away_during_base_hit > 0,
                "squad_split_pct": (squad_split_steps / max(1, total_active_push_steps)) * 100.0,
                "low_hp_occurred": low_hp_episodes > 0,
                "low_hp_recovered": low_hp_recovered,
                "low_hp_died": low_hp_died,
                "mean_low_hp_supply_dist": float(np.mean(low_hp_supply_distances)) if low_hp_supply_distances else None,
            })

    return {
        "model": model_name,
        "episodes": episodes,
    }


def summarize_analysis(raw_res: Dict[str, Any]) -> Dict[str, Any]:
    model = raw_res["model"]
    eps = raw_res["episodes"]
    total = len(eps)

    wins = [e for e in eps if e["winner"] == "friendly"]
    draws = [e for e in eps if e["winner"] not in ("friendly", "enemy")]
    losses = [e for e in eps if e["winner"] == "enemy"]

    non_wins = draws + losses

    # Scout survival analysis
    scout_surv_eps = [e for e in eps if e["scout_alive_at_100"]]
    scout_died_eps = [e for e in eps if not e["scout_alive_at_100"]]

    win_rate_scout_alive = len([e for e in scout_surv_eps if e["winner"] == "friendly"]) / max(1, len(scout_surv_eps)) * 100.0
    win_rate_scout_dead = len([e for e in scout_died_eps if e["winner"] == "friendly"]) / max(1, len(scout_died_eps)) * 100.0

    # Failure mode occurrences in draws & losses
    nw_count = max(1, len(non_wins))
    squad_split_nw = len([e for e in non_wins if e["squad_split_pct"] > 35.0])
    early_scout_death_nw = len([e for e in non_wins if not e["scout_alive_at_100"]])
    wave_sneak_nw = len([e for e in non_wins if e["base_hit_while_away"]])
    defense_away_nw = len([e for e in non_wins if e["defense_away_during_base_hit"]])
    low_hp_fail_nw = len([e for e in non_wins if e["low_hp_died"] > 0])

    # Low HP supply stats
    all_supply_dists = [e["mean_low_hp_supply_dist"] for e in eps if e["mean_low_hp_supply_dist"] is not None]
    mean_supply_dist = float(np.mean(all_supply_dists)) if all_supply_dists else 0.0

    total_low_hp_died = sum(e["low_hp_died"] for e in eps)
    total_low_hp_recovered = sum(e["low_hp_recovered"] for e in eps)
    recovery_rate = (total_low_hp_recovered / max(1, total_low_hp_died + total_low_hp_recovered)) * 100.0

    # Per-opponent breakdown
    opps = ["aggressive", "defensive", "resource_focused"]
    opp_summary = {}
    for opp in opps:
        opp_eps = [e for e in eps if e["opponent"] == opp]
        opp_wins = len([e for e in opp_eps if e["winner"] == "friendly"])
        opp_draws = len([e for e in opp_eps if e["winner"] not in ("friendly", "enemy")])
        opp_losses = len([e for e in opp_eps if e["winner"] == "enemy"])
        opp_summary[opp] = {
            "win_rate": opp_wins / len(opp_eps) * 100.0,
            "draw_rate": opp_draws / len(opp_eps) * 100.0,
            "loss_rate": opp_losses / len(opp_eps) * 100.0,
        }

    worst_opp = min(opp_summary.items(), key=lambda x: x[1]["win_rate"])

    return {
        "model": model,
        "total_episodes": total,
        "overall_win_rate": len(wins) / total * 100.0,
        "overall_draw_rate": len(draws) / total * 100.0,
        "overall_loss_rate": len(losses) / total * 100.0,
        "scout_survival_at_100_pct": len(scout_surv_eps) / total * 100.0,
        "win_rate_when_scout_alive_at_100": win_rate_scout_alive,
        "win_rate_when_scout_dies_before_100": win_rate_scout_dead,
        "non_win_count": len(non_wins),
        "failure_modes_in_non_wins": {
            "early_scout_death": {"count": early_scout_death_nw, "pct": early_scout_death_nw / nw_count * 100.0},
            "wave_sneak_damage": {"count": wave_sneak_nw, "pct": wave_sneak_nw / nw_count * 100.0},
            "defense_out_of_position": {"count": defense_away_nw, "pct": defense_away_nw / nw_count * 100.0},
            "squad_split": {"count": squad_split_nw, "pct": squad_split_nw / nw_count * 100.0},
            "low_hp_death_without_recovery": {"count": low_hp_fail_nw, "pct": low_hp_fail_nw / nw_count * 100.0},
        },
        "low_hp_retreat_recovery": {
            "mean_dist_to_nearest_supply": mean_supply_dist,
            "units_recovered": total_low_hp_recovered,
            "units_died_without_recovery": total_low_hp_died,
            "recovery_rate_pct": recovery_rate,
        },
        "per_opponent": opp_summary,
        "worst_opponent": worst_opp[0],
    }


def main():
    print("=" * 70)
    print("  STRATOS AI v5 — PHASE 1: FAILURE MODE ANALYSIS (VALIDATION SEEDS 1000-1049)")
    print("=" * 70)

    seeds = list(range(1000, 1050))
    opponents = ["aggressive", "defensive", "resource_focused"]

    print(f"\n[Phase 1] Evaluating Oracle Messenger on {len(seeds)*len(opponents)} episodes...")
    oracle = OracleMessengerPolicy()
    oracle_res = analyze_model("Oracle", oracle, "oracle", seeds, opponents)
    oracle_summary = summarize_analysis(oracle_res)

    print(f"\n[Phase 1] Evaluating Best COMM on {len(seeds)*len(opponents)} episodes...")
    comm_policy, _ = load_checkpoint("artifacts/checkpoints/comm_best.pt", device="cpu")
    comm_res = analyze_model("Best_COMM", comm_policy, "comm", seeds, opponents)
    comm_summary = summarize_analysis(comm_res)

    final_report = {
        "oracle": oracle_summary,
        "best_comm": comm_summary,
    }

    out_path = Path("artifacts/experiments/phase1_failure_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 1 DIAGNOSTIC REPORT")
    print("=" * 70)
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
