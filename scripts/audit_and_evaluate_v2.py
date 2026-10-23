"""Audited Research Evaluation and Statistical Analysis Suite for STRATOS AI (v2).

Implements rigorous, publication-grade evaluation:
1. Leakage-free validation checkpoint selection on validation seeds (1000..1019).
2. Large-sample held-out evaluation: 100 paired episodes per (condition x opponent x training seed)
   on fresh held-out seeds (5000..5099) -> 1,800 paired policy episodes + 600 baseline episodes.
3. Baselines: Random-action policy and Autonomous Heuristic policy evaluated on identical seeds.
4. Causal Ablation Suite: Intact vs Zeroed, Random, Temporally Shuffled, and Inter-Episode Swapped.
5. Cluster Bootstrap: Resamples training seeds (the true unit of replication) and episodes within seeds.
6. Token Distribution & Mutual Information Analysis: Computes empirical message frequencies,
   Shannon entropy, and mutual information with environmental events (enemies, low HP, hazards, resources, waves).
7. High-resolution publication-quality plots:
   - comm_vs_nocomm_performance.png (including baselines)
   - ablation_causality.png
   - token_distribution_and_mi.png
   - training_health_curves.png
"""

import argparse
import copy
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from environment.actions import ActionType
from agents.policies import ActorCriticPolicy
from training.checkpoint import load_checkpoint


# =====================================================================
#  Heuristic & Baseline Policies
# =====================================================================

class RandomPolicy:
    """Selects actions uniformly at random from {0..6}."""
    def select_action(self, agent: str, obs: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 7))


class ScriptedHeuristicPolicy:
    """
    Autonomous rule-based tactical policy operating purely on local observations:
    - Commander: Coordinates base defense, harvests crystals when low, attacks visible enemies.
    - Defense: Protects base perimeter, retreats when critically damaged, intercepts visible enemies.
    - Attack: Sieges enemy base, retreats when critical, eliminates visible threats.
    """
    def select_action(self, agent: str, obs: np.ndarray, rng: np.random.Generator) -> int:
        # Obs layout:
        # self status: 0..4 (x, y, hp/max_hp, cooldown, is_alive)
        # friendly base: 5..7 (x, y, hp/max_hp)
        # enemy base: 8..10 (x, y, hp/max_hp)
        # teammates: 11..18 (2 teammates * 4)
        # visible enemies: 19..30 (3 enemies * 4: rel_x, rel_y, hp, is_present)
        # visible resources: 31..39 (3 resources * 3)
        # global: 40..42 (friendly_res/500, enemy_res/500, step_ratio)
        # storm: 43..45 (rel_x, rel_y, is_active)

        self_hp_ratio = obs[2]
        friendly_base_hp = obs[7]
        friendly_res = obs[40] * 500.0

        # Check visible enemies
        enemy_visible = any(obs[19 + i * 4 + 3] > 0.5 for i in range(3))
        # Check visible resources
        res_visible = any(obs[31 + i * 3 + 2] > 0.05 for i in range(3))
        # Check active hazard storm nearby
        storm_near = obs[45] > 0.5

        if agent == "commander":
            if friendly_base_hp < 0.6 or (enemy_visible and friendly_base_hp < 0.9):
                return ActionType.DEFEND_PERIMETER
            if friendly_res < 60.0 and res_visible:
                return ActionType.MOVE_RESOURCE
            if enemy_visible:
                return ActionType.ATTACK_NEAREST_ENEMY
            return ActionType.MOVE_ENEMY_BASE

        elif agent == "defense":
            if self_hp_ratio < 0.25:
                return ActionType.TACTICAL_RETREAT
            if enemy_visible:
                return ActionType.ATTACK_NEAREST_ENEMY
            return ActionType.DEFEND_PERIMETER

        elif agent == "attack":
            if self_hp_ratio < 0.20:
                return ActionType.TACTICAL_RETREAT
            if enemy_visible:
                return ActionType.ATTACK_NEAREST_ENEMY
            return ActionType.MOVE_ENEMY_BASE

        return ActionType.HOLD


# =====================================================================
#  Core Episode Evaluation Function
# =====================================================================

def run_single_episode(
    env: StrategyGameEnv,
    ep_seed: int,
    policy: Optional[Any] = None,
    baseline_type: Optional[str] = None,  # "random", "heuristic", or None
    comm_ablation: Optional[str] = None,   # None, "zeroed", "random", "temporally_shuffled", "swapped"
    swapped_message_history: Optional[List[Dict[str, int]]] = None,
    collect_message_data: bool = False,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, int]]]:
    """
    Run one evaluation episode with complete metric and ablation support.
    Returns: (episode_summary, step_message_records, emitted_message_sequence)
    """
    rng = np.random.default_rng(ep_seed)
    obs, _ = env.reset(seed=ep_seed)

    step_count = 0
    total_team_reward = 0.0
    message_records: List[Dict[str, Any]] = []
    emitted_sequence: List[Dict[str, int]] = []
    past_episode_messages: List[Dict[str, int]] = []

    heuristic_bot = ScriptedHeuristicPolicy()
    random_bot = RandomPolicy()

    while True:
        step_count += 1
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
                    emitted_step = {}
                    for i, a in enumerate(env.possible_agents):
                        act = int(action_t[i].item())
                        m_val = int(msg_t[i].item())
                        emitted_step[a] = m_val

                        # Environmental context for message semantics analysis
                        if collect_message_data and env.state is not None:
                            unit = env.state.friendly_units.get(a)
                            enemy_vis = any(obs[a][19 + j * 4 + 3] > 0.5 for j in range(3))
                            low_hp = bool(unit.hp / unit.max_hp < 0.40) if unit else False
                            hazard_near = bool(obs[a][45] > 0.5)
                            res_found = any(obs[a][31 + j * 3 + 2] > 0.05 for j in range(3))
                            wave_active = bool(len(env.state.enemy_units) > 3)

                            message_records.append({
                                "agent_id": a,
                                "token": m_val,
                                "enemy_visible": int(enemy_vis),
                                "low_hp": int(low_hp),
                                "hazard_near": int(hazard_near),
                                "resource_found": int(res_found),
                                "wave_incoming": int(wave_active),
                            })

                        # Apply Ablation Injector
                        if comm_ablation == "zeroed":
                            m_val = 0
                        elif comm_ablation == "random":
                            m_val = int(rng.integers(0, env.msg_tokens))
                        elif comm_ablation == "temporally_shuffled":
                            if past_episode_messages:
                                rand_step = past_episode_messages[int(rng.integers(0, len(past_episode_messages)))]
                                m_val = rand_step.get(a, 0)
                            else:
                                m_val = 0
                        elif comm_ablation == "swapped":
                            if swapped_message_history and (step_count - 1) < len(swapped_message_history):
                                m_val = swapped_message_history[step_count - 1].get(a, 0)
                            else:
                                m_val = 0

                        action_dict[a] = {"action": act, "message": m_val}

                    emitted_sequence.append(emitted_step)
                    past_episode_messages.append(emitted_step)
                else:
                    action_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
                    for i, a in enumerate(env.possible_agents):
                        action_dict[a] = int(action_t[i].item())

        next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
        total_team_reward += float(sum(rewards.values()))
        obs = next_obs

        if any(terminations.values()) or any(truncations.values()):
            break

    state = env.state
    assert state is not None

    summary = {
        "winner": state.winner or "draw",
        "win_type": state.win_type or "unknown",
        "duration_steps": state.step_count,
        "team_reward": round(total_team_reward, 2),
        "friendly_base_final_hp": round(state.bases["friendly"].hp, 1),
        "enemy_base_final_hp": round(state.bases["enemy"].hp, 1),
        "friendly_units_survived": sum(1 for u in state.friendly_units.values() if u.is_alive),
        "enemy_units_eliminated": sum(1 for u in state.enemy_units.values() if not u.is_alive),
        "friendly_resources_collected": round(state.friendly_resources, 1),
        "messages_exchanged": len(env.coordinator.message_history),
    }

    return summary, message_records, emitted_sequence


# =====================================================================
#  Phase 1: Leakage-Free Validation Checkpoint Selection
# =====================================================================

def select_best_checkpoint(
    checkpoints: Dict[str, Dict[int, str]],
    val_seeds: List[int],
    opponents: List[str],
) -> Tuple[str, int, Dict[str, Any]]:
    """
    Selects the best COMM checkpoint using separate validation seeds (1000..1019).
    Guarantees held-out test evaluation has zero seed leakage.
    """
    print("\n=======================================================")
    print("  PHASE 1: Leakage-Free Validation Checkpoint Selection")
    print(f"  Validation Seeds: {len(val_seeds)} seeds ({val_seeds[0]}..{val_seeds[-1]})")
    print("=======================================================")

    results = {}
    for train_seed, ckpt_path in checkpoints["COMM"].items():
        policy, meta = load_checkpoint(ckpt_path)
        rewards = []
        wins = []
        for opp in opponents:
            env = StrategyGameEnv(coordination_mode="coordinated", opponent_strategy=opp, include_role_id=True)
            for s in val_seeds:
                rec, _, _ = run_single_episode(env=env, ep_seed=s, policy=policy)
                rewards.append(rec["team_reward"])
                wins.append(1 if rec["winner"] == "friendly" else 0)

        mean_rew = float(np.mean(rewards))
        win_rate = float(np.mean(wins))
        results[train_seed] = {
            "path": ckpt_path,
            "val_mean_reward": round(mean_rew, 2),
            "val_win_rate": round(win_rate, 4),
        }
        print(f"  Seed {train_seed:3d} Validation -> Mean Reward: {mean_rew:6.2f} | Win Rate: {win_rate:5.1%}")

    best_seed = max(results.keys(), key=lambda s: results[s]["val_mean_reward"])
    best_path = results[best_seed]["path"]
    print(f"\n  [Selection] Best COMM Checkpoint: Seed {best_seed} ({best_path})")

    # Copy / save alias as stratos_policy_comm_best.pt
    best_alias = Path("artifacts/checkpoints/stratos_policy_comm_best.pt")
    shutil.copyfile(best_path, best_alias)
    print(f"  Copied best checkpoint to: {best_alias}")

    return best_path, best_seed, results


# =====================================================================
#  Phase 2: Full Final Evaluation (100 Episodes per Cell)
# =====================================================================

def run_full_final_evaluation(
    checkpoints: Dict[str, Dict[int, str]],
    test_seeds: List[int],
    opponents: List[str],
    experiments_dir: Path,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[Tuple[int, str, int], List[Dict[str, int]]]]:
    """
    Evaluates:
    - COMM (seeds 42, 101, 202)
    - NO_COMM (seeds 42, 101, 202)
    - Random Baseline
    - Scripted Heuristic Baseline
    Across all opponents on 100 fresh held-out seeds.
    Total: 900 COMM + 900 NO_COMM + 300 Random + 300 Heuristic = 2,400 episodes.
    """
    print("\n=======================================================")
    print("  PHASE 2: Full Final Evaluation on Held-Out Test Seeds")
    print(f"  Test Seeds: {len(test_seeds)} fresh seeds ({test_seeds[0]}..{test_seeds[-1]})")
    print(f"  Opponents: {opponents}")
    print("=======================================================")

    all_records = []
    all_message_data = []
    emitted_histories: Dict[Tuple[int, str, int], List[Dict[str, int]]] = {}

    # Preload policies
    loaded_policies = {}
    for cond in ["COMM", "NO_COMM"]:
        loaded_policies[cond] = {}
        for s, path in checkpoints[cond].items():
            pol, _ = load_checkpoint(path)
            loaded_policies[cond][s] = pol

    # 1. Evaluate COMM and NO_COMM (Paired by test seed)
    for s_idx, train_seed in enumerate([42, 101, 202]):
        print(f"\n  [Evaluating Training Seed {train_seed} (COMM vs NO_COMM paired across {len(test_seeds)} seeds)]")
        comm_pol = loaded_policies["COMM"][train_seed]
        nocomm_pol = loaded_policies["NO_COMM"][train_seed]

        for opp in opponents:
            comm_env = StrategyGameEnv(coordination_mode="coordinated", opponent_strategy=opp, include_role_id=True)
            nocomm_env = StrategyGameEnv(coordination_mode="independent", opponent_strategy=opp, include_role_id=True)

            for seed_val in test_seeds:
                # COMM episode
                c_rec, c_msgs, c_seq = run_single_episode(
                    env=comm_env,
                    ep_seed=seed_val,
                    policy=comm_pol,
                    collect_message_data=True,
                )
                c_rec.update({
                    "condition": "COMM",
                    "train_seed": train_seed,
                    "eval_seed": seed_val,
                    "opponent_strategy": opp,
                })
                all_records.append(c_rec)
                all_message_data.extend(c_msgs)
                emitted_histories[(train_seed, opp, seed_val)] = c_seq

                # NO_COMM episode (Exact same env seed)
                nc_rec, _, _ = run_single_episode(
                    env=nocomm_env,
                    ep_seed=seed_val,
                    policy=nocomm_pol,
                )
                nc_rec.update({
                    "condition": "NO_COMM",
                    "train_seed": train_seed,
                    "eval_seed": seed_val,
                    "opponent_strategy": opp,
                })
                all_records.append(nc_rec)

    # 2. Evaluate Baselines: Random and Scripted Heuristic
    print(f"\n  [Evaluating Baselines: Random Policy & Scripted Heuristic across {len(test_seeds)} seeds]")
    for opp in opponents:
        b_env = StrategyGameEnv(coordination_mode="independent", opponent_strategy=opp, include_role_id=True)
        for seed_val in test_seeds:
            # Random baseline
            rnd_rec, _, _ = run_single_episode(
                env=b_env,
                ep_seed=seed_val,
                baseline_type="random",
            )
            rnd_rec.update({
                "condition": "RANDOM_BASELINE",
                "train_seed": 0,
                "eval_seed": seed_val,
                "opponent_strategy": opp,
            })
            all_records.append(rnd_rec)

            # Heuristic baseline
            heu_rec, _, _ = run_single_episode(
                env=b_env,
                ep_seed=seed_val,
                baseline_type="heuristic",
            )
            heu_rec.update({
                "condition": "SCRIPTED_HEURISTIC",
                "train_seed": 0,
                "eval_seed": seed_val,
                "opponent_strategy": opp,
            })
            all_records.append(heu_rec)

    df_eval = pd.DataFrame(all_records)
    out_csv = experiments_dir / "final_evaluation_100eps.csv"
    df_eval.to_csv(out_csv, index=False)
    print(f"\n  Final evaluation saved to: {out_csv} (Total records: {len(df_eval)})")

    return df_eval, all_message_data, emitted_histories


# =====================================================================
#  Phase 3: Comprehensive Ablation Suite on Full Evaluation Set
# =====================================================================

def run_full_ablation_suite(
    best_comm_ckpt: str,
    test_seeds: List[int],
    opponents: List[str],
    emitted_histories: Dict[Tuple[int, str, int], List[Dict[str, int]]],
    best_seed: int,
    experiments_dir: Path,
) -> pd.DataFrame:
    """
    Evaluates:
    1. intact_comm
    2. zeroed (all messages M=0)
    3. random (M ~ U{0..8})
    4. temporally_shuffled (messages sampled from past steps in episode)
    5. swapped_between_episodes (messages taken from a completely different episode)
    Evaluated across all 100 held-out test seeds and 3 opponents.
    """
    print("\n=======================================================")
    print("  PHASE 3: Comprehensive Causal Ablation Suite (100 eps)")
    print("=======================================================")

    policy, _ = load_checkpoint(best_comm_ckpt)
    ablations = ["intact", "zeroed", "random", "temporally_shuffled", "swapped"]
    ablation_records = []

    # Map of swapped seeds: ep seed i gets messages from seed (i+1)
    swapped_map = {test_seeds[i]: test_seeds[(i + 1) % len(test_seeds)] for i in range(len(test_seeds))}

    for abl in ablations:
        print(f"  Running ablation: {abl} across {len(test_seeds)} seeds...")
        for opp in opponents:
            env = StrategyGameEnv(coordination_mode="coordinated", opponent_strategy=opp, include_role_id=True)
            for s in test_seeds:
                donor_seed = swapped_map[s]
                swapped_hist = emitted_histories.get((best_seed, opp, donor_seed), None)

                rec, _, _ = run_single_episode(
                    env=env,
                    ep_seed=s,
                    policy=policy,
                    comm_ablation=None if abl == "intact" else abl,
                    swapped_message_history=swapped_hist,
                )
                rec["ablation_type"] = abl
                rec["eval_seed"] = s
                rec["opponent_strategy"] = opp
                rec["train_seed"] = best_seed
                ablation_records.append(rec)

    df_abl = pd.DataFrame(ablation_records)
    out_csv = experiments_dir / "ablation_study_100eps.csv"
    df_abl.to_csv(out_csv, index=False)
    print(f"  Ablation study saved to: {out_csv} (Total records: {len(df_abl)})")

    return df_abl


# =====================================================================
#  Phase 4: Message Semantics & Mutual Information Analysis
# =====================================================================

def analyze_message_semantics(
    message_records: List[Dict[str, Any]],
    experiments_dir: Path,
) -> Dict[str, Any]:
    """
    Computes:
    - Token usage frequency distribution per role (Commander, Defense, Attack)
    - Shannon entropy H(M) per agent role
    - Mutual information I(Token; Event) between tokens and 5 environmental events:
      enemy_visible, low_hp, hazard_near, resource_found, wave_incoming.
    """
    print("\n=======================================================")
    print("  PHASE 4: Message Analysis (Entropy & Mutual Info)")
    print(f"  Total Step Samples: {len(message_records):,}")
    print("=======================================================")

    df_msg = pd.DataFrame(message_records)
    roles = ["commander", "defense", "attack"]
    events = ["enemy_visible", "low_hp", "hazard_near", "resource_found", "wave_incoming"]
    num_tokens = 9

    analysis_res = {"roles": {}, "events": events}

    def calc_entropy(probs: np.ndarray) -> float:
        p = probs[probs > 0]
        return float(-np.sum(p * np.log2(p)))

    def calc_mi(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Mutual Information I(X; Y) in bits."""
        n = len(x)
        if n == 0:
            return 0.0
        # Joint distribution
        joint = defaultdict(float)
        p_x = defaultdict(float)
        p_y = defaultdict(float)
        for xi, yi in zip(x, y):
            joint[(xi, yi)] += 1.0 / n
            p_x[xi] += 1.0 / n
            p_y[yi] += 1.0 / n
        mi = 0.0
        for (xi, yi), pxy in joint.items():
            if pxy > 0:
                mi += pxy * math.log2(pxy / (p_x[xi] * p_y[yi]))
        return max(0.0, float(mi))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, role in enumerate(roles):
        sub = df_msg[df_msg["agent_id"] == role]
        counts = np.zeros(num_tokens)
        for t in range(num_tokens):
            counts[t] = (sub["token"] == t).sum()
        total_toks = max(1, len(sub))
        freqs = counts / total_toks
        entropy = calc_entropy(freqs)
        max_ent = math.log2(num_tokens)

        # Compute MI with each event
        mi_dict = {}
        for ev in events:
            ev_arr = sub[ev].to_numpy()
            tok_arr = sub["token"].to_numpy()
            mi_dict[ev] = round(calc_mi(tok_arr, ev_arr), 4)

        analysis_res["roles"][role] = {
            "token_frequencies": [round(float(f), 4) for f in freqs],
            "shannon_entropy_bits": round(entropy, 3),
            "max_possible_entropy": round(max_ent, 3),
            "entropy_ratio": round(entropy / max_ent, 3),
            "mutual_information_bits": mi_dict,
        }

        print(f"  [{role.upper()}] Shannon Entropy: {entropy:.3f} / {max_ent:.3f} bits (Ratio: {entropy/max_ent:.1%})")
        for ev, val in mi_dict.items():
            print(f"     -> I(Token; {ev:15s}): {val:.4f} bits")

        # Plot token distribution
        axes[idx].bar(range(num_tokens), freqs * 100, color="#0284c7")
        axes[idx].set_title(f"{role.capitalize()} Token Usage (H={entropy:.2f}b)", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Token ID (0=Silent, 1..8=Symbols)")
        axes[idx].set_ylabel("Frequency (%)")
        axes[idx].set_xticks(range(num_tokens))
        axes[idx].set_ylim(0, 100)

    plt.tight_layout()
    mi_plot_path = experiments_dir / "token_distribution_and_mi.png"
    plt.savefig(mi_plot_path, dpi=300)
    plt.close()
    print(f"  Token distribution plot saved to: {mi_plot_path}")

    out_json = experiments_dir / "message_analysis_semantics.json"
    with open(out_json, "w") as f:
        json.dump(analysis_res, f, indent=2)
    print(f"  Message semantics saved to: {out_json}")

    return analysis_res


# =====================================================================
#  Phase 5: Rigorous Cluster Bootstrap & Honest Statistics
# =====================================================================

def compute_cluster_bootstrap_statistics(
    df: pd.DataFrame,
    train_seeds: List[int],
    experiments_dir: Path,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """
    Computes rigorous statistical metrics treating the TRAINING SEED as the unit of replication.
    - Per-training-seed mean differences (COMM minus NO_COMM) for win rate, loss rate, and reward.
    - Cluster bootstrap: resamples training seeds with replacement, then resamples paired episodes within seed.
    - Separate tests for win rate, loss rate, and reward.
    """
    print("\n=======================================================")
    print("  PHASE 5: Cluster Bootstrap & Per-Seed Statistics     ")
    print(f"  Unit of Replication: TRAINING SEED (N={len(train_seeds)})")
    print("=======================================================")

    comm_df = df[df["condition"] == "COMM"]
    nocomm_df = df[df["condition"] == "NO_COMM"]

    per_seed_results = {}
    seed_diff_wins = []
    seed_diff_losses = []
    seed_diff_rewards = []

    for s in train_seeds:
        c_sub = comm_df[comm_df["train_seed"] == s]
        nc_sub = nocomm_df[nocomm_df["train_seed"] == s]

        c_win = float(c_sub["winner"].eq("friendly").mean())
        nc_win = float(nc_sub["winner"].eq("friendly").mean())
        d_win = c_win - nc_win

        c_loss = float(c_sub["winner"].eq("enemy").mean())
        nc_loss = float(nc_sub["winner"].eq("enemy").mean())
        d_loss = c_loss - nc_loss

        c_rew = float(c_sub["team_reward"].mean())
        nc_rew = float(nc_sub["team_reward"].mean())
        d_rew = c_rew - nc_rew

        seed_diff_wins.append(d_win)
        seed_diff_losses.append(d_loss)
        seed_diff_rewards.append(d_rew)

        per_seed_results[s] = {
            "comm_win_rate": round(c_win, 4),
            "nocomm_win_rate": round(nc_win, 4),
            "delta_win_rate": round(d_win, 4),
            "comm_loss_rate": round(c_loss, 4),
            "nocomm_loss_rate": round(nc_loss, 4),
            "delta_loss_rate": round(d_loss, 4),
            "comm_mean_reward": round(c_rew, 2),
            "nocomm_mean_reward": round(nc_rew, 2),
            "delta_mean_reward": round(d_rew, 2),
        }

        print(f"  Seed {s:3d} -> Delta Win: {d_win:+5.1%} | Delta Loss: {d_loss:+5.1%} | Delta Rew: {d_rew:+7.2f}")

    # Check sign consistency across all seeds
    win_consistent = all(d > 0 for d in seed_diff_wins) or all(d < 0 for d in seed_diff_wins)
    loss_consistent = all(d < 0 for d in seed_diff_losses) or all(d > 0 for d in seed_diff_losses)
    reward_consistent = all(d > 0 for d in seed_diff_rewards) or all(d < 0 for d in seed_diff_rewards)

    print(f"\n  Sign Consistency: Win Rate={win_consistent} | Loss Rate={loss_consistent} | Reward={reward_consistent}")

    # Cluster Bootstrap Resampling (resample seed, then episodes within seed)
    rng = np.random.default_rng(42)
    boot_delta_wins = []
    boot_delta_losses = []
    boot_delta_rewards = []

    # Pre-organize paired data per seed
    seed_paired_data = {}
    for s in train_seeds:
        c_sub = comm_df[comm_df["train_seed"] == s].sort_values(by=["opponent_strategy", "eval_seed"])
        nc_sub = nocomm_df[nocomm_df["train_seed"] == s].sort_values(by=["opponent_strategy", "eval_seed"])
        c_wins_arr = c_sub["winner"].eq("friendly").to_numpy(dtype=float)
        nc_wins_arr = nc_sub["winner"].eq("friendly").to_numpy(dtype=float)
        c_loss_arr = c_sub["winner"].eq("enemy").to_numpy(dtype=float)
        nc_loss_arr = nc_sub["winner"].eq("enemy").to_numpy(dtype=float)
        c_rew_arr = c_sub["team_reward"].to_numpy(dtype=float)
        nc_rew_arr = nc_sub["team_reward"].to_numpy(dtype=float)
        seed_paired_data[s] = (c_wins_arr, nc_wins_arr, c_loss_arr, nc_loss_arr, c_rew_arr, nc_rew_arr)

    for _ in range(n_boot):
        # 1. Sample seeds with replacement
        sampled_seeds = rng.choice(train_seeds, size=len(train_seeds), replace=True)
        rep_d_wins = []
        rep_d_losses = []
        rep_d_rews = []

        for s in sampled_seeds:
            c_w, nc_w, c_l, nc_l, c_r, nc_r = seed_paired_data[s]
            n_eps = len(c_w)
            ep_idx = rng.integers(0, n_eps, size=n_eps)
            rep_d_wins.append(np.mean(c_w[ep_idx] - nc_w[ep_idx]))
            rep_d_losses.append(np.mean(c_l[ep_idx] - nc_l[ep_idx]))
            rep_d_rews.append(np.mean(c_r[ep_idx] - nc_r[ep_idx]))

        boot_delta_wins.append(float(np.mean(rep_d_wins)))
        boot_delta_losses.append(float(np.mean(rep_d_losses)))
        boot_delta_rewards.append(float(np.mean(rep_d_rews)))

    ci_win = [round(float(np.percentile(boot_delta_wins, 2.5)), 4), round(float(np.percentile(boot_delta_wins, 97.5)), 4)]
    ci_loss = [round(float(np.percentile(boot_delta_losses, 2.5)), 4), round(float(np.percentile(boot_delta_losses, 97.5)), 4)]
    ci_reward = [round(float(np.percentile(boot_delta_rewards, 2.5)), 2), round(float(np.percentile(boot_delta_rewards, 97.5)), 2)]

    # Statistical significance at alpha = 0.05
    win_sig = bool(ci_win[0] > 0 or ci_win[1] < 0)
    loss_sig = bool(ci_loss[0] > 0 or ci_loss[1] < 0)
    reward_sig = bool(ci_reward[0] > 0 or ci_reward[1] < 0)

    summary = {
        "unit_of_replication": "training_seed",
        "num_training_seeds": len(train_seeds),
        "episodes_per_seed": len(comm_df) // len(train_seeds),
        "total_paired_episodes": len(comm_df),
        "comm_overall_win_rate": round(float(comm_df["winner"].eq("friendly").mean()), 4),
        "nocomm_overall_win_rate": round(float(nocomm_df["winner"].eq("friendly").mean()), 4),
        "overall_win_rate_delta": round(float(comm_df["winner"].eq("friendly").mean() - nocomm_df["winner"].eq("friendly").mean()), 4),
        "cluster_bootstrap_win_rate_ci_95": ci_win,
        "win_rate_significant_cluster_bootstrap": win_sig,
        "comm_overall_loss_rate": round(float(comm_df["winner"].eq("enemy").mean()), 4),
        "nocomm_overall_loss_rate": round(float(nocomm_df["winner"].eq("enemy").mean()), 4),
        "overall_loss_rate_delta": round(float(comm_df["winner"].eq("enemy").mean() - nocomm_df["winner"].eq("enemy").mean()), 4),
        "cluster_bootstrap_loss_rate_ci_95": ci_loss,
        "loss_rate_significant_cluster_bootstrap": loss_sig,
        "comm_overall_mean_reward": round(float(comm_df["team_reward"].mean()), 2),
        "nocomm_overall_mean_reward": round(float(nocomm_df["team_reward"].mean()), 2),
        "overall_mean_reward_delta": round(float(comm_df["team_reward"].mean() - nocomm_df["team_reward"].mean()), 2),
        "cluster_bootstrap_reward_ci_95": ci_reward,
        "reward_significant_cluster_bootstrap": reward_sig,
        "sign_consistency": {
            "win_rate_consistent_across_seeds": win_consistent,
            "loss_rate_consistent_across_seeds": loss_consistent,
            "reward_consistent_across_seeds": reward_consistent,
        },
        "per_seed_results": per_seed_results,
    }

    out_json = experiments_dir / "comm_vs_nocomm_statistical_report.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Statistical report saved to: {out_json}")

    return summary


# =====================================================================
#  Phase 6: High-Resolution Figures & Plots
# =====================================================================

def generate_audited_figures(
    eval_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    experiments_dir: Path,
) -> None:
    """Generates all publication plots including baselines and full ablations."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Performance Comparison Across Conditions and Baselines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    opponents = ["aggressive", "defensive", "resource_focused"]
    conditions = ["RANDOM_BASELINE", "SCRIPTED_HEURISTIC", "NO_COMM", "COMM"]
    labels = ["Random Action", "Scripted Heuristic", "NO_COMM (MARL)", "COMM (Learned)"]
    colors = ["#cbd5e1", "#f59e0b", "#94a3b8", "#0284c7"]

    x = np.arange(len(opponents))
    width = 0.20

    # Win Rates
    for i, (cond, lbl, col) in enumerate(zip(conditions, labels, colors)):
        rates = [
            eval_df[(eval_df["condition"] == cond) & (eval_df["opponent_strategy"] == opp)]["winner"].eq("friendly").mean() * 100
            for opp in opponents
        ]
        offset = (i - 1.5) * width
        ax1.bar(x + offset, rates, width, label=lbl, color=col)

    ax1.set_ylabel("Friendly Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Win Rate by Opponent Behavior", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([o.replace('_', ' ').capitalize() for o in opponents], fontsize=10)
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 50)

    # Mean Cumulative Team Reward
    for i, (cond, lbl, col) in enumerate(zip(conditions, labels, colors)):
        rewards = [
            eval_df[(eval_df["condition"] == cond) & (eval_df["opponent_strategy"] == opp)]["team_reward"].mean()
            for opp in opponents
        ]
        offset = (i - 1.5) * width
        ax2.bar(x + offset, rewards, width, label=lbl, color=col)

    ax2.set_ylabel("Mean Cumulative Team Reward", fontsize=11, fontweight="bold")
    ax2.set_title("Team Reward Across Opponents", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([o.replace('_', ' ').capitalize() for o in opponents], fontsize=10)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot1_path = experiments_dir / "comm_vs_nocomm_performance.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"  Performance plot saved to: {plot1_path}")

    # Figure 2: Causal Message Ablation Suite
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    abl_types = ["intact", "zeroed", "random", "temporally_shuffled", "swapped"]
    abl_labels = ["Intact COMM", "Zeroed (M=0)", "Random Tokens", "Time-Shuffled", "Cross-Ep Swapped"]
    abl_colors = ["#0284c7", "#64748b", "#f59e0b", "#8b5cf6", "#f43f5e"]

    # Reward across ablations
    abl_rewards = []
    abl_rew_errs = []
    abl_wins = []

    for abl in abl_types:
        sub = ablation_df[ablation_df["ablation_type"] == abl]
        r_mean = float(sub["team_reward"].mean())
        r_std = float(sub["team_reward"].std(ddof=1)) / math.sqrt(max(1, len(sub)))
        w_mean = float(sub["winner"].eq("friendly").mean()) * 100
        abl_rewards.append(r_mean)
        abl_rew_errs.append(r_std * 1.96)
        abl_wins.append(w_mean)

    ax1.bar(abl_labels, abl_rewards, yerr=abl_rew_errs, capsize=5, color=abl_colors)
    ax1.set_ylabel("Mean Team Reward", fontsize=11, fontweight="bold")
    ax1.set_title("Ablation Effect on Cumulative Reward", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(abl_labels)))
    ax1.set_xticklabels(abl_labels, rotation=15, ha="right")

    ax2.bar(abl_labels, abl_wins, color=abl_colors)
    ax2.set_ylabel("Win Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Ablation Effect on Win Rate", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(abl_labels)))
    ax2.set_xticklabels(abl_labels, rotation=15, ha="right")
    ax2.set_ylim(0, 30)

    plt.tight_layout()
    plot2_path = experiments_dir / "ablation_causality.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"  Ablation causality plot saved to: {plot2_path}")


# =====================================================================
#  Phase 7: Training Health Audit & Plot
# =====================================================================

def audit_training_health(experiments_dir: Path) -> Dict[str, Any]:
    """Inspects training history CSVs and plots learning dynamics."""
    print("\n=======================================================")
    print("  PHASE 7: Training Health Audit & Convergence Check   ")
    print("=======================================================")

    seeds = [42, 101, 202]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    health_summary = {}

    for s in seeds:
        for cond, col, style in [("comm", "#0284c7", "-"), ("nocomm", "#94a3b8", "--")]:
            fname = experiments_dir / f"training_history_{cond}_seed{s}.csv"
            if not fname.exists():
                continue
            df = pd.read_csv(fname)
            # Rolling reward
            df["rolling_reward"] = df["episode_reward"].rolling(window=10, min_periods=1).mean()

            ax1.plot(df["step"], df["rolling_reward"], label=f"{cond.upper()} Seed {s}", color=col, linestyle=style, alpha=0.8)

            # Checkpoint metrics
            ckpt_path = Path("artifacts/checkpoints") / f"stratos_policy_{cond}_seed{s}.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                m = ckpt.get("metrics", {})
                health_summary[f"{cond}_seed{s}"] = {
                    "total_steps": ckpt.get("total_steps", 0),
                    "final_entropy": round(float(m.get("entropy", 0.0)), 4),
                    "final_value_loss": round(float(m.get("value_loss", 0.0)), 4),
                    "final_policy_loss": round(float(m.get("policy_loss", 0.0)), 6),
                    "learning_rate": m.get("learning_rate", 0.0),
                }

    ax1.set_xlabel("Environment Steps", fontsize=11, fontweight="bold")
    ax1.set_ylabel("10-Episode Rolling Team Reward", fontsize=11, fontweight="bold")
    ax1.set_title("Training Reward Trajectories Across Seeds", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")

    # Bar plot of final entropy
    labels = list(health_summary.keys())
    entropies = [health_summary[k]["final_entropy"] for k in labels]
    colors = ["#0284c7" if "comm" in k else "#94a3b8" for k in labels]
    clean_labels = [k.replace('_', ' ').upper() for k in labels]

    ax2.bar(clean_labels, entropies, color=colors)
    ax2.set_ylabel("Final Policy Entropy (nats)", fontsize=11, fontweight="bold")
    ax2.set_title("Policy Entropy by Seed (Action + Comm Heads)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(clean_labels)))
    ax2.set_xticklabels(clean_labels, rotation=25, ha="right")

    plt.tight_layout()
    curve_plot = experiments_dir / "training_health_curves.png"
    plt.savefig(curve_plot, dpi=300)
    plt.close()
    print(f"  Training health curves saved to: {curve_plot}")

    out_json = experiments_dir / "training_health_summary.json"
    with open(out_json, "w") as f:
        json.dump(health_summary, f, indent=2)

    return health_summary


# =====================================================================
#  Main Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Full STRATOS v2 Audited Evaluation Suite")
    parser.add_argument("--episodes-per-cell", type=int, default=100, help="Episodes per (condition x opponent x seed) cell")
    args = parser.parse_args()

    experiments_dir = Path("artifacts/experiments")
    checkpoints_dir = Path("artifacts/checkpoints")
    experiments_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = [42, 101, 202]
    val_seeds = list(range(1000, 1020))  # 20 separate validation seeds (1000..1019)
    test_seeds = list(range(5000, 5000 + args.episodes_per_cell))  # 100 fresh held-out seeds (5000..5099)
    opponents = ["aggressive", "defensive", "resource_focused"]

    checkpoints = {
        "COMM": {s: str(checkpoints_dir / f"stratos_policy_comm_seed{s}.pt") for s in train_seeds},
        "NO_COMM": {s: str(checkpoints_dir / f"stratos_policy_nocomm_seed{s}.pt") for s in train_seeds},
    }

    # 1. Validation Checkpoint Selection (Leakage-Free)
    best_comm_ckpt, best_seed, val_results = select_best_checkpoint(
        checkpoints=checkpoints,
        val_seeds=val_seeds,
        opponents=opponents,
    )

    # 2. Full Final Evaluation on Held-Out Test Seeds (2,400 episodes)
    eval_df, message_records, emitted_histories = run_full_final_evaluation(
        checkpoints=checkpoints,
        test_seeds=test_seeds,
        opponents=opponents,
        experiments_dir=experiments_dir,
    )

    # 3. Full Ablation Suite (5 conditions x 3 opponents x 100 seeds = 1,500 episodes)
    ablation_df = run_full_ablation_suite(
        best_comm_ckpt=best_comm_ckpt,
        test_seeds=test_seeds,
        opponents=opponents,
        emitted_histories=emitted_histories,
        best_seed=best_seed,
        experiments_dir=experiments_dir,
    )

    # 4. Message Semantics and Mutual Information Analysis
    mi_analysis = analyze_message_semantics(
        message_records=message_records,
        experiments_dir=experiments_dir,
    )

    # 5. Cluster Bootstrap and Per-Seed Statistics
    stat_summary = compute_cluster_bootstrap_statistics(
        df=eval_df,
        train_seeds=train_seeds,
        experiments_dir=experiments_dir,
        n_boot=2000,
    )

    # 6. Training Health Audit
    health_summary = audit_training_health(experiments_dir)

    # 7. Generate High-Resolution Publication Figures
    generate_audited_figures(eval_df, ablation_df, experiments_dir)

    print("\n=======================================================")
    print("  AUDITED EVALUATION SUITE COMPLETE!                   ")
    print("=======================================================")
    print(f"Total Evaluated Episodes : {len(eval_df) + len(ablation_df):,}")
    print(f"COMM Overall Win Rate    : {stat_summary['comm_overall_win_rate']:.1%}")
    print(f"NO_COMM Win Rate         : {stat_summary['nocomm_overall_win_rate']:.1%}")
    print(f"Win Rate Delta           : {stat_summary['overall_win_rate_delta']:+.1%} (95% Cluster CI: {stat_summary['cluster_bootstrap_win_rate_ci_95']})")
    print(f"Loss Rate Delta          : {stat_summary['overall_loss_rate_delta']:+.1%} (95% Cluster CI: {stat_summary['cluster_bootstrap_loss_rate_ci_95']})")
    print(f"Reward Delta             : {stat_summary['overall_mean_reward_delta']:+.2f} (95% Cluster CI: {stat_summary['cluster_bootstrap_reward_ci_95']})")
    print(f"Win Rate Significant     : {stat_summary['win_rate_significant_cluster_bootstrap']}")
    print(f"Reward Significant       : {stat_summary['reward_significant_cluster_bootstrap']}")


if __name__ == "__main__":
    main()
