"""Phase 3: Scaled Long RL Training Runs for STRATOS v5.

Conditions (all running across CPU cores in parallel):
1. COMM_LONG (seeds 42, 101, 202):
   - Initialized from comm_bc_student.pt
   - lr: 1e-4 -> 1e-6 (linear decay)
   - bc_kl_coef: 0.5 annealed to 0 over the first 20% of steps
   - comm_entropy_coef: 0.01, action entropy_coef: 0.01
   - aux_loss_coef: 0.1
   - message_cost: 0.01
   - enable_team_cohesion: True
2. NO_COMM_LONG (seeds 42, 101, 202):
   - Initialized from no_comm_bc_student.pt (fair independent baseline)
   - independent coordination mode, no communication head
   - lr: 1e-4 -> 1e-6
   - bc_kl_coef: 0.5 annealed to 0 over first 20%
   - entropy_coef: 0.01
   - enable_team_cohesion: True
3. EMERGENT_LONG (seed 42):
   - Trained from scratch (random weights), no imitation, no aux loss
   - comm_entropy_coef: 0.01, entropy_coef: 0.01
   - message_cost: 0.01
   - enable_team_cohesion: True

Saves checkpoints every 100k steps up to 1,000,000 steps.
Post-training: evaluates candidates on validation seeds 1000-1049 (50 validation seeds x 3 opponents = 150 eps)
to select the best checkpoint per seed.
"""

import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import train_stratos
from training.checkpoint import load_checkpoint, save_checkpoint
from environment.strategy_game import StrategyGameEnv


def run_training_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    torch.set_num_threads(1)
    label = kwargs.pop("job_label", "worker")
    print(f"[{label}] Starting training job (seed={kwargs.get('seed')}, mode={kwargs.get('coordination_mode')})...")
    res = train_stratos(**kwargs, verbose=False)
    print(f"[{label}] Finished in {res['training_duration_sec']}s (Win Rate: {res['final_win_rate']:.1%}).")
    return {"label": label, "results": res}


def evaluate_checkpoint_on_validation(
    ckpt_path: str,
    coordination_mode: str,
    seed_start: int = 1000,
    num_seeds: int = 50,
) -> Dict[str, Any]:
    """Evaluate a candidate checkpoint on validation seeds (1000-1049) across all 3 opponents."""
    torch.set_num_threads(1)
    opponents = ["aggressive", "defensive", "resource_focused"]
    policy, _ = load_checkpoint(ckpt_path, device="cpu")
    policy.eval()

    has_comm = (coordination_mode == "coordinated") and getattr(policy, "has_comm_head", False)
    results = []

    for opp in opponents:
        for s_idx in range(num_seeds):
            seed = seed_start + s_idx
            env = StrategyGameEnv(
                opponent_strategy=opp,
                coordination_mode=coordination_mode,
                enable_communication=has_comm,
                include_role_id=True,
                enable_team_cohesion=True,
            )
            obs, _ = env.reset(seed=seed)
            done = False
            ep_reward = 0.0

            while not done:
                obs_t = torch.as_tensor(
                    np.array([obs[ag] for ag in env.possible_agents]),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    if has_comm:
                        a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=True)
                        actions_dict = {
                            ag: {"action": int(a_t[i].item()), "message": int(m_t[i].item())}
                            for i, ag in enumerate(env.possible_agents)
                        }
                    else:
                        a_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
                        actions_dict = {
                            ag: int(a_t[i].item())
                            for i, ag in enumerate(env.possible_agents)
                        }

                obs, rewards, term, trunc, _ = env.step(actions_dict)
                ep_reward += sum(rewards.values()) / len(rewards)
                done = any(term.values()) or any(trunc.values())

            winner = env.state.winner if env.state else "draw"
            results.append({
                "win": winner == "friendly",
                "loss": winner == "enemy",
                "draw": winner not in ("friendly", "enemy"),
                "reward": ep_reward,
                "steps": env.state.step_count if env.state else 300,
            })

    win_rate = float(np.mean([r["win"] for r in results]) * 100.0)
    draw_rate = float(np.mean([r["draw"] for r in results]) * 100.0)
    loss_rate = float(np.mean([r["loss"] for r in results]) * 100.0)
    mean_reward = float(np.mean([r["reward"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))

    return {
        "checkpoint": str(ckpt_path),
        "coordination_mode": coordination_mode,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
    }


def main():
    print("=" * 70)
    print("  STRATOS AI v5 — PHASE 3: SCALED LONG RL TRAINING RUNS")
    print("=" * 70)
    t_start = time.time()

    comm_bc_path = "artifacts/checkpoints/comm_bc_student.pt"
    no_comm_bc_path = "artifacts/checkpoints/no_comm_bc_student.pt"

    out_dir = Path("artifacts/checkpoints/v5_scaled")
    exp_dir = Path("artifacts/experiments/v5_scaled")
    best_dir = Path("artifacts/checkpoints/v5_best")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 101, 202]
    total_timesteps = 1000000  # 1.0M steps per run (7M total steps across 7 jobs)
    save_interval = 100000    # Checkpoint every 100k steps

    job_configs = []

    # 1. COMM_LONG (seeds 42, 101, 202)
    for s in seeds:
        job_configs.append({
            "job_label": f"COMM_LONG_s{s}",
            "seed": s,
            "total_timesteps": total_timesteps,
            "save_interval_steps": save_interval,
            "coordination_mode": "coordinated",
            "opponent_strategy": "mixed",
            "init_checkpoint": comm_bc_path,
            "bc_checkpoint": comm_bc_path,
            "bc_kl_coef": 0.5,
            "bc_kl_anneal_fraction": 0.20,
            "lr": 1e-4,
            "entropy_coef": 0.01,
            "comm_entropy_coef": 0.01,
            "aux_loss_coef": 0.1,
            "message_cost": 0.01,
            "enable_team_cohesion": True,
            "output_checkpoint_path": str(out_dir / f"comm_long_s{s}.pt"),
            "history_csv_path": str(exp_dir / f"history_comm_long_s{s}.csv"),
            "history_json_path": str(exp_dir / f"history_comm_long_s{s}.json"),
        })

    # 2. NO_COMM_LONG (seeds 42, 101, 202)
    for s in seeds:
        job_configs.append({
            "job_label": f"NO_COMM_LONG_s{s}",
            "seed": s,
            "total_timesteps": total_timesteps,
            "save_interval_steps": save_interval,
            "coordination_mode": "independent",
            "opponent_strategy": "mixed",
            "init_checkpoint": no_comm_bc_path,
            "bc_checkpoint": no_comm_bc_path,
            "bc_kl_coef": 0.5,
            "bc_kl_anneal_fraction": 0.20,
            "lr": 1e-4,
            "entropy_coef": 0.01,
            "message_cost": 0.0,
            "enable_team_cohesion": True,
            "output_checkpoint_path": str(out_dir / f"no_comm_long_s{s}.pt"),
            "history_csv_path": str(exp_dir / f"history_no_comm_long_s{s}.csv"),
            "history_json_path": str(exp_dir / f"history_no_comm_long_s{s}.json"),
        })

    # 3. EMERGENT_LONG (seed 42)
    job_configs.append({
        "job_label": "EMERGENT_LONG_s42",
        "seed": 42,
        "total_timesteps": total_timesteps,
        "save_interval_steps": save_interval,
        "coordination_mode": "coordinated",
        "opponent_strategy": "mixed",
        "init_checkpoint": None,
        "bc_checkpoint": None,
        "bc_kl_coef": 0.0,
        "lr": 1e-4,
        "entropy_coef": 0.01,
        "comm_entropy_coef": 0.01,
        "aux_loss_coef": 0.0,
        "message_cost": 0.01,
        "enable_team_cohesion": True,
        "output_checkpoint_path": str(out_dir / "emergent_long_s42.pt"),
        "history_csv_path": str(exp_dir / "history_emergent_long_s42.csv"),
        "history_json_path": str(exp_dir / "history_emergent_long_s42.json"),
    })

    print(f"[Phase 3] Launching {len(job_configs)} parallel training jobs across CPU workers...")
    pool_size = min(len(job_configs), mp.cpu_count())
    with mp.Pool(processes=pool_size) as pool:
        results = pool.map(run_training_worker, job_configs)

    t_train = time.time() - t_start
    print(f"\n[Phase 3] All training jobs completed in {t_train:.1f}s ({t_train/60:.2f} mins).")

    # Evaluate all saved checkpoints on validation seeds 1000-1049 (50 seeds x 3 opp = 150 eps)
    print("\n[Phase 3] Evaluating Candidate Checkpoints on Validation Seeds 1000-1049 (150 eps each)...")
    eval_candidates = []

    # Include BC fallbacks
    eval_candidates.append((comm_bc_path, "coordinated", "comm_bc_fallback"))
    eval_candidates.append((no_comm_bc_path, "independent", "no_comm_bc_fallback"))

    check_steps = [100000 * i for i in range(1, 11)]

    for s in seeds:
        for st in check_steps:
            c_p = out_dir / (f"comm_long_s{s}_step_{st}.pt" if st < total_timesteps else f"comm_long_s{s}.pt")
            if c_p.exists():
                eval_candidates.append((str(c_p), "coordinated", f"comm_long_s{s}_{st}"))

            nc_p = out_dir / (f"no_comm_long_s{s}_step_{st}.pt" if st < total_timesteps else f"no_comm_long_s{s}.pt")
            if nc_p.exists():
                eval_candidates.append((str(nc_p), "independent", f"no_comm_long_s{s}_{st}"))

    for st in check_steps:
        em_p = out_dir / (f"emergent_long_s42_step_{st}.pt" if st < total_timesteps else "emergent_long_s42.pt")
        if em_p.exists():
            eval_candidates.append((str(em_p), "coordinated", f"emergent_long_s42_{st}"))

    print(f"[Phase 3] Found {len(eval_candidates)} candidate checkpoints to evaluate...")
    val_records = []
    for ckpt_path, mode, label in eval_candidates:
        eval_res = evaluate_checkpoint_on_validation(ckpt_path, mode, seed_start=1000, num_seeds=50)
        eval_res["label"] = label
        val_records.append(eval_res)
        print(f"  [{label}] Win={eval_res['win_rate']:.1f}% | Draw={eval_res['draw_rate']:.1f}% | "
              f"Loss={eval_res['loss_rate']:.1f}% | Reward={eval_res['mean_reward']:.2f}")

    # Save validation records
    with open(exp_dir / "validation_selection_summary.json", "w") as f:
        json.dump(val_records, f, indent=2)

    # Select best checkpoints
    best_summary = {}
    for s in seeds:
        comm_group = [r for r in val_records if r["label"].startswith(f"comm_long_s{s}_")]
        if comm_group:
            best_comm = max(comm_group, key=lambda x: (x["win_rate"], x["mean_reward"]))
            best_summary[f"comm_long_s{s}"] = best_comm
            # Copy to best_dir
            src_p = Path(best_comm["checkpoint"])
            dst_p = best_dir / f"comm_long_s{s}_best.pt"
            dst_p.write_bytes(src_p.read_bytes())
            print(f"[Best] COMM Seed {s}: {best_comm['label']} (Win: {best_comm['win_rate']:.1f}%) -> {dst_p}")

        nc_group = [r for r in val_records if r["label"].startswith(f"no_comm_long_s{s}_")]
        if nc_group:
            best_nc = max(nc_group, key=lambda x: (x["win_rate"], x["mean_reward"]))
            best_summary[f"no_comm_long_s{s}"] = best_nc
            src_p = Path(best_nc["checkpoint"])
            dst_p = best_dir / f"no_comm_long_s{s}_best.pt"
            dst_p.write_bytes(src_p.read_bytes())
            print(f"[Best] NO_COMM Seed {s}: {best_nc['label']} (Win: {best_nc['win_rate']:.1f}%) -> {dst_p}")

    em_group = [r for r in val_records if r["label"].startswith("emergent_long_s42_")]
    if em_group:
        best_em = max(em_group, key=lambda x: (x["win_rate"], x["mean_reward"]))
        best_summary["emergent_long_s42"] = best_em
        src_p = Path(best_em["checkpoint"])
        dst_p = best_dir / "emergent_long_s42_best.pt"
        dst_p.write_bytes(src_p.read_bytes())
        print(f"[Best] EMERGENT Seed 42: {best_em['label']} (Win: {best_em['win_rate']:.1f}%) -> {dst_p}")

    with open(best_dir / "best_checkpoints_metadata.json", "w") as f:
        json.dump(best_summary, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[Phase 3 Complete] Total time: {total_time:.1f}s ({total_time/60:.2f} mins).")


if __name__ == "__main__":
    main()
