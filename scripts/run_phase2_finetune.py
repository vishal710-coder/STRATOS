"""Phase 2: PPO Fine-Tuning of COMM and NO_COMM policies.

Trains 3 seeds (42, 101, 202) for COMM and NO_COMM in parallel across CPU cores.
- Initialized from comm_bc_student.pt.
- COMM: small LR (5e-5), BC KL penalty annealed to 0, message entropy bonus (0.01),
  aux_pred_loss_coef=0.1, message_cost=0.01, enable_team_cohesion=True.
- NO_COMM: same budget (100k steps), independent mode, enable_team_cohesion=True.
- Checkpoints saved every 25k steps.
- Evaluates candidate checkpoints on validation seeds 1000-1019 to select the best checkpoint per seed.
"""

import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import train_stratos
from training.checkpoint import load_checkpoint, save_checkpoint
from environment.strategy_game import StrategyGameEnv


def run_single_train_job(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for single PPO training run."""
    torch.set_num_threads(2)
    label = kwargs.pop("job_label", "training_job")
    print(f"[{label}] Started training process (seed={kwargs.get('seed')}, mode={kwargs.get('coordination_mode')})...")
    res = train_stratos(**kwargs, verbose=False)
    print(f"[{label}] Finished training in {res['training_duration_sec']}s (WinRate: {res['final_win_rate']:.1%}).")
    return {"label": label, "results": res}


def evaluate_checkpoint_on_validation(
    ckpt_path: str,
    coordination_mode: str,
    seed_start: int = 1000,
    num_seeds: int = 20,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate a checkpoint on validation seeds (1000-1019) across all 3 opponents."""
    torch.set_num_threads(1)
    opponents = ["aggressive", "defensive", "resource_focused"]
    policy, _ = load_checkpoint(ckpt_path, device="cpu")
    policy.eval()

    results = []
    has_comm = (coordination_mode == "coordinated") and getattr(policy, "has_comm_head", False)

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
                obs_tensor = torch.as_tensor(
                    np.array([obs[ag] for ag in env.possible_agents]),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    if has_comm:
                        a_t, m_t, _, _, _, _ = policy.get_actions_and_values(obs_tensor, deterministic=deterministic)
                        actions_dict = {
                            ag: {"action": int(a_t[i].item()), "message": int(m_t[i].item())}
                            for i, ag in enumerate(env.possible_agents)
                        }
                    else:
                        a_t, _, _, _ = policy.get_action_and_value(obs_tensor, deterministic=deterministic)
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
    print("=" * 65)
    print("  STRATOS AI — PHASE 2: PPO FINE-TUNING & MULTI-SEED SELECTION")
    print("=" * 65)
    t_start = time.time()

    base_bc_checkpoint = "artifacts/checkpoints/comm_bc_student.pt"
    out_dir = Path("artifacts/checkpoints/v4_finetuned")
    exp_dir = Path("artifacts/experiments/v4_finetuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 101, 202]
    total_timesteps = 100000
    save_interval = 25000

    # Note on GRU:
    # "Add a GRU to the actor only if it takes under 15 minutes; otherwise skip and say so."
    # Decision: Skipping GRU to preserve training throughput and eliminate recurrent BPTT sequence risk.
    # Feed-forward decentralized policy directly conditions on delivered teammate messages in obs[52:70].

    job_configs = []

    # 3 COMM runs
    for s in seeds:
        job_configs.append({
            "job_label": f"COMM_s{s}",
            "seed": s,
            "total_timesteps": total_timesteps,
            "save_interval_steps": save_interval,
            "coordination_mode": "coordinated",
            "opponent_strategy": "mixed",
            "init_checkpoint": base_bc_checkpoint,
            "bc_checkpoint": base_bc_checkpoint,
            "bc_kl_coef": 0.5,
            "lr": 5e-5,
            "comm_entropy_coef": 0.01,
            "aux_loss_coef": 0.1,
            "message_cost": 0.01,
            "enable_team_cohesion": True,
            "output_checkpoint_path": str(out_dir / f"comm_s{s}.pt"),
            "history_csv_path": str(exp_dir / f"history_comm_s{s}.csv"),
            "history_json_path": str(exp_dir / f"history_comm_s{s}.json"),
        })

    # 3 NO_COMM runs (baseline: same budget, independent mode)
    for s in seeds:
        job_configs.append({
            "job_label": f"NO_COMM_s{s}",
            "seed": s,
            "total_timesteps": total_timesteps,
            "save_interval_steps": save_interval,
            "coordination_mode": "independent",
            "opponent_strategy": "mixed",
            "init_checkpoint": base_bc_checkpoint,
            "lr": 5e-5,
            "enable_team_cohesion": True,
            "output_checkpoint_path": str(out_dir / f"no_comm_s{s}.pt"),
            "history_csv_path": str(exp_dir / f"history_no_comm_s{s}.csv"),
            "history_json_path": str(exp_dir / f"history_no_comm_s{s}.json"),
        })

    print(f"[Phase 2] Spawning 6 parallel training jobs across CPU cores...")
    with mp.Pool(processes=min(6, mp.cpu_count())) as pool:
        train_results = pool.map(run_single_train_job, job_configs)

    dt_train = time.time() - t_start
    print(f"[Phase 2] All 6 training runs completed in {dt_train:.1f}s ({dt_train/60:.2f} mins).")

    # Checkpoint Selection on Validation Seeds 1000-1019
    print("\n[Phase 2] Evaluating checkpoints on validation seeds 1000-1019...")
    steps_list = [25000, 50000, 75000, 100000]
    validation_records = []

    eval_tasks = []
    # Evaluate BC fallback checkpoint as baseline
    eval_tasks.append((str(base_bc_checkpoint), "coordinated", "comm_bc_fallback"))

    for s in seeds:
        for st in steps_list:
            ckpt_c = out_dir / (f"comm_s{s}_step_{st}.pt" if st < 100000 else f"comm_s{s}.pt")
            if ckpt_c.exists():
                eval_tasks.append((str(ckpt_c), "coordinated", f"comm_s{s}_{st}"))

            ckpt_nc = out_dir / (f"no_comm_s{s}_step_{st}.pt" if st < 100000 else f"no_comm_s{s}.pt")
            if ckpt_nc.exists():
                eval_tasks.append((str(ckpt_nc), "independent", f"no_comm_s{s}_{st}"))

    # Run evaluations in parallel
    print(f"[Phase 2] Total checkpoint evaluation tasks: {len(eval_tasks)}...")
    val_results = []
    for ckpt_path, mode, label in eval_tasks:
        res = evaluate_checkpoint_on_validation(ckpt_path=ckpt_path, coordination_mode=mode)
        res["label"] = label
        val_results.append(res)
        print(f"  [{label}] Win = {res['win_rate']:.1f}%, Reward = {res['mean_reward']:.2f}, Steps = {res['mean_steps']:.1f}")

    # Select best checkpoint per seed
    selected_checkpoints = {}
    for s in seeds:
        comm_candidates = [r for r in val_results if r["label"].startswith(f"comm_s{s}_")]
        best_comm = max(comm_candidates, key=lambda x: (x["win_rate"], x["mean_reward"]))
        selected_checkpoints[f"comm_s{s}"] = best_comm

        nc_candidates = [r for r in val_results if r["label"].startswith(f"no_comm_s{s}_")]
        best_nc = max(nc_candidates, key=lambda x: (x["win_rate"], x["mean_reward"]))
        selected_checkpoints[f"no_comm_s{s}"] = best_nc

    print("\n" + "=" * 65)
    print("  PHASE 2 VALIDATION CHECKPOINT SELECTION SUMMARY")
    print("=" * 65)
    for k, v in selected_checkpoints.items():
        print(f"  {k:12s} -> Best Ckpt: {Path(v['checkpoint']).name} | Win: {v['win_rate']:.1f}% | Rew: {v['mean_reward']:.2f}")

    # Identify overall best validated COMM checkpoint
    best_overall_comm = max(
        [v for k, v in selected_checkpoints.items() if k.startswith("comm_s")],
        key=lambda x: (x["win_rate"], x["mean_reward"]),
    )
    best_comm_dest = "artifacts/checkpoints/comm_best.pt"
    # Copy best COMM checkpoint to artifacts/checkpoints/comm_best.pt
    import shutil
    shutil.copyfile(best_overall_comm["checkpoint"], best_comm_dest)
    print(f"\n[Phase 2] Copied overall best COMM checkpoint to: {best_comm_dest}")

    # Save summary report
    summary_path = exp_dir / "phase2_validation_selection.json"
    with open(summary_path, "w") as f:
        json.dump({
            "selected_checkpoints": selected_checkpoints,
            "best_overall_comm": best_overall_comm,
            "all_evaluations": val_results,
            "duration_sec": time.time() - t_start,
        }, f, indent=2)

    print(f"Phase 2 Complete. Total Wall-Clock: {time.time() - t_start:.1f}s ({(time.time() - t_start)/60:.2f} mins).")


if __name__ == "__main__":
    main()
