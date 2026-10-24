"""Training pipeline execution for STRATOS AI."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import yaml

from environment.actions import NUM_ACTIONS
from environment.strategy_game import StrategyGameEnv
from agents.policies import ActorCriticPolicy
from training.rollout import MultiAgentRolloutBuffer
from training.ppo import PPOTrainer
from training.checkpoint import load_checkpoint, save_checkpoint


def train_stratos(
    config_path: str = "config/training.yaml",
    game_config_path: str = "config/game.yaml",
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    coordination_mode: str = "coordinated",
    opponent_strategy: str = "aggressive",
    resume_checkpoint: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    bc_checkpoint: Optional[str] = None,
    bc_kl_coef: float = 0.0,
    lr: Optional[float] = None,
    message_cost: Optional[float] = None,
    enable_team_cohesion: bool = False,
    output_checkpoint_path: Optional[str] = None,
    history_csv_path: Optional[str] = None,
    history_json_path: Optional[str] = None,
    include_role_id: bool = True,
    verbose: bool = True,
    aux_loss_coef: Optional[float] = None,
    comm_entropy_coef: Optional[float] = None,
    entropy_coef: Optional[float] = None,
    bc_kl_anneal_fraction: float = 0.20,
    save_interval_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute PPO training pipeline."""
    start_time = time.time()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg.get("training", {}).get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if total_timesteps is None:
        total_timesteps = int(cfg["training"]["total_timesteps"])

    rollout_steps = int(cfg["training"]["rollout_steps"])
    if save_interval_steps is not None:
        save_interval = int(save_interval_steps)
    else:
        save_interval = int(cfg["training"].get("save_interval_steps", 5000))
    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    experiments_dir = Path(cfg["paths"]["experiments_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Opponent strategy management
    opponents_pool = ["aggressive", "defensive", "resource_focused"]
    current_opponent = opponent_strategy
    if opponent_strategy == "mixed":
        current_opponent = opponents_pool[0]

    if comm_entropy_coef is None:
        comm_entropy_coef = float(cfg.get("hyperparameters", {}).get("comm_entropy_coef", cfg.get("hyperparameters", {}).get("entropy_coef", 0.01)))
    else:
        comm_entropy_coef = float(comm_entropy_coef)

    if aux_loss_coef is None:
        aux_loss_coef = float(cfg.get("hyperparameters", {}).get("aux_pred_loss_coef", 0.0))
    else:
        aux_loss_coef = float(aux_loss_coef)

    if message_cost is not None:
        msg_cost = float(message_cost)
    else:
        msg_cost = float(cfg.get("training", {}).get("message_cost", 0.0))

    chosen_lr = float(lr) if lr is not None else float(cfg["hyperparameters"]["learning_rate"])

    # Initialize Environment
    env = StrategyGameEnv(
        config_path=game_config_path,
        coordination_mode=coordination_mode,
        opponent_strategy=current_opponent,
        include_role_id=include_role_id,
        message_cost=msg_cost,
        enable_team_cohesion=enable_team_cohesion,
    )

    num_agents = len(env.possible_agents)
    obs_dim = env.obs_dim
    act_dim = NUM_ACTIONS
    has_comm_head = (coordination_mode == "coordinated")

    # Initialize Policy & PPO Trainer
    policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        msg_dim=env.msg_tokens,
        has_comm_head=has_comm_head,
        has_aux_head=(has_comm_head and (aux_loss_coef > 0.0)),
    ).to(device)

    # Optional BC policy for KL divergence penalty
    bc_pol = None
    if bc_checkpoint and Path(bc_checkpoint).exists():
        if verbose:
            print(f"[STRATOS Training] Loading BC anchor policy from: {bc_checkpoint}")
        bc_pol, _ = load_checkpoint(bc_checkpoint, device=str(device))
        bc_pol.eval()

    trainer = PPOTrainer(
        policy=policy,
        lr=chosen_lr,
        clip_eps=float(cfg["hyperparameters"]["clip_eps"]),
        value_coef=float(cfg["hyperparameters"]["value_loss_coef"]),
        entropy_coef=float(cfg["hyperparameters"]["entropy_coef"]),
        comm_entropy_coef=comm_entropy_coef,
        aux_loss_coef=aux_loss_coef,
        max_grad_norm=float(cfg["hyperparameters"]["max_grad_norm"]),
        bc_policy=bc_pol,
        bc_kl_coef=bc_kl_coef,
        device=str(device),
    )

    starting_step = 0
    starting_episodes = 0
    if init_checkpoint and Path(init_checkpoint).exists():
        if verbose:
            print(f"[STRATOS Training] Initializing policy weights from: {init_checkpoint}")
        policy, _ = load_checkpoint(init_checkpoint, policy=policy, device=str(device))
        policy.train()
    elif resume_checkpoint and Path(resume_checkpoint).exists():
        if verbose:
            print(f"[STRATOS Training] Resuming from checkpoint: {resume_checkpoint}")
        policy, meta = load_checkpoint(resume_checkpoint, policy=policy, optimizer=trainer.optimizer, device=str(device))
        policy.train()
        starting_step = meta.get("total_steps", 0)
        starting_episodes = meta.get("episode", 0)
        if verbose:
            print(f"[STRATOS Training] Loaded state from step {starting_step}, episode {starting_episodes}")

    buffer = MultiAgentRolloutBuffer(
        buffer_size=rollout_steps,
        num_agents=num_agents,
        obs_dim=obs_dim,
        device=device,
        gamma=float(cfg["hyperparameters"]["gamma"]),
        gae_lambda=float(cfg["hyperparameters"]["gae_lambda"]),
    )

    # Seed RNGs
    torch.manual_seed(seed)
    np.random.seed(seed)

    # State variables
    if opponent_strategy == "mixed":
        current_opponent = opponents_pool[starting_episodes % len(opponents_pool)]
    obs, _ = env.reset(seed=seed, options={"opponent_strategy": current_opponent})
    current_step = 0
    last_save_step = 0
    episode_count = 0
    episode_reward = 0.0
    episode_wins = 0

    history_records: List[Dict[str, Any]] = []
    latest_update_metrics: Dict[str, float] = {}

    if verbose:
        print(f"[STRATOS Training] Device: {device} | Additional Timesteps: {total_timesteps} | Base Step: {starting_step} | Mode: {coordination_mode} | CommHead: {has_comm_head} | Opponent: {opponent_strategy}")

    last_done_val = 0.0

    while current_step < total_timesteps:
        # Collect Rollout
        for step_idx in range(rollout_steps):
            current_step += 1

            # Prepare obs tensor [num_agents, obs_dim]
            agent_obs_list = [obs[agent] for agent in env.possible_agents]
            obs_tensor = torch.as_tensor(np.array(agent_obs_list), dtype=torch.float32, device=device)

            with torch.no_grad():
                if policy.has_comm_head:
                    action_tensor, msg_tensor, act_logp, msg_logp, _, value_tensor = (
                        policy.get_actions_and_values(obs_tensor)
                    )
                    logprob_tensor = act_logp + msg_logp
                    action_dict = {
                        agent: {
                            "action": int(action_tensor[i].item()),
                            "message": int(msg_tensor[i].item()),
                        }
                        for i, agent in enumerate(env.possible_agents)
                    }
                else:
                    action_tensor, logprob_tensor, _, value_tensor = policy.get_action_and_value(obs_tensor)
                    msg_tensor = None
                    action_dict = {
                        agent: int(action_tensor[i].item())
                        for i, agent in enumerate(env.possible_agents)
                    }

            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            reward_tensor = torch.as_tensor(
                [rewards[agent] for agent in env.possible_agents],
                dtype=torch.float32,
                device=device,
            )
            done_val = float(any(terminations.values()) or any(truncations.values()))
            last_done_val = done_val
            done_tensor = torch.full((num_agents,), done_val, dtype=torch.float32, device=device)

            hidden_fact = torch.as_tensor(
                [[env.state.weak_lane, env.state.wave_flank]] * num_agents,
                dtype=torch.long,
                device=device,
            ) if env.state else None

            # Store in buffer
            buffer.insert(
                obs=obs_tensor,
                action=action_tensor,
                logprob=logprob_tensor,
                reward=reward_tensor,
                done=done_tensor,
                value=value_tensor.squeeze(-1),
                message=msg_tensor,
                hidden_fact=hidden_fact,
            )

            episode_reward += float(sum(rewards.values()))
            obs = next_obs

            # Check episode completion
            if done_val > 0.5:
                episode_count += 1
                winner = env.state.winner if env.state else "unknown"
                win_type = env.state.win_type if env.state else "unknown"
                if winner == "friendly":
                    episode_wins += 1

                record = {
                    "step": starting_step + current_step,
                    "added_step": current_step,
                    "episode": starting_episodes + episode_count,
                    "opponent_strategy": current_opponent,
                    "episode_reward": round(episode_reward, 2),
                    "winner": winner,
                    "win_type": win_type,
                    "friendly_win_rate": round(episode_wins / max(1, episode_count), 3),
                    "friendly_base_hp": env.state.bases["friendly"].hp if env.state else 0,
                    "enemy_base_hp": env.state.bases["enemy"].hp if env.state else 0,
                    "resources_harvested": env.state.friendly_resources if env.state else 0,
                    "messages_sent": len(env.coordinator.message_history),
                }
                history_records.append(record)

                if verbose and (episode_count % 5 == 0 or episode_count == 1):
                    print(
                        f"Step {current_step}/{total_timesteps} (Global: {starting_step + current_step}) | Ep {starting_episodes + episode_count} | "
                        f"Opp: {current_opponent} | Reward: {episode_reward:.1f} | Winner: {winner} ({win_type}) | WinRate: {record['friendly_win_rate']:.2f}"
                    )

                episode_reward = 0.0
                if opponent_strategy == "mixed":
                    current_opponent = opponents_pool[(starting_episodes + episode_count) % len(opponents_pool)]
                obs, _ = env.reset(options={"opponent_strategy": current_opponent})

            if current_step >= total_timesteps:
                break

        # Compute GAE and Update Policy
        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(
                np.array([obs[agent] for agent in env.possible_agents]),
                dtype=torch.float32,
                device=device,
            )
            next_value = policy.get_value(next_obs_tensor).squeeze(-1)
            next_done = torch.tensor(
                [last_done_val] * num_agents,
                dtype=torch.float32,
                device=device,
            )

        buffer.compute_returns_and_advantages(next_value, next_done)

        # LR decay & Entropy annealing & BC KL penalty decay
        progress = min(1.0, current_step / max(1, total_timesteps))
        trainer.set_learning_rate(chosen_lr * max(0.01, 1.0 - progress))
        # Entropy: use provided fixed entropy_coef or default annealing
        if entropy_coef is not None:
            trainer.set_entropy_coef(float(entropy_coef))
        else:
            trainer.set_entropy_coef(0.02 * (1.0 - progress) + 0.005 * progress)
        if bc_pol is not None and bc_kl_coef > 0.0:
            anneal_decay = max(0.0, 1.0 - (progress / max(0.001, bc_kl_anneal_fraction)))
            trainer.set_bc_kl_coef(bc_kl_coef * anneal_decay)

        update_metrics = trainer.update(
            buffer=buffer,
            batch_size=int(cfg["hyperparameters"]["batch_size"]),
            ppo_epochs=int(cfg["hyperparameters"]["ppo_epochs"]),
        )
        latest_update_metrics = update_metrics

        if verbose:
            print(
                f"[PPO Update @ Step {starting_step + current_step}] Policy Loss: {update_metrics['policy_loss']:.5f} | "
                f"Value Loss: {update_metrics['value_loss']:.3f} | Entropy: {update_metrics['entropy']:.4f} | "
                f"Approx KL: {update_metrics['approx_kl']:.5f}"
            )

        # Periodic checkpoint
        if (current_step - last_save_step) >= save_interval and current_step < total_timesteps:
            last_save_step = (current_step // save_interval) * save_interval
            if output_checkpoint_path:
                out_p = Path(output_checkpoint_path)
                ckpt_path = out_p.parent / f"{out_p.stem}_step_{starting_step + last_save_step}{out_p.suffix}"
            else:
                ckpt_path = checkpoints_dir / f"stratos_policy_step_{starting_step + last_save_step}.pt"

            save_checkpoint(
                path=str(ckpt_path),
                policy=policy,
                optimizer=trainer.optimizer,
                total_steps=starting_step + current_step,
                episode=starting_episodes + episode_count,
                metrics=update_metrics,
                config=cfg,
            )

    # Save final checkpoint
    final_ckpt_path = Path(output_checkpoint_path) if output_checkpoint_path else (checkpoints_dir / "stratos_policy_latest.pt")
    save_checkpoint(
        path=str(final_ckpt_path),
        policy=policy,
        optimizer=trainer.optimizer,
        total_steps=starting_step + current_step,
        episode=starting_episodes + episode_count,
        metrics=latest_update_metrics,
        config=cfg,
    )

    # Save history logs
    out_csv = Path(history_csv_path) if history_csv_path else (experiments_dir / "training_history.csv")
    out_json = Path(history_json_path) if history_json_path else (experiments_dir / "training_history.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if history_records:
        df_history = pd.DataFrame(history_records)
        df_history.to_csv(out_csv, index=False)
        with open(out_json, "w") as f:
            json.dump(history_records, f, indent=2)

    duration = round(time.time() - start_time, 2)
    if verbose:
        print(f"[Training Complete] Duration: {duration}s | Final checkpoint saved to: {final_ckpt_path}")

    return {
        "final_checkpoint": str(final_ckpt_path),
        "total_episodes": starting_episodes + episode_count,
        "episodes_trained": episode_count,
        "total_steps": starting_step + current_step,
        "steps_trained": current_step,
        "final_win_rate": episode_wins / max(1, episode_count),
        "training_duration_sec": duration,
        "latest_metrics": latest_update_metrics,
        "history_records": history_records,
    }
