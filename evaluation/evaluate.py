"""Standalone policy evaluation module for STRATOS AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import yaml

from environment.strategy_game import StrategyGameEnv
from agents.policies import ActorCriticPolicy
from training.checkpoint import load_checkpoint
from evaluation.metrics import EpisodeMetric, aggregate_metrics, to_dataframe


def evaluate_policy(
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 10,
    coordination_mode: str = "coordinated",
    opponent_strategy: str = "aggressive",
    seed: int = 12345,
    game_config_path: str = "config/game.yaml",
    deterministic: bool = True,
    device: str = "cpu",
    comm_ablation: Optional[str] = None,
) -> Tuple[List[EpisodeMetric], Dict[str, Any]]:
    """
    Run clean, independent evaluation episodes on specified configuration.
    Supports ablations: comm_ablation in (None, 'zeroed', 'random', 'shuffled').
    """
    policy = None
    include_role_id = False
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        policy, meta = load_checkpoint(checkpoint_path, device=device)
        include_role_id = meta.get("obs_dim", 73) in (52, 67, 73) or meta.get("include_role_id", True)

    env = StrategyGameEnv(
        config_path=game_config_path,
        coordination_mode=coordination_mode,
        opponent_strategy=opponent_strategy,
        include_role_id=include_role_id,
    )

    records: List[EpisodeMetric] = []
    rng = np.random.RandomState(seed)

    for ep_idx in range(num_episodes):
        ep_seed = int(rng.randint(0, 1000000))
        obs, _ = env.reset(seed=ep_seed)

        step_count = 0
        total_team_reward = 0.0

        while True:
            step_count += 1
            action_dict = {}

            if policy is not None:
                agent_obs = np.array([obs[agent] for agent in env.possible_agents])
                obs_t = torch.as_tensor(agent_obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    if getattr(policy, "has_comm_head", False):
                        action_t, msg_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=deterministic)
                        for i, agent in enumerate(env.possible_agents):
                            m_val = int(msg_t[i].item())
                            if comm_ablation == "zeroed":
                                m_val = 0
                            elif comm_ablation == "random":
                                m_val = int(rng.randint(0, env.msg_tokens))
                            action_dict[agent] = {"action": int(action_t[i].item()), "message": m_val}
                        if comm_ablation == "shuffled":
                            m_vals = [action_dict[a]["message"] for a in env.possible_agents]
                            rng.shuffle(m_vals)
                            for i, a in enumerate(env.possible_agents):
                                action_dict[a]["message"] = m_vals[i]
                    else:
                        action_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=deterministic)
                        for i, agent in enumerate(env.possible_agents):
                            action_dict[agent] = int(action_t[i].item())
            else:
                # Heuristic / random baseline if no policy
                for agent in env.possible_agents:
                    action_dict[agent] = int(rng.randint(0, 7))

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            total_team_reward += float(sum(rewards.values()))
            obs = next_obs

            if any(terminations.values()) or any(truncations.values()):
                break

        state = env.state
        assert state is not None

        friendly_survived = sum(1 for u in state.friendly_units.values() if u.is_alive)
        enemy_killed = sum(1 for u in state.enemy_units.values() if not u.is_alive)
        messages_sent = len(env.coordinator.message_history)

        record = EpisodeMetric(
            episode_id=ep_idx + 1,
            scenario_name=f"{coordination_mode}_vs_{opponent_strategy}",
            coordination_mode=coordination_mode,
            opponent_strategy=opponent_strategy,
            winner=state.winner or "draw",
            duration_steps=state.step_count,
            team_reward=round(total_team_reward, 2),
            friendly_base_final_hp=round(state.bases["friendly"].hp, 1),
            enemy_base_final_hp=round(state.bases["enemy"].hp, 1),
            friendly_units_survived=friendly_survived,
            enemy_units_eliminated=enemy_killed,
            friendly_resources_collected=round(state.friendly_resources, 1),
            messages_exchanged=messages_sent,
            win_type=state.win_type or "unknown",
        )
        records.append(record)

    summary = aggregate_metrics(records)
    return records, summary
