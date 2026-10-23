"""Vectorized Environment Wrapper for STRATOS AI (v2).

Provides high-throughput synchronous batched stepping across N parallel simulation instances.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from environment.strategy_game import StrategyGameEnv


class SyncVecEnv:
    """
    Synchronous Vectorized Multi-Agent Environment.
    Steps N independent StrategyGameEnv instances synchronously for fast batch rollout collection.
    """

    def __init__(
        self,
        num_envs: int = 4,
        config_path: str = "config/game.yaml",
        coordination_mode: str = "coordinated",
        opponent_strategy: str = "mixed",
        enable_communication: bool = True,
        include_role_id: bool = True,
    ):
        self.num_envs = num_envs
        self.config_path = config_path
        self.coordination_mode = coordination_mode
        self.opponent_strategy = opponent_strategy
        self.enable_communication = enable_communication
        self.include_role_id = include_role_id

        self.envs: List[StrategyGameEnv] = [
            StrategyGameEnv(
                config_path=config_path,
                coordination_mode=coordination_mode,
                opponent_strategy=opponent_strategy,
                enable_communication=enable_communication,
                include_role_id=include_role_id,
            )
            for _ in range(num_envs)
        ]

        self.possible_agents = self.envs[0].possible_agents
        self.agents = list(self.possible_agents)
        self.obs_dim = self.envs[0].obs_dim
        self.num_agents = len(self.possible_agents)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
        """Reset all environments with distinct per-env seeds."""
        all_obs = []
        all_infos = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i * 1000) if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            all_obs.append(obs)
            all_infos.append(info)
        return all_obs, all_infos

    def step(
        self,
        actions_list: List[Dict[str, Any]],
    ) -> Tuple[
        List[Dict[str, np.ndarray]],
        List[Dict[str, float]],
        List[Dict[str, bool]],
        List[Dict[str, bool]],
        List[Dict[str, Any]],
    ]:
        """
        Step all environments synchronously.
        Auto-resets any environment that reaches terminal or truncation.
        """
        assert len(actions_list) == self.num_envs, f"Expected {self.num_envs} action dicts, got {len(actions_list)}"

        obs_list = []
        rewards_list = []
        terms_list = []
        truncs_list = []
        infos_list = []

        for i, env in enumerate(self.envs):
            obs, rews, terms, truncs, info = env.step(actions_list[i])

            # Auto-reset if episode ended
            if any(terms.values()) or any(truncs.values()):
                terminal_obs = obs
                terminal_info = dict(info)
                terminal_info["terminal_observation"] = terminal_obs
                reset_obs, _ = env.reset()
                obs = reset_obs
                info = terminal_info

            obs_list.append(obs)
            rewards_list.append(rews)
            terms_list.append(terms)
            truncs_list.append(truncs)
            infos_list.append(info)

        return obs_list, rewards_list, terms_list, truncs_list, infos_list

    def close(self) -> None:
        """Clean up all environment instances."""
        pass
