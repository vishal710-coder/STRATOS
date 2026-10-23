"""Capture high-resolution battlefield visual frames for live demo presentation."""

import os
import sys
from pathlib import Path
import pygame
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.strategy_game import StrategyGameEnv
from visualization.battlefield import BattlefieldRenderer
from training.checkpoint import load_checkpoint

def capture_demo(output_dir: str, checkpoint_path: str, mode: str = "coordinated", opponent: str = "aggressive"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    env = StrategyGameEnv(coordination_mode=mode, opponent_strategy=opponent)
    policy, meta = load_checkpoint(checkpoint_path)

    renderer = BattlefieldRenderer(env=env, policy=policy, headless=True)
    obs, _ = env.reset(seed=42)

    key_steps = [1, 40, 90, 160, 230, 300]
    captured_files = []

    step = 0
    while step < 300:
        step += 1

        agent_obs = np.array([obs[agent] for agent in env.possible_agents])
        obs_t = torch.as_tensor(agent_obs, dtype=torch.float32)
        with torch.no_grad():
            if getattr(policy, "has_comm_head", False):
                action_t, msg_t, _, _, _, _ = policy.get_actions_and_values(obs_t, deterministic=True)
                actions = {
                    agent: {"action": int(action_t[i].item()), "message": int(msg_t[i].item())}
                    for i, agent in enumerate(env.possible_agents)
                }
            else:
                action_t, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
                actions = {agent: int(action_t[i].item()) for i, agent in enumerate(env.possible_agents)}

        obs, rewards, terminations, truncations, infos = env.step(actions)

        renderer.render_frame()

        if step in key_steps or any(terminations.values()):
            fname = f"demo_step_{step:03d}.png"
            file_dest = out_path / fname
            pygame.image.save(renderer.screen, str(file_dest))
            captured_files.append((step, str(file_dest)))
            print(f"Captured frame at step {step}: {file_dest}")

        if any(terminations.values()) or any(truncations.values()):
            break

    pygame.quit()
    print(f"Demo capture complete. {len(captured_files)} frames saved.")
    return captured_files

if __name__ == "__main__":
    out_dir = "artifacts/demo_frames"
    comm_ckpt = "artifacts/checkpoints/stratos_policy_comm_seed42.pt"
    latest_ckpt = "artifacts/checkpoints/stratos_policy_latest.pt"
    target_ckpt = comm_ckpt if Path(comm_ckpt).exists() else latest_ckpt
    capture_demo(out_dir, target_ckpt, mode="coordinated", opponent="aggressive")
