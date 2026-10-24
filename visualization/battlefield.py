"""Futuristic Pygame Interactive Battlefield Visualizer for STRATOS AI.

Includes full Presentation Demo Modes:
- [1] ORACLE Messenger Policy
- [2] SILENT Reference Policy
- [3] Learned COMM Policy (PPO)
- [4] Learned NO_COMM Policy (PPO)
- Key toggles: [1-4], [TAB] Cycle, [R] Replay Same Seed, [N] Next Seed, [O] Opponent
- Live Visuals: Winner banner with win type, message tokens & rays, weak lane scout indicator,
  incoming wave flank warning beacon, and session scoreboard.
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pygame
import numpy as np
import torch

from environment.strategy_game import StrategyGameEnv
from environment.actions import ACTION_NAMES, ActionType
from agents.policies import ActorCriticPolicy
from agents.commander import CommanderAgent
from agents.defense import DefenseAgent
from agents.attack import AttackAgent
from agents.reference_policies import OracleMessengerPolicy, SilentReferencePolicy
from training.checkpoint import load_checkpoint


# Palette Definitions
COLOR_BG = (10, 14, 24)
COLOR_GRID = (22, 30, 48)
COLOR_PANEL_BG = (16, 22, 36)
COLOR_PANEL_BORDER = (45, 62, 92)
COLOR_TEXT_MAIN = (225, 235, 250)
COLOR_TEXT_DIM = (130, 145, 175)
COLOR_TEXT_GOLD = (255, 215, 80)
COLOR_TEXT_CYAN = (80, 225, 255)
COLOR_TEXT_RED = (255, 85, 105)
COLOR_TEXT_GREEN = (85, 240, 140)
COLOR_TEXT_PURPLE = (200, 120, 255)

COLOR_FRIENDLY_BASE = (30, 140, 255)
COLOR_ENEMY_BASE = (240, 50, 75)
COLOR_RESOURCE_NODE = (50, 230, 210)

DEMO_MODES = ["oracle", "silent", "comm", "no_comm"]


def decode_token_label(token: int) -> str:
    """Translate discrete message token to human-readable tactical symbol."""
    if token == 0:
        return "Silent"
    elif token in (1, 2, 3):
        lane_names = ["Top Lane", "Mid Lane", "Bot Lane"]
        return f"Weak: {lane_names[token - 1]}"
    elif token in (4, 5, 6):
        flank_names = ["Top Flank", "Mid Flank", "Bot Flank"]
        return f"Flank: {flank_names[token - 4]}"
    elif token == 7:
        return "Push / Focus"
    elif token == 8:
        return "Defend / Regroup"
    return f"Token {token}"


class BattlefieldRenderer:
    """Renders real-time battlefield simulation with rich tactical HUD and presentation controls."""

    def __init__(
        self,
        env: StrategyGameEnv,
        policy: Optional[ActorCriticPolicy] = None,
        width: int = 1120,
        height: int = 700,
        headless: bool = False,
        demo_mode: str = "oracle",
        seed: int = 5000,
        comm_checkpoint: Optional[str] = None,
        nocomm_checkpoint: Optional[str] = None,
    ):
        self.env = env
        self.policy = policy
        self.width = width
        self.height = height
        self.map_w = env.map_w
        self.map_h = env.map_h
        self.headless = headless
        self.current_seed = seed
        self.demo_mode = demo_mode.lower() if demo_mode.lower() in DEMO_MODES else "oracle"

        # Checkpoint paths
        self.comm_ckpt_path = comm_checkpoint or self._resolve_best_checkpoint("comm")
        self.nocomm_ckpt_path = nocomm_checkpoint or self._resolve_best_checkpoint("no_comm")

        # Policies
        self.oracle_policy = OracleMessengerPolicy()
        self.silent_policy = SilentReferencePolicy()
        self.comm_policy: Optional[ActorCriticPolicy] = None
        self.nocomm_policy: Optional[ActorCriticPolicy] = None
        self.checkpoint_warnings: Dict[str, Optional[str]] = {"comm": None, "no_comm": None}

        self._load_demo_policies()

        self.heuristic_agents = {
            "commander": CommanderAgent(role="commander"),
            "defense": DefenseAgent(role="defense"),
            "attack": AttackAgent(role="attack"),
        }

        # Scoreboard (running tally per demo mode)
        self.scoreboard: Dict[str, Dict[str, Any]] = {
            m: {"wins": 0, "draws": 0, "losses": 0, "total": 0} for m in DEMO_MODES
        }

        # Last sent tokens per agent
        self.last_sent_tokens: Dict[str, int] = {"commander": 0, "defense": 0, "attack": 0}

        # Terminal state banner tracking
        self.match_finished = False
        self.terminal_winner: Optional[str] = None
        self.terminal_win_type: Optional[str] = None

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("STRATOS AI - Tactical Strategy Battlefield [EXPO DEMO]")

        # Fonts
        self.font_large = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 17, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 13)
        self.font_small = pygame.font.SysFont("Consolas", 11)
        self.font_tiny = pygame.font.SysFont("Consolas", 10)

        self.clock = pygame.time.Clock()
        self.fps = 30
        self.paused = False

    def _resolve_best_checkpoint(self, cond: str) -> Optional[str]:
        """Resolve best available checkpoint path for condition."""
        if cond == "comm":
            candidates = [
                "artifacts/checkpoints/comm_best.pt",
                "artifacts/checkpoints/comm_bc_student.pt",
                "artifacts/checkpoints/v4_finetuned/comm_s101.pt",
                "artifacts/checkpoints/v4_finetuned/comm_s42.pt",
                "artifacts/checkpoints/v4_finetuned/comm_s202_step_75000.pt",
                "artifacts/checkpoints/v3_resumed/comm_s101.pt",
                "artifacts/checkpoints/stratos_policy_comm_best.pt",
            ]
        else:
            candidates = [
                "artifacts/checkpoints/v4_finetuned/no_comm_s42_step_75000.pt",
                "artifacts/checkpoints/v4_finetuned/no_comm_s101_step_75000.pt",
                "artifacts/checkpoints/v4_finetuned/no_comm_s202_step_75000.pt",
                "artifacts/checkpoints/v4_finetuned/no_comm_s42.pt",
                "artifacts/checkpoints/v3_resumed/no_comm_s101.pt",
            ]
        for c in candidates:
            if Path(c).exists():
                return c
        return None

    def _load_demo_policies(self) -> None:
        """Load trained neural policies if checkpoints exist."""
        # Load COMM policy
        if self.comm_ckpt_path and Path(self.comm_ckpt_path).exists():
            try:
                p, meta = load_checkpoint(self.comm_ckpt_path, device="cpu")
                self.comm_policy = p
                self.checkpoint_warnings["comm"] = None
            except Exception as e:
                self.checkpoint_warnings["comm"] = f"Failed to load COMM checkpoint: {e}"
        else:
            self.checkpoint_warnings["comm"] = f"COMM checkpoint not found ({self.comm_ckpt_path or 'none'})"

        # Load NO_COMM policy
        if self.nocomm_ckpt_path and Path(self.nocomm_ckpt_path).exists():
            try:
                p, meta = load_checkpoint(self.nocomm_ckpt_path, device="cpu")
                self.nocomm_policy = p
                self.checkpoint_warnings["no_comm"] = None
            except Exception as e:
                self.checkpoint_warnings["no_comm"] = f"Failed to load NO_COMM checkpoint: {e}"
        else:
            self.checkpoint_warnings["no_comm"] = f"NO_COMM checkpoint not found ({self.nocomm_ckpt_path or 'none'})"

    def set_demo_mode(self, mode: str, reset_env: bool = True) -> None:
        """Switch demo mode and configure environment."""
        if mode not in DEMO_MODES:
            return
        self.demo_mode = mode
        if mode in ("oracle", "comm"):
            self.env.coordination_mode = "coordinated"
            self.env.enable_communication = True
            if hasattr(self.env, "coordinator"):
                self.env.coordinator.coordination_mode = "coordinated"
        else:
            self.env.coordination_mode = "independent"
            self.env.enable_communication = False
            if hasattr(self.env, "coordinator"):
                self.env.coordinator.coordination_mode = "independent"

        if reset_env:
            self.env.reset(seed=self.current_seed)
            if hasattr(self.oracle_policy, "reset"):
                self.oracle_policy.reset()
            self.last_sent_tokens = {"commander": 0, "defense": 0, "attack": 0}
            self.match_finished = False

    def select_actions(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Select actions according to the active presentation demo mode."""
        action_dict: Dict[str, Any] = {}

        if self.demo_mode == "oracle":
            for a in self.env.possible_agents:
                act, msg = self.oracle_policy.select_action_and_message(a, obs[a])
                action_dict[a] = {"action": int(act), "message": int(msg)}
                self.last_sent_tokens[a] = int(msg)

        elif self.demo_mode == "silent":
            for a in self.env.possible_agents:
                act, msg = self.silent_policy.select_action_and_message(a, obs[a])
                action_dict[a] = {"action": int(act), "message": int(msg)}
                self.last_sent_tokens[a] = 0

        elif self.demo_mode == "comm":
            if self.comm_policy is not None:
                obs_arr = np.array([obs[a] for a in self.env.possible_agents])
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
                with torch.no_grad():
                    a_t, m_t, _, _, _, _ = self.comm_policy.get_actions_and_values(obs_t, deterministic=True)
                    for i, a in enumerate(self.env.possible_agents):
                        act = int(a_t[i].item())
                        msg = int(m_t[i].item())
                        action_dict[a] = {"action": act, "message": msg}
                        self.last_sent_tokens[a] = msg
            else:
                for a in self.env.possible_agents:
                    action_dict[a] = {"action": int(self.heuristic_agents[a].select_action(obs[a])), "message": 0}
                    self.last_sent_tokens[a] = 0

        elif self.demo_mode == "no_comm":
            if self.nocomm_policy is not None:
                obs_arr = np.array([obs[a] for a in self.env.possible_agents])
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
                with torch.no_grad():
                    a_t, _, _, _ = self.nocomm_policy.get_action_and_value(obs_t, deterministic=True)
                    for i, a in enumerate(self.env.possible_agents):
                        action_dict[a] = int(a_t[i].item())
                        self.last_sent_tokens[a] = 0
            else:
                for a in self.env.possible_agents:
                    action_dict[a] = int(self.heuristic_agents[a].select_action(obs[a]))
                    self.last_sent_tokens[a] = 0

        return action_dict

    def record_match_result(self, winner: Optional[str], win_type: Optional[str]) -> None:
        """Update running scoreboard for the active demo mode."""
        self.match_finished = True
        self.terminal_winner = winner
        self.terminal_win_type = win_type
        sb = self.scoreboard[self.demo_mode]
        sb["total"] += 1
        if winner == "friendly":
            sb["wins"] += 1
        elif winner == "enemy":
            sb["losses"] += 1
        else:
            sb["draws"] += 1

    def render_frame(self) -> None:
        """Render complete simulation frame with all presentation HUD elements."""
        state = self.env.state
        if state is None:
            return

        self.screen.fill(COLOR_BG)

        # 1. Battlefield Arena (800 x 600, top=40)
        battle_rect = pygame.Rect(0, 40, int(self.map_w), int(self.map_h))
        pygame.draw.rect(self.screen, (13, 18, 30), battle_rect)

        # Subtle tactical grid
        for x in range(0, int(self.map_w), 50):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 40), (x, 40 + int(self.map_h)), 1)
        for y in range(40, 40 + int(self.map_h), 50):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (int(self.map_w), y), 1)

        # 2. Draw Lane Corridors & Generators
        self._draw_lane_corridors_and_generators(state)

        # 3. Draw Resource Nodes
        for node in state.resource_nodes:
            ny = int(node.y + 40)
            nx = int(node.x)
            radius = int(14 + (node.amount / node.max_amount) * 8)
            pygame.draw.circle(self.screen, (25, 70, 80), (nx, ny), radius + 4)
            pygame.draw.circle(self.screen, COLOR_RESOURCE_NODE, (nx, ny), radius)
            txt = self.font_small.render(f"RES:{int(node.amount)}", True, (10, 30, 30))
            self.screen.blit(txt, (nx - txt.get_width() // 2, ny - 6))

        # 4. Draw Bases
        for team, base in state.bases.items():
            bx = int(base.x)
            by = int(base.y + 40)
            color = COLOR_FRIENDLY_BASE if team == "friendly" else COLOR_ENEMY_BASE

            pygame.draw.circle(self.screen, (*color[:2], 120), (bx, by), 40, 2)
            pygame.draw.rect(self.screen, color, (bx - 26, by - 26, 52, 52), border_radius=8)

            hp_ratio = max(0.0, base.hp / base.max_hp)
            bar_w = 64
            pygame.draw.rect(self.screen, (40, 40, 40), (bx - 32, by - 38, bar_w, 6))
            pygame.draw.rect(self.screen, COLOR_TEXT_GREEN if hp_ratio > 0.4 else COLOR_TEXT_RED, (bx - 32, by - 38, int(bar_w * hp_ratio), 6))

            name_txt = self.font_small.render(f"{team.upper()} HQ ({int(base.hp)})", True, COLOR_TEXT_MAIN)
            self.screen.blit(name_txt, (bx - name_txt.get_width() // 2, by + 30))

        # 5. Draw Dynamic Events: Storm Hazard & Supply Drops
        if getattr(state, "storm", None) and state.storm.active:
            sx, sy = int(state.storm.x), int(state.storm.y + 40)
            s_rad = int(state.storm.radius)
            storm_surf = pygame.Surface((s_rad * 2 + 10, s_rad * 2 + 10), pygame.SRCALPHA)
            pygame.draw.circle(storm_surf, (220, 50, 80, 75), (s_rad + 5, s_rad + 5), s_rad)
            pygame.draw.circle(storm_surf, (255, 90, 110, 190), (s_rad + 5, s_rad + 5), s_rad, 2)
            self.screen.blit(storm_surf, (sx - s_rad - 5, sy - s_rad - 5))
            s_txt = self.font_small.render("! HAZARD STORM !", True, (255, 130, 150))
            self.screen.blit(s_txt, (sx - s_txt.get_width() // 2, sy - 8))

        if getattr(state, "supply_drop", None) and not state.supply_drop.collected:
            dx, dy = int(state.supply_drop.x), int(state.supply_drop.y + 40)
            pygame.draw.rect(self.screen, (30, 160, 100), (dx - 12, dy - 12, 24, 24), border_radius=4)
            pygame.draw.rect(self.screen, (255, 255, 255), (dx - 12, dy - 12, 24, 24), 2, border_radius=4)
            d_txt = self.font_small.render("+SUPPLY+", True, (60, 240, 160))
            self.screen.blit(d_txt, (dx - d_txt.get_width() // 2, dy + 14))

        # 6. Draw Units & Communicating Rays
        self._draw_units_and_messages(state)

        # 7. Draw Flank Alert Warning Beacon
        self._draw_flank_warning(state)

        # 8. Draw Winner Banner across screen if match concluded
        if state.winner is not None or self.match_finished:
            self._draw_winner_banner(state)

        # 9. Top Navigation & Demo Mode Control Bar
        self._draw_top_bar(state)

        # 10. Right Sidebar: Telemetry & Scoreboard
        self._draw_sidebar(state)

        # 11. Bottom Ticker: Decoded Live Message Stream
        self._draw_bottom_ticker(state)

        # 12. Checkpoint warning modal banner if missing
        active_warning = self.checkpoint_warnings.get(self.demo_mode)
        if active_warning:
            self._draw_checkpoint_missing_banner(active_warning)

        pygame.display.flip()

    def _draw_lane_corridors_and_generators(self, state: Any) -> None:
        """Render 3 tactical lane paths, weak lane highlight, and armored generator shields."""
        lanes = [
            (0, "TOP LANE", 120.0),
            (1, "MID LANE", 300.0),
            (2, "BOT LANE", 480.0),
        ]
        weak_lane = getattr(state, "weak_lane", 1)

        for l_idx, l_name, l_y in lanes:
            gy = int(l_y + 40)

            # Draw dashed lane corridor line across battlefield
            for seg_x in range(120, 680, 24):
                pygame.draw.line(self.screen, (28, 38, 58), (seg_x, gy), (seg_x + 12, gy), 1)

            # Lane identifier text
            l_txt = self.font_tiny.render(l_name, True, (70, 85, 115))
            self.screen.blit(l_txt, (140, gy - 14))

            # Generator objective at (700, gy)
            gx = 700
            is_weak = (l_idx == weak_lane)

            if is_weak:
                # Vulnerable weak lane generator (4.0x bonus damage)
                pulse = int(math.sin(pygame.time.get_ticks() * 0.006) * 4)
                pygame.draw.circle(self.screen, (255, 180, 40, 80), (gx, gy), 22 + pulse, 2)
                pygame.draw.circle(self.screen, COLOR_TEXT_GOLD, (gx, gy), 16)
                pygame.draw.circle(self.screen, (30, 20, 10), (gx, gy), 8)

                # Prominent weak lane label
                badge_bg = pygame.Rect(gx - 170, gy - 28, 160, 20)
                pygame.draw.rect(self.screen, (40, 30, 10), badge_bg, border_radius=4)
                pygame.draw.rect(self.screen, COLOR_TEXT_GOLD, badge_bg, 1, border_radius=4)
                lbl = self.font_small.render("★ WEAK LANE (4.0x Vuln)", True, COLOR_TEXT_GOLD)
                self.screen.blit(lbl, (gx - 166, gy - 26))

                scout_note = self.font_tiny.render("[Scout Obs: obs[46:49]]", True, (240, 200, 120))
                self.screen.blit(scout_note, (gx - 166, gy - 10))
            else:
                # Armored generator (0.1x damage shield)
                pygame.draw.circle(self.screen, (40, 70, 110), (gx, gy), 16)
                pygame.draw.circle(self.screen, (80, 140, 210), (gx, gy), 16, 2)
                lbl = self.font_tiny.render("0.1x ARMORED", True, (110, 145, 185))
                self.screen.blit(lbl, (gx - 90, gy - 7))

    def _draw_flank_warning(self, state: Any) -> None:
        """Render flashing warning beacon at the enemy reinforcement flank corridor."""
        warning_active = getattr(state, "wave_warning_active", False)
        flank_idx = getattr(state, "wave_flank", 1)
        flank_y_map = {0: 120.0, 1: 300.0, 2: 480.0}
        fy = int(flank_y_map.get(flank_idx, 300.0) + 40)
        fx = 765

        if warning_active:
            flash = (pygame.time.get_ticks() // 250) % 2 == 0
            color = COLOR_TEXT_RED if flash else (160, 30, 45)

            # Warning beacon
            pygame.draw.polygon(self.screen, color, [(fx, fy - 14), (fx + 14, fy + 12), (fx - 14, fy + 12)])
            excl = self.font_small.render("!", True, (255, 255, 255))
            self.screen.blit(excl, (fx - excl.get_width() // 2, fy - 5))

            alert_box = pygame.Rect(fx - 240, fy + 16, 250, 28)
            pygame.draw.rect(self.screen, (45, 15, 20), alert_box, border_radius=4)
            pygame.draw.rect(self.screen, COLOR_TEXT_RED, alert_box, 1, border_radius=4)
            t1 = self.font_small.render("⚠ FLANK WARNING: INCOMING WAVE", True, COLOR_TEXT_RED)
            t2 = self.font_tiny.render("[Visible ONLY to Defense obs[49:52]]", True, (240, 150, 160))
            self.screen.blit(t1, (fx - 235, fy + 18))
            self.screen.blit(t2, (fx - 235, fy + 30))

    def _draw_units_and_messages(self, state: Any) -> None:
        """Render friendly and enemy units, vision circles, and message transmission links."""
        role_colors = {
            "commander": COLOR_TEXT_GOLD,
            "defense": COLOR_TEXT_GREEN,
            "attack": COLOR_TEXT_PURPLE,
        }

        # 1. Draw Message Transmission Links between communicating teammates
        if self.env.enable_communication and self.demo_mode in ("oracle", "comm"):
            for sender_id, token in self.last_sent_tokens.items():
                if token > 0:
                    su = state.friendly_units.get(sender_id)
                    if su and su.is_alive:
                        for recip_id, ru in state.friendly_units.items():
                            if recip_id != sender_id and ru.is_alive:
                                # Dotted transmission line
                                sx, sy = int(su.x), int(su.y + 40)
                                rx, ry = int(ru.x), int(ru.y + 40)
                                pygame.draw.line(self.screen, (100, 180, 255, 120), (sx, sy), (rx, ry), 1)

                                # Midpoint token pill
                                mx = (sx + rx) // 2
                                my = (sy + ry) // 2
                                pygame.draw.circle(self.screen, (25, 38, 58), (mx, my), 10)
                                pygame.draw.circle(self.screen, COLOR_TEXT_CYAN, (mx, my), 10, 1)
                                ts = self.font_tiny.render(f"T{token}", True, COLOR_TEXT_CYAN)
                                self.screen.blit(ts, (mx - ts.get_width() // 2, my - 5))

        # 2. Draw Friendly units
        for u in state.friendly_units.values():
            if u.is_alive:
                self._draw_single_unit(u, role_colors.get(u.role, (100, 200, 255)), is_friendly=True)

        # 3. Draw Enemy units
        for u in state.enemy_units.values():
            if u.is_alive:
                self._draw_single_unit(u, COLOR_TEXT_RED, is_friendly=False)

    def _draw_single_unit(self, unit: Any, color: tuple, is_friendly: bool) -> None:
        """Render individual battlefield unit."""
        ux = int(unit.x)
        uy = int(unit.y + 40)

        # Attack range indicator
        pygame.draw.circle(self.screen, (*color[:2], 30), (ux, uy), int(unit.attack_range), 1)

        # Body circle
        radius = 12 if is_friendly else 10
        pygame.draw.circle(self.screen, color, (ux, uy), radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (ux, uy), radius, 2)

        # Health bar
        hp_ratio = max(0.0, unit.hp / unit.max_hp)
        bar_w = 28
        pygame.draw.rect(self.screen, (30, 30, 30), (ux - 14, uy - 18, bar_w, 4))
        pygame.draw.rect(self.screen, COLOR_TEXT_GREEN if hp_ratio > 0.3 else COLOR_TEXT_RED, (ux - 14, uy - 18, int(bar_w * hp_ratio), 4))

        # Label & Role
        lbl = unit.name[:3].upper() if is_friendly else "ENM"
        txt = self.font_small.render(lbl, True, COLOR_TEXT_MAIN)
        self.screen.blit(txt, (ux - txt.get_width() // 2, uy + 12))

        # If friendly: render transmitted token badge above head
        if is_friendly:
            token = self.last_sent_tokens.get(unit.role, 0)
            if token > 0:
                t_badge = pygame.Rect(ux - 22, uy - 32, 44, 12)
                pygame.draw.rect(self.screen, (30, 45, 70), t_badge, border_radius=3)
                pygame.draw.rect(self.screen, COLOR_TEXT_GOLD, t_badge, 1, border_radius=3)
                t_lbl = self.font_tiny.render(f"TX: T{token}", True, COLOR_TEXT_GOLD)
                self.screen.blit(t_lbl, (ux - t_lbl.get_width() // 2, uy - 32))

    def _draw_winner_banner(self, state: Any) -> None:
        """Draw prominent victory / draw / defeat banner across top of battlefield."""
        winner = self.terminal_winner or state.winner
        win_type = self.terminal_win_type or state.win_type

        is_win = (winner == "friendly")
        is_loss = (winner == "enemy")
        is_draw = (winner is None or winner == "draw")

        if is_win:
            theme_color = COLOR_TEXT_GREEN
            bg_color = (15, 45, 25)
            title = f"★ TACTICAL VICTORY (FRIENDLY) // {str(win_type).upper()} ★"
            sub = "Coordinated Squad Destroyed Enemy Objective"
        elif is_draw:
            theme_color = COLOR_TEXT_GOLD
            bg_color = (45, 38, 15)
            title = "⚖ TACTICAL DRAW // 300 STEPS TIMEOUT (NO HOARDING SUBSIDY) ⚖"
            sub = "Passivity Penalized: Draw Penalty (-30/agent) + Step Pressure Applied"
        else:
            theme_color = COLOR_TEXT_RED
            bg_color = (45, 15, 20)
            title = f"✕ TACTICAL DEFEAT // {str(win_type).upper()} ✕"
            sub = "Friendly Headquarters Overrun by Hostile Swarm"

        banner_rect = pygame.Rect(40, 52, int(self.map_w) - 80, 54)
        pygame.draw.rect(self.screen, bg_color, banner_rect, border_radius=8)
        pygame.draw.rect(self.screen, theme_color, banner_rect, 2, border_radius=8)

        t_surf = self.font_title.render(title, True, theme_color)
        s_surf = self.font_small.render(sub, True, COLOR_TEXT_MAIN)
        self.screen.blit(t_surf, (banner_rect.centerx - t_surf.get_width() // 2, banner_rect.y + 8))
        self.screen.blit(s_surf, (banner_rect.centerx - s_surf.get_width() // 2, banner_rect.y + 30))

    def _draw_top_bar(self, state: Any) -> None:
        """Top navigation and mode selection bar."""
        header_rect = pygame.Rect(0, 0, self.width, 40)
        pygame.draw.rect(self.screen, (10, 14, 22), header_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (0, 40), (self.width, 40), 2)

        title = self.font_title.render("STRATOS AI //", True, COLOR_TEXT_CYAN)
        self.screen.blit(title, (12, 10))

        # Mode Buttons / Indicators
        mode_configs = [
            ("oracle", "[1] ORACLE", COLOR_TEXT_GOLD),
            ("silent", "[2] SILENT", (160, 180, 200)),
            ("comm", "[3] COMM (PPO)", COLOR_TEXT_CYAN),
            ("no_comm", "[4] NO_COMM (PPO)", (200, 140, 100)),
        ]
        x_btn = 145
        for m_key, m_label, m_col in mode_configs:
            is_active = (self.demo_mode == m_key)
            btn_w = 110
            btn_rect = pygame.Rect(x_btn, 7, btn_w, 26)

            if is_active:
                pygame.draw.rect(self.screen, (35, 48, 72), btn_rect, border_radius=4)
                pygame.draw.rect(self.screen, m_col, btn_rect, 2, border_radius=4)
                lbl = self.font_small.render(m_label, True, m_col)
            else:
                pygame.draw.rect(self.screen, (18, 24, 36), btn_rect, border_radius=4)
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, btn_rect, 1, border_radius=4)
                lbl = self.font_small.render(m_label, True, COLOR_TEXT_DIM)

            self.screen.blit(lbl, (btn_rect.centerx - lbl.get_width() // 2, btn_rect.y + 6))
            x_btn += btn_w + 8

        # Episode Seed & Controls
        seed_txt = self.font_small.render(f"SEED: {self.current_seed} ([R]eplay | [N]ext)", True, COLOR_TEXT_GOLD)
        self.screen.blit(seed_txt, (x_btn + 10, 12))

        # Opponent & Step Counter
        step_txt = self.font_small.render(f"STEP: {state.step_count}/{state.max_steps}", True, COLOR_TEXT_MAIN)
        self.screen.blit(step_txt, (self.width - 240, 12))

        opp_txt = self.font_small.render(f"OPP: {self.env.opponent_strategy[:4].upper()} ([O])", True, COLOR_TEXT_RED)
        self.screen.blit(opp_txt, (self.width - 120, 12))

    def _draw_sidebar(self, state: Any) -> None:
        """Right telemetry, scoreboard, and match status sidebar."""
        panel_x = int(self.map_w)
        panel_w = self.width - panel_x
        panel_rect = pygame.Rect(panel_x, 40, panel_w, int(self.map_h))
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (panel_x, 40), (panel_x, 40 + int(self.map_h)), 2)

        y = 50

        # 1. Mode Status Card
        mode_box = pygame.Rect(panel_x + 12, y, panel_w - 24, 52)
        pygame.draw.rect(self.screen, (22, 30, 48), mode_box, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_TEXT_CYAN, mode_box, 1, border_radius=6)

        m_name = self.demo_mode.upper()
        if self.demo_mode == "oracle":
            m_sub = "Oracle Messenger (Benchmark 54.7% Win)"
        elif self.demo_mode == "silent":
            m_sub = "Silent Reference (Benchmark 22.7% Win)"
        elif self.demo_mode == "comm":
            m_sub = "Learned PPO Coordinated Policy"
        else:
            m_sub = "Learned PPO Independent Policy"

        self.screen.blit(self.font_title.render(f"MODE: {m_name}", True, COLOR_TEXT_CYAN), (panel_x + 20, y + 6))
        self.screen.blit(self.font_tiny.render(m_sub, True, COLOR_TEXT_DIM), (panel_x + 20, y + 30))
        y += 62

        # 2. Small Session Scoreboard
        sb_box = pygame.Rect(panel_x + 12, y, panel_w - 24, 112)
        pygame.draw.rect(self.screen, (20, 26, 42), sb_box, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, sb_box, 1, border_radius=6)

        self.screen.blit(self.font_body.render("SCOREBOARD (Session Matches)", True, COLOR_TEXT_GOLD), (panel_x + 20, y + 6))
        sb_y = y + 26

        labels = [
            ("oracle", "ORACLE  "),
            ("silent", "SILENT  "),
            ("comm", "COMM    "),
            ("no_comm", "NO_COMM "),
        ]
        for m_key, m_tag in labels:
            data = self.scoreboard[m_key]
            tot = data["total"]
            w = data["wins"]
            d = data["draws"]
            l = data["losses"]
            pct = (w / tot * 100.0) if tot > 0 else 0.0

            is_curr = (self.demo_mode == m_key)
            row_col = COLOR_TEXT_CYAN if is_curr else COLOR_TEXT_MAIN
            star = ">" if is_curr else " "

            txt = self.font_tiny.render(f"{star}{m_tag}: {w:2d}W - {d:2d}D - {l:2d}L ({pct:4.1f}%)", True, row_col)
            self.screen.blit(txt, (panel_x + 20, sb_y))
            sb_y += 18

        y += 122

        # 3. Squad Telemetry Cards
        self.screen.blit(self.font_body.render("SQUAD TELEMETRY", True, COLOR_TEXT_MAIN), (panel_x + 16, y))
        y += 20

        agents = [
            ("commander", "COMMANDER", COLOR_TEXT_GOLD),
            ("defense", "DEFENSE GUARD", COLOR_TEXT_GREEN),
            ("attack", "ATTACK SCOUT", COLOR_TEXT_PURPLE),
        ]
        for aid, aname, acol in agents:
            u = state.friendly_units.get(aid)
            if not u:
                continue

            card = pygame.Rect(panel_x + 12, y, panel_w - 24, 82)
            pygame.draw.rect(self.screen, (22, 28, 44), card, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, card, 1, border_radius=6)

            # Name and active indicator
            status_txt = "[ALIVE]" if u.is_alive else "[OFFLINE]"
            status_col = COLOR_TEXT_GREEN if u.is_alive else COLOR_TEXT_RED
            self.screen.blit(self.font_small.render(aname, True, acol), (panel_x + 20, y + 6))
            self.screen.blit(self.font_tiny.render(status_txt, True, status_col), (panel_x + panel_w - 75, y + 7))

            # HP Bar
            hp_txt = self.font_tiny.render(f"HP: {int(u.hp)}/{int(u.max_hp)}", True, COLOR_TEXT_DIM)
            self.screen.blit(hp_txt, (panel_x + 20, y + 24))

            # Current Action
            act_name = ACTION_NAMES.get(ActionType(u.current_action), "Unknown")
            act_txt = self.font_tiny.render(f"Act: {act_name}", True, COLOR_TEXT_CYAN)
            self.screen.blit(act_txt, (panel_x + 20, y + 42))

            # Communication Sent Token
            t = self.last_sent_tokens.get(aid, 0)
            if self.demo_mode == "comm":
                comm_display = f"Comm: Raw Token {t}" if t > 0 else "Comm: Token 0 (Silent)"
            else:
                comm_display = f"Comm: T{t} ({decode_token_label(t)})"
            t_txt = self.font_tiny.render(comm_display, True, COLOR_TEXT_GOLD if t > 0 else (120, 135, 155))
            self.screen.blit(t_txt, (panel_x + 20, y + 60))

            y += 90

        # 4. Keyboard Shortcuts Footer in Sidebar
        guide_box = pygame.Rect(panel_x + 12, y, panel_w - 24, 76)
        pygame.draw.rect(self.screen, (18, 22, 34), guide_box, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, guide_box, 1, border_radius=4)
        self.screen.blit(self.font_tiny.render("KEYS: [1-4] Switch Mode", True, COLOR_TEXT_GOLD), (panel_x + 18, y + 8))
        self.screen.blit(self.font_tiny.render("      [TAB] Cycle Mode", True, COLOR_TEXT_MAIN), (panel_x + 18, y + 22))
        self.screen.blit(self.font_tiny.render("      [R] Replay Same Seed", True, COLOR_TEXT_MAIN), (panel_x + 18, y + 36))
        self.screen.blit(self.font_tiny.render("      [N] Next Seed | [O] Opponent", True, COLOR_TEXT_MAIN), (panel_x + 18, y + 50))
        self.screen.blit(self.font_tiny.render("      [SPACE] Pause | [UP/DN] Speed", True, COLOR_TEXT_DIM), (panel_x + 18, y + 64))

    def _draw_bottom_ticker(self, state: Any) -> None:
        """Bottom communication message stream with raw token IDs for learned modes."""
        ticker_y = 40 + int(self.map_h)
        ticker_h = self.height - ticker_y
        ticker_rect = pygame.Rect(0, ticker_y, self.width, ticker_h)
        pygame.draw.rect(self.screen, (10, 13, 20), ticker_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (0, ticker_y), (self.width, ticker_y), 2)

        lbl = self.font_body.render("COMM CHANNEL //", True, COLOR_TEXT_GOLD)
        self.screen.blit(lbl, (14, ticker_y + 16))

        if self.demo_mode in ("oracle", "comm"):
            t_atk = self.last_sent_tokens.get("attack", 0)
            t_def = self.last_sent_tokens.get("defense", 0)
            t_cmd = self.last_sent_tokens.get("commander", 0)

            if self.demo_mode == "comm":
                # For learned modes, show raw token IDs instead of decoding into lane names
                msg_str = (
                    f"ATTACK: [Token ID {t_atk}]  |  "
                    f"DEFENSE: [Token ID {t_def}]  |  "
                    f"COMMANDER: [Token ID {t_cmd}]"
                )
            else:
                msg_str = (
                    f"ATTACK: T{t_atk} [{decode_token_label(t_atk)}]  |  "
                    f"DEFENSE: T{t_def} [{decode_token_label(t_def)}]  |  "
                    f"COMMANDER: T{t_cmd} [{decode_token_label(t_cmd)}]"
                )
            color = COLOR_TEXT_MAIN if (t_atk > 0 or t_def > 0) else COLOR_TEXT_DIM
            stream = self.font_body.render(msg_str, True, color)
            self.screen.blit(stream, (180, ticker_y + 16))
        else:
            silence_txt = self.font_body.render(
                f"[CHANNEL SUPPRESSED] Messages blocked in {self.demo_mode.upper()} mode. Units receive zero comm tokens.",
                True,
                (140, 155, 175),
            )
            self.screen.blit(silence_txt, (180, ticker_y + 16))

    def _draw_checkpoint_missing_banner(self, warning_msg: str) -> None:
        """Render floating amber warning if a requested neural model checkpoint is absent."""
        warn_rect = pygame.Rect(50, 110, int(self.map_w) - 100, 36)
        pygame.draw.rect(self.screen, (50, 25, 10), warn_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_TEXT_GOLD, warn_rect, 2, border_radius=6)
        txt = self.font_small.render(f"⚠ {warning_msg} — Falling back to autonomous heuristic.", True, COLOR_TEXT_GOLD)
        self.screen.blit(txt, (warn_rect.centerx - txt.get_width() // 2, warn_rect.y + 10))

    def handle_events(self) -> bool:
        """Handle keyboard & window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.set_demo_mode("oracle", reset_env=True)
                elif event.key == pygame.K_2:
                    self.set_demo_mode("silent", reset_env=True)
                elif event.key == pygame.K_3:
                    self.set_demo_mode("comm", reset_env=True)
                elif event.key == pygame.K_4:
                    self.set_demo_mode("no_comm", reset_env=True)
                elif event.key == pygame.K_TAB:
                    curr_i = DEMO_MODES.index(self.demo_mode)
                    next_m = DEMO_MODES[(curr_i + 1) % len(DEMO_MODES)]
                    self.set_demo_mode(next_m, reset_env=True)
                elif event.key == pygame.K_r:
                    # Replay exact same seed
                    self.set_demo_mode(self.demo_mode, reset_env=True)
                elif event.key == pygame.K_n:
                    # Next seed
                    self.current_seed += 1
                    self.set_demo_mode(self.demo_mode, reset_env=True)
                elif event.key == pygame.K_p:
                    # Previous seed
                    self.current_seed = max(0, self.current_seed - 1)
                    self.set_demo_mode(self.demo_mode, reset_env=True)
                elif event.key == pygame.K_o:
                    opps = ["aggressive", "defensive", "resource_focused"]
                    curr_idx = opps.index(self.env.opponent_strategy) if self.env.opponent_strategy in opps else 0
                    self.env.opponent_strategy = opps[(curr_idx + 1) % len(opps)]
                    self.set_demo_mode(self.demo_mode, reset_env=True)
                elif event.key == pygame.K_UP:
                    self.fps = min(120, self.fps + 10)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(5, self.fps - 5)
        return True

    def run_episode(self) -> None:
        """Run interactive presentation episode loop with full mode toggling."""
        self.env.reset(seed=self.current_seed)
        obs, _ = self.env.reset(seed=self.current_seed)
        running = True

        while running:
            running = self.handle_events()
            if not running:
                break

            if not self.paused and not self.match_finished:
                actions = self.select_actions(obs)
                next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
                obs = next_obs

                if any(terminations.values()) or any(truncations.values()):
                    self.record_match_result(self.env.state.winner, self.env.state.win_type)

            self.render_frame()
            self.clock.tick(self.fps)

        pygame.quit()
