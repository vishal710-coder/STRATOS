"""Futuristic Pygame Interactive Battlefield Visualizer for STRATOS AI."""

import math
import os
import sys
from typing import Any, Dict, Optional
import pygame
import numpy as np
import torch

from environment.strategy_game import StrategyGameEnv
from environment.actions import ACTION_NAMES, ActionType
from agents.policies import ActorCriticPolicy
from agents.commander import CommanderAgent
from agents.defense import DefenseAgent
from agents.attack import AttackAgent
from training.checkpoint import load_checkpoint


# Palette Definitions
COLOR_BG = (12, 16, 26)
COLOR_GRID = (24, 32, 50)
COLOR_PANEL_BG = (18, 24, 38)
COLOR_PANEL_BORDER = (45, 60, 90)
COLOR_TEXT_MAIN = (220, 230, 245)
COLOR_TEXT_DIM = (130, 145, 175)
COLOR_TEXT_GOLD = (255, 215, 80)
COLOR_TEXT_CYAN = (80, 220, 255)
COLOR_TEXT_RED = (255, 85, 105)
COLOR_TEXT_GREEN = (90, 240, 140)

COLOR_FRIENDLY_BASE = (30, 140, 255)
COLOR_ENEMY_BASE = (240, 50, 75)
COLOR_RESOURCE_NODE = (50, 230, 210)


class BattlefieldRenderer:
    """Renders the real-time strategy simulation with rich tactical HUD."""

    def __init__(
        self,
        env: StrategyGameEnv,
        policy: Optional[ActorCriticPolicy] = None,
        width: int = 1100,
        height: int = 680,
        headless: bool = False,
    ):
        self.env = env
        self.policy = policy
        self.width = width
        self.height = height
        self.map_w = env.map_w
        self.map_h = env.map_h
        self.headless = headless

        self.heuristic_agents = {
            "commander": CommanderAgent(role="commander"),
            "defense": DefenseAgent(role="defense"),
            "attack": AttackAgent(role="attack"),
        }

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("STRATOS AI - Tactical Strategy Battlefield")

        # Fonts
        self.font_large = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 13)
        self.font_small = pygame.font.SysFont("Consolas", 11)

        self.clock = pygame.time.Clock()
        self.fps = 30
        self.paused = False

    def render_frame(self) -> None:
        """Render complete simulation frame."""
        state = self.env.state
        if state is None:
            return

        self.screen.fill(COLOR_BG)

        # 1. Draw Grid Lines on Battlefield area (800 x 600)
        battle_rect = pygame.Rect(0, 40, int(self.map_w), int(self.map_h))
        pygame.draw.rect(self.screen, (15, 20, 32), battle_rect)

        for x in range(0, int(self.map_w), 40):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 40), (x, 40 + int(self.map_h)), 1)
        for y in range(40, 40 + int(self.map_h), 40):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (int(self.map_w), y), 1)

        # 2. Draw Resource Nodes
        for node in state.resource_nodes:
            ny = int(node.y + 40)
            nx = int(node.x)
            # Pulse glow based on amount
            radius = int(14 + (node.amount / node.max_amount) * 8)
            pygame.draw.circle(self.screen, (30, 80, 90), (nx, ny), radius + 4)
            pygame.draw.circle(self.screen, COLOR_RESOURCE_NODE, (nx, ny), radius)
            # Resource text
            txt = self.font_small.render(f"RES:{int(node.amount)}", True, (10, 30, 30))
            self.screen.blit(txt, (nx - txt.get_width() // 2, ny - 6))

        # 3. Draw Bases
        for team, base in state.bases.items():
            bx = int(base.x)
            by = int(base.y + 40)
            color = COLOR_FRIENDLY_BASE if team == "friendly" else COLOR_ENEMY_BASE

            # Outer shield aura
            shield_radius = 42
            pygame.draw.circle(self.screen, (*color[:2], 120), (bx, by), shield_radius, 2)
            pygame.draw.rect(self.screen, color, (bx - 26, by - 26, 52, 52), border_radius=8)

            # Health bar above base
            hp_ratio = max(0.0, base.hp / base.max_hp)
            bar_w = 60
            pygame.draw.rect(self.screen, (50, 50, 50), (bx - 30, by - 38, bar_w, 6))
            pygame.draw.rect(self.screen, COLOR_TEXT_GREEN if hp_ratio > 0.4 else COLOR_TEXT_RED, (bx - 30, by - 38, int(bar_w * hp_ratio), 6))

            name_txt = self.font_small.render(f"{team.upper()} HQ ({int(base.hp)})", True, COLOR_TEXT_MAIN)
            self.screen.blit(name_txt, (bx - name_txt.get_width() // 2, by + 30))

        # 4. Draw Units
        # Friendly units
        role_colors = {
            "commander": (255, 215, 40),
            "defense": (60, 220, 110),
            "attack": (190, 90, 240),
        }
        for u in state.friendly_units.values():
            if u.is_alive:
                self._draw_unit(u, role_colors.get(u.role, (100, 200, 255)), is_friendly=True)

        # Enemy units
        for u in state.enemy_units.values():
            if u.is_alive:
                self._draw_unit(u, COLOR_TEXT_RED, is_friendly=False)

        # 5. Top Status Header
        self._draw_top_bar(state)

        # 6. Right Side Status Panel (Agent telemetry)
        self._draw_agent_telemetry_panel(state)

        # 7. Bottom Message Ticker (Live agent communication)
        self._draw_communication_ticker()

        pygame.display.flip()

    def _draw_unit(self, unit: Any, color: tuple, is_friendly: bool) -> None:
        ux = int(unit.x)
        uy = int(unit.y + 40)

        # Vision radius hint
        pygame.draw.circle(self.screen, (*color[:2], 30), (ux, uy), int(unit.attack_range), 1)

        # Body circle
        radius = 11 if is_friendly else 9
        pygame.draw.circle(self.screen, color, (ux, uy), radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (ux, uy), radius, 2)

        # Health bar
        hp_ratio = max(0.0, unit.hp / unit.max_hp)
        bar_w = 26
        pygame.draw.rect(self.screen, (30, 30, 30), (ux - 13, uy - 18, bar_w, 4))
        pygame.draw.rect(self.screen, COLOR_TEXT_GREEN if hp_ratio > 0.3 else COLOR_TEXT_RED, (ux - 13, uy - 18, int(bar_w * hp_ratio), 4))

        # Label
        lbl = unit.name[:3].upper() if is_friendly else "ENM"
        txt = self.font_small.render(lbl, True, COLOR_TEXT_MAIN)
        self.screen.blit(txt, (ux - txt.get_width() // 2, uy + 12))

    def _draw_top_bar(self, state: Any) -> None:
        """Top navigation bar."""
        header_rect = pygame.Rect(0, 0, self.width, 40)
        pygame.draw.rect(self.screen, (10, 14, 22), header_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (0, 40), (self.width, 40), 2)

        title_txt = self.font_title.render("STRATOS AI // TACTICAL SIMULATION", True, COLOR_TEXT_CYAN)
        self.screen.blit(title_txt, (14, 10))

        step_txt = self.font_body.render(f"STEP: {state.step_count}/{state.max_steps}", True, COLOR_TEXT_MAIN)
        self.screen.blit(step_txt, (360, 12))

        f_res = self.font_body.render(f"RES: {int(state.friendly_resources)}", True, COLOR_TEXT_GOLD)
        self.screen.blit(f_res, (500, 12))

        directive_txt = self.font_body.render(
            f"DIRECTIVE: {self.env.coordinator.active_directive.value}",
            True,
            COLOR_TEXT_GREEN if self.env.coordinator.threat_level < 0.3 else COLOR_TEXT_RED,
        )
        self.screen.blit(directive_txt, (620, 12))

        mode_txt = self.font_small.render(f"MODE: {self.env.coordination_mode.upper()}", True, COLOR_TEXT_DIM)
        self.screen.blit(mode_txt, (self.width - 160, 14))

    def _draw_agent_telemetry_panel(self, state: Any) -> None:
        """Telemetry sidebar on right."""
        panel_x = int(self.map_w)
        panel_w = self.width - panel_x
        panel_rect = pygame.Rect(panel_x, 40, panel_w, int(self.map_h))
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (panel_x, 40), (panel_x, 40 + int(self.map_h)), 2)

        header = self.font_title.render("SQUAD TELEMETRY", True, COLOR_TEXT_GOLD)
        self.screen.blit(header, (panel_x + 16, 52))

        agents = [("commander", "COMMANDER"), ("defense", "DEFENSE GUARD"), ("attack", "ATTACK STRIKER")]
        y_cursor = 86

        for aid, label in agents:
            u = state.friendly_units.get(aid)
            if not u:
                continue

            pygame.draw.rect(self.screen, (24, 32, 48), (panel_x + 10, y_cursor, panel_w - 20, 120), border_radius=6)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (panel_x + 10, y_cursor, panel_w - 20, 120), 1, border_radius=6)

            # Header
            status_color = COLOR_TEXT_GREEN if u.is_alive else COLOR_TEXT_RED
            status_txt = "ACTIVE" if u.is_alive else "OFFLINE"
            t1 = self.font_body.render(f"{label}", True, COLOR_TEXT_MAIN)
            t2 = self.font_small.render(f"[{status_txt}]", True, status_color)
            self.screen.blit(t1, (panel_x + 18, y_cursor + 8))
            self.screen.blit(t2, (panel_x + panel_w - 95, y_cursor + 9))

            # HP
            hp_txt = self.font_small.render(f"HP: {int(u.hp)}/{int(u.max_hp)}", True, COLOR_TEXT_DIM)
            self.screen.blit(hp_txt, (panel_x + 18, y_cursor + 28))

            # Current Action
            act_name = ACTION_NAMES.get(ActionType(u.current_action), "Unknown")
            act_txt = self.font_small.render(f"Action: {act_name}", True, COLOR_TEXT_CYAN)
            self.screen.blit(act_txt, (panel_x + 18, y_cursor + 48))

            # Decision Source
            source = "Learned Policy (PPO)" if self.policy else "Autonomous Heuristic"
            src_txt = self.font_small.render(f"Source: {source}", True, (160, 175, 200))
            self.screen.blit(src_txt, (panel_x + 18, y_cursor + 68))

            # Combat Stats
            combat_txt = self.font_small.render(f"Kills: {u.kills} | Dmg: {int(u.damage_dealt)}", True, COLOR_TEXT_GOLD)
            self.screen.blit(combat_txt, (panel_x + 18, y_cursor + 88))

            y_cursor += 134

        # Opponent strategy card
        pygame.draw.rect(self.screen, (32, 20, 24), (panel_x + 10, y_cursor, panel_w - 20, 80), border_radius=6)
        opp_title = self.font_body.render("OPPONENT PROFILE", True, COLOR_TEXT_RED)
        opp_strat = self.font_small.render(f"Strategy: {self.env.opponent_strategy.upper()}", True, COLOR_TEXT_MAIN)
        self.screen.blit(opp_title, (panel_x + 18, y_cursor + 8))
        self.screen.blit(opp_strat, (panel_x + 18, y_cursor + 32))

    def _draw_communication_ticker(self) -> None:
        """Bottom communication message stream."""
        ticker_y = 40 + int(self.map_h)
        ticker_h = self.height - ticker_y
        ticker_rect = pygame.Rect(0, ticker_y, self.width, ticker_h)
        pygame.draw.rect(self.screen, (10, 13, 20), ticker_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (0, ticker_y), (self.width, ticker_y), 2)

        lbl = self.font_body.render("COMMUNICATION BUS //", True, COLOR_TEXT_GOLD)
        self.screen.blit(lbl, (14, ticker_y + 8))

        msgs = self.env.coordinator.get_recent_messages(limit=3)
        if msgs:
            msg_strs = [f"[{m.sender.upper()} -> {m.recipient.upper()}: {m.msg_type.value}]" for m in msgs]
            stream_txt = self.font_body.render("  |  ".join(msg_strs), True, COLOR_TEXT_MAIN)
            self.screen.blit(stream_txt, (200, ticker_y + 8))
        else:
            hint = self.font_small.render("No active transmissions (Independent Mode or Standby)", True, COLOR_TEXT_DIM)
            self.screen.blit(hint, (200, ticker_y + 9))

    def handle_events(self) -> bool:
        """Handle keyboard & window events. Return False to exit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.env.reset()
                elif event.key == pygame.K_m:
                    new_mode = "independent" if self.env.coordination_mode == "coordinated" else "coordinated"
                    self.env.coordination_mode = new_mode
                    self.env.coordinator.coordination_mode = new_mode
                    self.env.reset()
                elif event.key == pygame.K_UP:
                    self.fps = min(120, self.fps + 10)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(5, self.fps - 5)
        return True

    def _draw_game_over_overlay(self, state: Any) -> None:
        """Render prominent post-match victory/defeat modal with full statistics."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((8, 12, 20, 215))
        self.screen.blit(overlay, (0, 0))

        # Modal box
        card_w, card_h = 680, 400
        cx = (self.width - card_w) // 2
        cy = (self.height - card_h) // 2
        modal_rect = pygame.Rect(cx, cy, card_w, card_h)

        is_win = state.winner == "friendly"
        is_loss = state.winner == "enemy"
        theme_color = COLOR_TEXT_GREEN if is_win else (COLOR_TEXT_RED if is_loss else COLOR_TEXT_GOLD)

        pygame.draw.rect(self.screen, (16, 22, 34), modal_rect, border_radius=10)
        pygame.draw.rect(self.screen, theme_color, modal_rect, 2, border_radius=10)

        # Header Title
        title_str = "TACTICAL VICTORY // SQUAD TRIUMPH" if is_win else ("TACTICAL DEFEAT // BASE OVERRUN" if is_loss else "TACTICAL STALEMATE // DRAW")
        t_surf = self.font_large.render(title_str, True, theme_color)
        self.screen.blit(t_surf, (cx + (card_w - t_surf.get_width()) // 2, cy + 24))

        # Termination Reason
        fb = state.bases.get("friendly")
        eb = state.bases.get("enemy")
        friendly_alive = any(u.is_alive for u in state.friendly_units.values())
        enemy_alive = any(u.is_alive for u in state.enemy_units.values())

        f_score = (fb.hp if fb else 0) + state.friendly_resources * 0.5
        e_score = (eb.hp if eb else 0) + state.enemy_resources * 0.5

        if fb and fb.hp <= 0:
            reason = "Friendly Headquarters Structure Destroyed"
        elif eb and eb.hp <= 0:
            reason = "Enemy Headquarters Base Destroyed"
        elif not friendly_alive:
            reason = "Friendly Squad Eliminated (Squad Wipeout)"
        elif not enemy_alive:
            reason = "Enemy Squad Eliminated (Full Wipeout)"
        elif state.truncated:
            reason = f"300-Step Limit: Score Decision ({f_score:.1f} vs {e_score:.1f})"
        else:
            reason = "Simulation Episode Concluded"

        r_surf = self.font_body.render(f"Conclusion: {reason}", True, COLOR_TEXT_MAIN)
        self.screen.blit(r_surf, (cx + (card_w - r_surf.get_width()) // 2, cy + 62))

        # Divider line
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (cx + 30, cy + 92), (cx + card_w - 30, cy + 92), 1)

        # Match Stats Grid (2 columns)
        col1_x = cx + 50
        col2_x = cx + 370
        y_pos = cy + 115

        # Col 1: Friendly Squad
        self.screen.blit(self.font_title.render("FRIENDLY SQUAD STATS", True, COLOR_TEXT_CYAN), (col1_x, y_pos))
        self.screen.blit(self.font_body.render(f"HQ Base HP: {int(fb.hp if fb else 0)} / 1000", True, COLOR_TEXT_MAIN), (col1_x, y_pos + 30))
        surv = sum(1 for u in state.friendly_units.values() if u.is_alive)
        self.screen.blit(self.font_body.render(f"Units Survived: {surv} / 3", True, COLOR_TEXT_MAIN), (col1_x, y_pos + 55))
        self.screen.blit(self.font_body.render(f"Resources: {int(state.friendly_resources)} Crystals", True, COLOR_TEXT_GOLD), (col1_x, y_pos + 80))
        tot_dmg = sum(u.damage_dealt for u in state.friendly_units.values())
        self.screen.blit(self.font_body.render(f"Damage Dealt: {int(tot_dmg)} HP", True, COLOR_TEXT_MAIN), (col1_x, y_pos + 105))

        # Col 2: Tactical Data
        self.screen.blit(self.font_title.render("TACTICAL DATA", True, COLOR_TEXT_CYAN), (col2_x, y_pos))
        self.screen.blit(self.font_body.render(f"Mode: {self.env.coordination_mode.upper()}", True, COLOR_TEXT_MAIN), (col2_x, y_pos + 30))
        self.screen.blit(self.font_body.render(f"Opponent: {self.env.opponent_strategy.upper()}", True, COLOR_TEXT_MAIN), (col2_x, y_pos + 55))
        msgs = len(self.env.coordinator.message_history)
        self.screen.blit(self.font_body.render(f"Messages: {msgs} Transmissions", True, COLOR_TEXT_GREEN if msgs > 0 else COLOR_TEXT_DIM), (col2_x, y_pos + 80))
        en_killed = sum(1 for u in state.enemy_units.values() if not u.is_alive)
        self.screen.blit(self.font_body.render(f"Enemies Killed: {en_killed} / 3", True, COLOR_TEXT_MAIN), (col2_x, y_pos + 105))

        # Footer Action Guide
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (cx + 30, cy + 285), (cx + card_w - 30, cy + 285), 1)
        hint = self.font_body.render("[PRESS ANY KEY] Start Next Match  |  [M] Toggle Mode  |  [Q] Exit", True, COLOR_TEXT_GOLD)
        self.screen.blit(hint, (cx + (card_w - hint.get_width()) // 2, cy + 310))
        sub_hint = self.font_small.render("Match paused for inspection. Press any key whenever you are ready.", True, COLOR_TEXT_DIM)
        self.screen.blit(sub_hint, (cx + (card_w - sub_hint.get_width()) // 2, cy + 345))

    def run_episode(self) -> None:
        """Run single visual episode to completion."""
        obs, _ = self.env.reset()
        running = True

        while running:
            running = self.handle_events()
            if not running:
                break

            if not self.paused:
                action_dict = {}
                if self.policy is not None:
                    agent_obs = np.array([obs[agent] for agent in self.env.possible_agents])
                    obs_t = torch.as_tensor(agent_obs, dtype=torch.float32)
                    with torch.no_grad():
                        action_t, _, _, _ = self.policy.get_action_and_value(obs_t, deterministic=True)
                    for i, agent in enumerate(self.env.possible_agents):
                        action_dict[agent] = int(action_t[i].item())
                else:
                    # Autonomous heuristic fallback
                    for agent in self.env.possible_agents:
                        action_dict[agent] = self.heuristic_agents[agent].select_action(obs[agent])

                next_obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
                obs = next_obs

                if any(terminations.values()) or any(truncations.values()):
                    # Render final frame and game over screen
                    self.render_frame()
                    self._draw_game_over_overlay(self.env.state)
                    pygame.display.flip()

                    # Pause indefinitely: wait for user to press any key to inspect details
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return
                            elif event.type == pygame.KEYDOWN:
                                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                                    pygame.quit()
                                    return
                                elif event.key == pygame.K_m:
                                    new_mode = "independent" if self.env.coordination_mode == "coordinated" else "coordinated"
                                    self.env.coordination_mode = new_mode
                                    self.env.coordinator.coordination_mode = new_mode
                                    waiting = False
                                    break
                                else:
                                    # Any other key starts the next match!
                                    waiting = False
                                    break
                        self.clock.tick(30)

                    obs, _ = self.env.reset()

            self.render_frame()
            self.clock.tick(self.fps)

        pygame.quit()
