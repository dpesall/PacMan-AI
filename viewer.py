"""Pygame viewer for PacManEnv (authentic 28x31 arcade board).

Modes:
    python viewer.py --human            play it yourself (arrows / WASD)
    python viewer.py --model PATH       watch a trained model play
    python viewer.py                    watch the built-in random policy

Human play uses 3 lives, an endless level loop, score + high score, an arcade
HUD, a Pac death animation, and synthesized sound. The RL env underneath is
unchanged (the agent uses lives=1, terminate-on-clear).

The env is tile-based; this viewer interpolates sprite positions for smooth
movement. Options: --speed, --ghosts, --ghost-speed, --frightened, --cell,
--seed, --no-sound. Headless self-test: python viewer.py --headless --max-frames 120
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from pacman_env import PacManEnv, DIRS, UP, DOWN, LEFT, RIGHT, OUT, EYES, HOUSE, EXITING


# --- Colors ------------------------------------------------------------------
BG = (0, 0, 0)
WALL = (33, 38, 222)
WALL_EDGE = (80, 86, 255)
DOOR = (255, 170, 205)
PELLET = (255, 200, 160)
POWER = (255, 220, 180)
PAC = (255, 230, 0)
WHITE = (255, 255, 255)
EYE_PUPIL = (40, 40, 170)
TEXT = (235, 235, 235)
DIM = (150, 150, 160)
SCORE_POP = (120, 230, 255)
FRIGHT = (36, 40, 200)
FRIGHT_FLASH = (230, 230, 255)
CHERRY = (220, 30, 40)
STEM = (60, 170, 70)
GHOST_COLORS = [(222, 0, 0), (255, 150, 210), (0, 220, 220), (255, 160, 60)]

VEC = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
KEY_TO_DIR = {}

FREEZE_SECONDS = 0.5
DEATH_SECONDS = 1.1
HIGH_SCORE_FILE = "highscore.txt"


def lerp(a, b, t):
    return a + (b - a) * t


class SoundBank:
    """Retro sound effects synthesized with numpy (no asset files)."""

    def __init__(self, pygame, np, sr=22050):
        self.pygame = pygame
        self.np = np
        self.sr = sr
        self._chomp = [self._tone(520, 55, "square", 0.22), self._tone(380, 55, "square", 0.22)]
        self._eat_ghost = self._sweep(280, 1150, 260, "sine", 0.30)
        self._fruit = self._arp([660, 880, 1175], 70, 0.28)
        self._extra = self._arp([523, 659, 784, 1046], 85, 0.30)
        self._death = self._sweep(720, 120, 650, "square", 0.30)

    def _finish(self, wave):
        a = (self.np.clip(wave, -1, 1) * 0.9 * 32767).astype(self.np.int16)
        return self.pygame.sndarray.make_sound(self.np.ascontiguousarray(self.np.column_stack([a, a])))

    def _adsr(self, n, rel=0.02):
        t = self.np.arange(n) / self.sr
        dur = n / self.sr
        return self.np.clip(self.np.minimum(t / 0.005, (dur - t) / rel), 0, 1)

    def _tone(self, freq, ms, wave, vol):
        n = int(self.sr * ms / 1000)
        t = self.np.arange(n) / self.sr
        w = self.np.sin(2 * self.np.pi * freq * t)
        if wave == "square":
            w = self.np.sign(w)
        return self._finish(w * self._adsr(n) * vol)

    def _sweep(self, f0, f1, ms, wave, vol):
        n = int(self.sr * ms / 1000)
        freq = self.np.linspace(f0, f1, n)
        phase = 2 * self.np.pi * self.np.cumsum(freq) / self.sr
        w = self.np.sin(phase)
        if wave == "square":
            w = self.np.sign(w)
        return self._finish(w * self._adsr(n) * vol)

    def _arp(self, freqs, ms_each, vol):
        parts = []
        for f in freqs:
            n = int(self.sr * ms_each / 1000)
            t = self.np.arange(n) / self.sr
            parts.append(self.np.sin(2 * self.np.pi * f * t) * self._adsr(n) * vol)
        return self._finish(self.np.concatenate(parts))

    def chomp(self, i):
        self._chomp[i].play()

    def eat_ghost(self):
        self._eat_ghost.play()

    def fruit(self):
        self._fruit.play()

    def extra_life(self):
        self._extra.play()

    def death(self):
        self._death.play()


class Viewer:
    def __init__(self, args):
        self.args = args
        self.human = args.human
        self.sound = None  # set by main() after the mixer is up

        self.env = PacManEnv(
            num_ghosts=args.ghosts,
            ghost_speed=args.ghost_speed,
            frightened_steps=args.frightened,
            reward_ghost=10.0,
            reward_fruit=100.0,
            lives=3 if args.human else 1,
            endless=args.human,
            max_steps=10_000,
        )
        self.model = self._load_model(args.model) if args.model else None

        self.cell = args.cell
        self.W, self.H = self.env.W, self.env.H
        self.hud_h = max(58, int(self.cell * 2.5))
        self.move_interval = 1.0 / max(args.speed, 0.5)
        # Ghost render smoothing: glide at the ghost's own speed; only a gentle
        # nudge (low gain) when far behind, so velocity stays near-constant.
        self.render_target = 1.2
        self.render_gain = 1.0
        self.high_score = self._load_high_score()
        self._reset_state()

    @staticmethod
    def _load_model(path):
        try:
            from stable_baselines3 import PPO
        except ImportError:
            sys.exit("stable-baselines3 not installed. Run: pip install -r "
                     "requirements-train.txt   (or omit --model to watch random play)")
        return PPO.load(path)

    @staticmethod
    def _load_high_score():
        try:
            with open(HIGH_SCORE_FILE) as f:
                return int(f.read().strip() or 0)
        except (OSError, ValueError):
            return 0

    def _save_high_score(self):
        try:
            with open(HIGH_SCORE_FILE, "w") as f:
                f.write(str(self.high_score))
        except OSError:
            pass

    def _reset_state(self):
        self.obs, _ = self.env.reset(seed=self.args.seed)
        self.pac_prev = self.pac_curr = self.env.pac
        self.g_render = [[float(g.pos[0]), float(g.pos[1])] for g in self.env.ghosts]
        self.g_queue = [[] for _ in self.env.ghosts]
        self.g_last = [g.pos for g in self.env.ghosts]
        self.since_step = 0.0
        self.freeze = 0.0
        self.death_anim = 0.0
        self.death_pos = self.env.pac
        self.death_dir = LEFT
        self.popups = []
        self.game_over = False
        self.status = "playing"
        self.cur_dir = LEFT
        self.desired_dir = None
        self.started = not self.human
        self._prev_eaten = 0
        self._prev_lives = self.env.lives
        self._chomp_toggle = 0

    # ----------------------------------------------------------- simulation
    def _select_action(self):
        if self.human:
            cur = self.cur_dir
            if self.desired_dir is not None:
                nr, nc = self.env._neighbor(self.env.pac[0], self.env.pac[1], self.desired_dir)
                if not self.env._is_wall(nr, nc):
                    cur = self.desired_dir
            return cur if cur is not None else LEFT
        if self.model is not None:
            action, _ = self.model.predict(self.obs, deterministic=True)
            return int(action)
        return int(self.env.action_space.sample())

    def _trigger_sounds(self, info, terminated):
        if self.sound is None:
            return
        if info["pellets_eaten"] > self._prev_eaten:
            self.sound.chomp(self._chomp_toggle)
            self._chomp_toggle ^= 1
        self._prev_eaten = info["pellets_eaten"]
        if info["ate_ghost"]:
            self.sound.eat_ghost()
        if info["ate_fruit"]:
            self.sound.fruit()
        if info["lives"] > self._prev_lives:
            self.sound.extra_life()
        self._prev_lives = info["lives"]
        if info["life_lost"] or (terminated and info["status"] == "died"):
            self.sound.death()

    def _sync_render(self):
        """Snap render state to the env (after a respawn or a level refill)."""
        self.pac_prev = self.pac_curr = self.env.pac
        self.g_render = [[float(g.pos[0]), float(g.pos[1])] for g in self.env.ghosts]
        self.g_queue = [[] for _ in self.env.ghosts]
        self.g_last = [g.pos for g in self.env.ghosts]
        self.since_step = 0.0

    def _logical_step(self):
        self.pac_prev = self.pac_curr
        action = self._select_action()
        self.cur_dir = action
        self.obs, _, terminated, truncated, info = self.env.step(action)
        self.status = info["status"]
        self.high_score = max(self.high_score, info["score"])
        self._trigger_sounds(info, terminated)

        if info["life_lost"] or (terminated and info["status"] == "died"):
            self.death_anim = DEATH_SECONDS
            self.death_pos = self.pac_prev
            self.death_dir = self.cur_dir
            if terminated or truncated:
                self.game_over = True
                self._save_high_score()
            return  # freeze the scene for the death animation

        self.pac_curr = self.env.pac
        if info["level_cleared"]:
            self._sync_render()
            return
        for i, g in enumerate(self.env.ghosts):
            if g.pos != self.g_last[i]:
                self.g_queue[i].append(g.pos)
                self.g_last[i] = g.pos
        if info["ate_ghosts"]:
            self.freeze = FREEZE_SECONDS
            self.popups = list(info["ate_ghosts"])
        if terminated or truncated:
            self.game_over = True
            self._save_high_score()

    def update(self, dt):
        if self.death_anim > 0:
            self.death_anim -= dt
            if self.death_anim <= 0:
                self.death_anim = 0.0
                if not self.game_over:
                    self._sync_render()
            return
        if self.game_over or not self.started:
            return
        if self.freeze > 0:
            self.freeze -= dt
            return
        if self.popups:
            self.popups = []
        self.since_step += dt
        guard = 0
        while self.since_step >= self.move_interval and not self.game_over and guard < 8:
            self._logical_step()
            self.since_step -= self.move_interval
            guard += 1
            if self.freeze > 0 or self.game_over or self.death_anim > 0:
                break

    def advance_render(self, dt):
        if self.death_anim > 0:
            return  # freeze ghosts during the death animation
        for i, g in enumerate(self.env.ghosts):
            q = self.g_queue[i]
            if not q:
                continue
            rp = self.g_render[i]
            if abs(q[0][1] - rp[1]) > self.W / 2:       # next tile is across the tunnel
                rp[0], rp[1] = float(q[0][0]), float(q[0][1])
                q.pop(0)
                continue
            lag = math.hypot(q[0][0] - rp[0], q[0][1] - rp[1]) + (len(q) - 1)
            base = self.env._ghost_speed(g) / self.move_interval
            budget = (base + max(0.0, lag - self.render_target) * self.render_gain) * dt
            guard = 0
            while q and budget > 1e-9 and guard < 16:
                guard += 1
                tr, tc = q[0]
                if abs(tc - rp[1]) > self.W / 2:
                    rp[0], rp[1] = float(tr), float(tc)
                    q.pop(0)
                    continue
                dr, dc = tr - rp[0], tc - rp[1]
                dist = math.hypot(dr, dc)
                if dist <= budget or dist < 1e-9:
                    rp[0], rp[1] = float(tr), float(tc)
                    q.pop(0)
                    budget -= dist
                else:
                    rp[0] += dr / dist * budget
                    rp[1] += dc / dist * budget
                    budget = 0

    @property
    def frac(self):
        if self.game_over or not self.started or self.freeze > 0:
            return 1.0
        return min(self.since_step / self.move_interval, 1.0)

    # -------------------------------------------------------------- drawing
    def _interp_center(self, prev_rc, curr_rc):
        pr, pc = prev_rc
        cr, cc = curr_rc
        if abs(cc - pc) > self.W / 2:
            pr, pc = cr, cc
        r = lerp(pr, cr, self.frac)
        c = lerp(pc, cc, self.frac)
        return c * self.cell + self.cell / 2, r * self.cell + self.cell / 2

    def draw(self, screen, pygame, font, big_font, t):
        screen.fill(BG)
        cell = self.cell

        for r in range(self.H):
            for c in range(self.W):
                if self.env.walls[r, c]:
                    rect = pygame.Rect(c * cell, r * cell, cell, cell)
                    pygame.draw.rect(screen, WALL, rect.inflate(-2, -2), border_radius=4)
                    pygame.draw.rect(screen, WALL_EDGE, rect.inflate(-2, -2), width=1, border_radius=4)
        for (r, c) in self.env.gate:
            pygame.draw.rect(screen, DOOR, (c * cell + 2, r * cell + cell // 2 - 2, cell - 4, 4))

        for (r, c) in self.env.dots:
            pygame.draw.circle(screen, PELLET, (c * cell + cell // 2, r * cell + cell // 2), max(1, cell // 10))
        if int(t * 4) % 2 == 0:
            for (r, c) in self.env.power_pellets:
                pygame.draw.circle(screen, POWER, (c * cell + cell // 2, r * cell + cell // 2), max(3, int(cell * 0.34)))

        if self.env.fruit_active:
            fr, fc = self.env._fruit_pos
            self._draw_cherry(screen, pygame, fc * cell + cell / 2, fr * cell + cell / 2, cell * 0.42)

        if self.death_anim <= 0:                          # ghosts vanish during death
            for i, g in enumerate(self.env.ghosts):
                rp = self.g_render[i]
                x, y = rp[1] * cell + cell / 2, rp[0] * cell + cell / 2
                if g.mode == EYES:
                    self._draw_eyes(screen, pygame, x, y, cell * 0.42, VEC.get(g.dir, (0, 1)))
                else:
                    self._draw_ghost(screen, pygame, x, y, cell * 0.46, g, t)

        if self.death_anim > 0:
            dx = self.death_pos[1] * cell + cell / 2
            dy = self.death_pos[0] * cell + cell / 2
            self._draw_death_pac(screen, pygame, dx, dy, cell * 0.48, self.death_dir,
                                 1.0 - self.death_anim / DEATH_SECONDS)
        else:
            px, py = self._interp_center(self.pac_prev, self.pac_curr)
            self._draw_pac(screen, pygame, px, py, cell * 0.48, self.cur_dir, t)

        for (r, c, pts) in self.popups:
            label = font.render(str(pts), True, SCORE_POP)
            screen.blit(label, (c * cell + cell // 2 - label.get_width() // 2,
                                r * cell + cell // 2 - label.get_height() // 2))

        self._draw_hud(screen, pygame, font, t)
        self._draw_overlays(screen, pygame, big_font, font)

    def _draw_pac(self, screen, pygame, cx, cy, r, direction, t):
        moving = self.started and not self.game_over and self.freeze <= 0
        mouth = (0.5 + 0.5 * math.sin(t * 9)) * 0.38 * math.pi if moving else 0.20 * math.pi
        self._pac_poly(screen, pygame, cx, cy, r, direction, mouth)

    def _draw_death_pac(self, screen, pygame, cx, cy, r, direction, p):
        mouth = (0.2 + 0.8 * p) * math.pi          # mouth opens until Pac vanishes
        if mouth < math.pi:
            self._pac_poly(screen, pygame, cx, cy, r, direction, mouth)

    def _pac_poly(self, screen, pygame, cx, cy, r, direction, mouth):
        dx, dy = VEC.get(direction, (1, 0))
        ang = math.atan2(dy, dx)
        pts = [(cx, cy)]
        for i in range(27):
            a = (ang + mouth) + (2 * math.pi - 2 * mouth) * i / 26
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        pygame.draw.polygon(screen, PAC, pts)

    def _draw_eyes(self, screen, pygame, cx, cy, r, look):
        er = r * 0.5
        for s in (-1, 1):
            ex, ey = cx + s * r * 0.45, cy
            pygame.draw.circle(screen, WHITE, (int(ex), int(ey)), int(er))
            pygame.draw.circle(screen, EYE_PUPIL,
                               (int(ex + look[0] * er * 0.5), int(ey + look[1] * er * 0.5)), max(2, int(er * 0.55)))

    def _draw_ghost(self, screen, pygame, cx, cy, r, g, t):
        frightened = g.mode == OUT and g.frightened > 0
        flashing = frightened and g.frightened <= max(8, self.env.frightened_steps // 4)
        if frightened:
            color = FRIGHT_FLASH if (flashing and int(t * 6) % 2 == 0) else FRIGHT
        else:
            color = GHOST_COLORS[g.kind % len(GHOST_COLORS)]
            if g.mode in (HOUSE, EXITING):
                color = tuple(int(v * 0.85) for v in color)
        top = cy - r * 0.2
        pygame.draw.circle(screen, color, (int(cx), int(top)), int(r))
        pygame.draw.rect(screen, color, pygame.Rect(int(cx - r), int(top), int(2 * r), int(r * 1.25)))
        bottom = top + r * 1.25
        seg = 2 * r / 3
        for i in range(3):
            x0 = cx - r + i * seg
            pygame.draw.polygon(screen, color, [(x0, bottom), (x0 + seg / 2, bottom - r * 0.4), (x0 + seg, bottom)])
        if frightened:
            for s in (-1, 1):
                pygame.draw.circle(screen, WHITE, (int(cx + s * r * 0.4), int(top - r * 0.05)), max(2, int(r * 0.16)))
        else:
            look = VEC.get(g.dir, (0, 0))
            er = r * 0.34
            for s in (-1, 1):
                ex, ey = cx + s * r * 0.42, top - r * 0.05
                pygame.draw.circle(screen, WHITE, (int(ex), int(ey)), int(er))
                pygame.draw.circle(screen, EYE_PUPIL,
                                   (int(ex + look[0] * er * 0.45), int(ey + look[1] * er * 0.45)), int(er * 0.5))

    def _draw_cherry(self, screen, pygame, cx, cy, r):
        pygame.draw.line(screen, STEM, (int(cx - r * 0.4), int(cy + r * 0.3)), (int(cx + r * 0.2), int(cy - r * 0.7)), 2)
        pygame.draw.line(screen, STEM, (int(cx + r * 0.5), int(cy + r * 0.3)), (int(cx + r * 0.2), int(cy - r * 0.7)), 2)
        pygame.draw.circle(screen, CHERRY, (int(cx - r * 0.4), int(cy + r * 0.4)), max(2, int(r * 0.5)))
        pygame.draw.circle(screen, CHERRY, (int(cx + r * 0.5), int(cy + r * 0.45)), max(2, int(r * 0.45)))

    def _draw_hud(self, screen, pygame, font, t):
        y0 = self.H * self.cell
        pygame.draw.rect(screen, (12, 12, 18), (0, y0, self.W * self.cell, self.hud_h))
        info = self.env._get_info()
        pad = 10
        s = font.render(f"SCORE {info['score']:>6}", True, PAC)
        screen.blit(s, (pad, y0 + 5))
        hi = font.render(f"HIGH {max(self.high_score, info['score']):>6}", True, TEXT)
        screen.blit(hi, (self.W * self.cell // 2 - hi.get_width() // 2, y0 + 5))
        lv = font.render(f"LEVEL {info['level']}", True, TEXT)
        screen.blit(lv, (self.W * self.cell - lv.get_width() - pad, y0 + 5))

        ly = y0 + 8 + s.get_height()
        ir = max(5, self.cell // 2 - 2)
        for k in range(min(info["lives"], 6)):
            self._pac_poly(screen, pygame, pad + ir + k * (ir * 2 + 6), ly + ir, ir, LEFT, 0.28 * math.pi)
        mode = "HUMAN" if self.human else ("AGENT" if self.model else "RANDOM")
        right = font.render(f"{mode}  Dots {info['pellets_eaten']}/{info['pellets_total']}  "
                            f"{info['mode'].upper()}", True, DIM)
        screen.blit(right, (self.W * self.cell - right.get_width() - pad, ly))

    def _draw_overlays(self, screen, pygame, big_font, font):
        msg = None
        if self.game_over and self.death_anim <= 0:
            msg = {"won": "YOU WIN!", "died": "GAME OVER", "timeout": "TIME UP"}.get(self.status, "DONE")
            sub = "press R to play again"
        elif self.human and not self.started and self.death_anim <= 0:
            msg, sub = "READY!", "press an arrow key (or WASD) to start"
        if not msg:
            return
        cx, cy = self.W * self.cell // 2, self.H * self.cell // 2
        panel = pygame.Surface((self.W * self.cell, int(self.cell * 4)), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        screen.blit(panel, (0, cy - int(self.cell * 2)))
        t1 = big_font.render(msg, True, PAC)
        screen.blit(t1, (cx - t1.get_width() // 2, cy - t1.get_height()))
        t2 = font.render(sub, True, TEXT)
        screen.blit(t2, (cx - t2.get_width() // 2, cy + 6))


def build_parser():
    p = argparse.ArgumentParser(description="Pac-Man viewer (human / agent / random).")
    p.add_argument("--human", action="store_true", help="play it yourself")
    p.add_argument("--model", type=str, default=None, help="path to a saved SB3 model")
    p.add_argument("--speed", type=float, default=7.0, help="logical moves per second")
    p.add_argument("--fps", type=int, default=60, help="render frames per second")
    p.add_argument("--cell", type=int, default=24, help="pixels per tile")
    p.add_argument("--ghosts", type=int, default=4, help="number of ghosts")
    p.add_argument("--ghost-speed", type=float, default=0.75, help="ghost speed (fraction of Pac)")
    p.add_argument("--frightened", type=int, default=45, help="frightened steps after a power pellet (0 disables)")
    p.add_argument("--seed", type=int, default=None, help="env seed")
    p.add_argument("--max-frames", type=int, default=None, help="exit after N frames")
    p.add_argument("--headless", action="store_true", help="no window (testing)")
    p.add_argument("--no-sound", action="store_true", help="disable sound")
    return p


def _make_sound(pygame, no_sound):
    if no_sound:
        return None
    try:
        import numpy as np
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(16)
        return SoundBank(pygame, np)
    except Exception:
        return None  # no audio device (e.g. headless) -> silent


def main():
    args = build_parser().parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    KEY_TO_DIR.update({
        pygame.K_UP: UP, pygame.K_DOWN: DOWN, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
        pygame.K_w: UP, pygame.K_s: DOWN, pygame.K_a: LEFT, pygame.K_d: RIGHT,
    })
    pygame.init()
    viewer = Viewer(args)
    viewer.sound = _make_sound(pygame, args.no_sound or args.headless)
    screen = pygame.display.set_mode((viewer.W * viewer.cell, viewer.H * viewer.cell + viewer.hud_h))
    pygame.display.set_caption("Pac-Man")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(13, viewer.cell // 2))
    big_font = pygame.font.SysFont("consolas", viewer.cell * 2, bold=True)

    frames = 0
    running = True
    while running:
        dt = clock.tick(args.fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    viewer._reset_state()
                elif viewer.human and event.key in KEY_TO_DIR:
                    viewer.desired_dir = KEY_TO_DIR[event.key]
                    viewer.started = True

        viewer.update(dt)
        viewer.advance_render(dt)
        t = pygame.time.get_ticks() / 1000.0
        viewer.draw(screen, pygame, font, big_font, t)
        pygame.display.flip()

        frames += 1
        if args.max_frames is not None and frames >= args.max_frames:
            running = False

    if viewer.high_score:
        viewer._save_high_score()
    pygame.quit()


if __name__ == "__main__":
    main()
