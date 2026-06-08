"""Tile-based Pac-Man as a Gymnasium environment (authentic arcade rules).

The simulation is purely tile-based: one decision per cell. Smooth movement is a
*rendering* concern handled by viewer.py, never here.

Observation : Box(0, 1, shape=(7, H, W), float32)
    channels  0 walls, 1 dots, 2 power-pellets, 3 pac-man,
              4 ghosts, 5 frightened-ghosts, 6 bonus-fruit
    (Only ghosts that are OUT in the maze appear in 4/5. Eyes and house-bound
    ghosts are harmless and omitted.)
Actions     : Discrete(4) -> 0 UP, 1 DOWN, 2 LEFT, 3 RIGHT
              A blocked move leaves Pac-Man in place (and eats the step penalty).

Arcade-accurate behaviors:
  * Exact 28x31 board, 240 dots + 4 energizers, side tunnel (row wraps L<->R).
  * Ghost house life-cycle: out --(eaten while frightened)--> eyes --> house
    --> re-emerge NORMAL (never frightened, even mid power-pellet).
  * Scatter/Chase alternation on the level-1 schedule; ghosts reverse on change.
  * Four personalities: Blinky/Pinky/Inky/Clyde target differently (incl. the
    "look 2/4 ahead" up-direction overflow quirk and Inky's vector double).
  * Per-ghost fractional speeds: slower than Pac, much slower when frightened or
    in the tunnel, fast as eyes; Blinky speeds up as dots run low (Cruise Elroy).
  * Bonus fruit appears below the house after 70 and 170 dots.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from arcade_maze import ARCADE_MAZE


# --- Actions -> (drow, dcol) -------------------------------------------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
_ORDER = (UP, LEFT, DOWN, RIGHT)  # arcade tie-break preference

# --- Observation channels ----------------------------------------------------
CH_WALL, CH_DOT, CH_POWER, CH_PAC, CH_GHOST, CH_FRIGHT, CH_FRUIT = range(7)
N_CHANNELS = 7

# --- Ghost modes / kinds -----------------------------------------------------
OUT, EYES, HOUSE, EXITING = "out", "eyes", "house", "exiting"
BLINKY, PINKY, INKY, CLYDE = 0, 1, 2, 3
SCATTER, CHASE = "scatter", "chase"

# Level-1 scatter/chase schedule, in steps (~7 steps/sec): 7s,20s,7s,20s,5s,20s,5s.
_SCHEDULE = [(SCATTER, 52), (CHASE, 140), (SCATTER, 52), (CHASE, 140),
             (SCATTER, 35), (CHASE, 140), (SCATTER, 35), (CHASE, 10 ** 9)]

_LARGE = 10 ** 6


class _Ghost:
    __slots__ = ("pos", "dir", "kind", "frightened", "mode",
                 "house_timer", "accum", "just_eaten", "reverse_now")

    def __init__(self, pos, kind=BLINKY, mode=OUT):
        self.pos = pos
        self.dir = None
        self.kind = kind
        self.frightened = 0
        self.mode = mode
        self.house_timer = 0
        self.accum = 0.0
        self.just_eaten = False
        self.reverse_now = False


class PacManEnv(gym.Env):
    """Authentic clear-the-board Pac-Man. Win = eat every dot; lose = touch a ghost."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 30}

    def __init__(
        self,
        maze=None,
        num_ghosts=4,
        ghost_speed=0.75,
        frightened_steps=0,
        max_steps=2000,
        reward_pellet=1.0,
        reward_power=1.0,
        reward_clear=100.0,
        reward_death=-50.0,
        reward_step=-0.01,
        reward_ghost=0.0,
        reward_fruit=0.0,
        lives=1,
        endless=False,
        render_mode=None,
    ):
        super().__init__()
        self.maze = list(maze) if maze is not None else list(ARCADE_MAZE)
        self.H = len(self.maze)
        self.W = max(len(row) for row in self.maze)
        self.maze = [row.ljust(self.W) for row in self.maze]  # guard trailing spaces

        self.num_ghosts = num_ghosts
        self.ghost_speed = ghost_speed
        self.frightened_steps = frightened_steps
        self.max_steps = max_steps
        self.reward_pellet = reward_pellet
        self.reward_power = reward_power
        self.reward_clear = reward_clear
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_ghost = reward_ghost
        self.reward_fruit = reward_fruit
        self._lives_start = lives
        self.endless = endless
        self.render_mode = render_mode

        # Speeds (fraction of Pac, who moves once per step) and pacing.
        self.frightened_speed = 0.5
        self.tunnel_speed = 0.4
        self.eyes_speed = 1.0
        self.house_speed = 0.5
        self.respawn_delay = 8
        self.exit_delay = 12
        self.elroy1, self.elroy2 = 20, 10          # Blinky speed-up dot thresholds

        # Bonus fruit.
        self.fruit_thresholds = (70, 170)
        self.fruit_steps = 70
        self.fruit_points = 100
        self.fruit_name = "cherry"

        # --- Parse the maze --------------------------------------------------
        self.walls = np.zeros((self.H, self.W), dtype=bool)
        self.gate = set()
        self.house = set()
        self._dot_start = set()
        self._power_start = set()
        self._ghost_slots = []
        self._pac_start = None
        for r, row in enumerate(self.maze):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls[r, c] = True
                elif ch == ".":
                    self._dot_start.add((r, c))
                elif ch == "o":
                    self._power_start.add((r, c))
                elif ch == "G":
                    self._ghost_slots.append((r, c))
                    self.house.add((r, c))
                elif ch == "-":
                    self.house.add((r, c))
                elif ch == "=":
                    self.gate.add((r, c))
                elif ch == "P":
                    self._pac_start = (r, c)
        if self._pac_start is None:
            raise ValueError("Maze has no Pac-Man start ('P').")
        if not self.gate or not self._ghost_slots:
            raise ValueError("Maze needs a ghost-house door ('=') and slots ('G').")
        self._total_edible = len(self._dot_start) + len(self._power_start)

        # Door geometry: exit tiles are the floor cells directly above the door.
        self._house_exits = set()
        for (gr, gc) in self.gate:
            er, ec = gr - 1, gc
            if 0 <= er < self.H and not self.walls[er, ec] and (er, ec) not in self.gate:
                self._house_exits.add((er, ec))
        if not self._house_exits:
            raise ValueError("No open floor above the ghost-house door.")
        self._blinky_start = sorted(self._house_exits)[0]

        # Precomputed routing fields (static maze, wrap-aware).
        self._home_dist = self._bfs(self.house, lambda r, c: not self.walls[r, c])
        exit_zone = self.house | self.gate | self._house_exits
        self._exit_dist = self._bfs(self._house_exits, lambda r, c: (r, c) in exit_zone)

        # Observation wall mask (the door reads as a wall to the agent).
        self._wall_mask = self.walls.copy()
        for (r, c) in self.gate:
            self._wall_mask[r, c] = True

        # Scatter corners (just outside each ghost's home corner).
        self._scatter_corner = {
            BLINKY: (0, self.W - 3), PINKY: (0, 2),
            INKY: (self.H - 1, self.W - 2), CLYDE: (self.H - 1, 1),
        }

        self._check_connectivity()

        # Tunnel = the *reachable* row with an open edge (off-board pockets also
        # have open edges but aren't reachable, so exclude them).
        self.tunnel_row = next((r for r in range(self.H)
                                if not self.walls[r, 0] and (r, 0) not in self.gate
                                and self._reachable[r, 0]), None)
        self.tunnel_tiles = set()
        if self.tunnel_row is not None:
            for c in range(self.W):
                if not self.walls[self.tunnel_row, c] and (c < 6 or c >= self.W - 6):
                    self.tunnel_tiles.add((self.tunnel_row, c))

        self._fruit_pos = self._pick_fruit_tile()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CHANNELS, self.H, self.W), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # Runtime state (reset()).
        self.pac = None
        self.dots = set()
        self.power_pellets = set()
        self.ghosts = []
        self.steps = 0
        self.pac_dir = LEFT
        self._status = "playing"
        self._fright_chain = 0
        self._ate_this_step = []
        self._global_mode = SCATTER
        self._mode_idx = 0
        self._mode_timer = 0
        self.fruit_active = False
        self.fruit_timer = 0
        self._fruit_done = set()

    # ------------------------------------------------------------- geometry
    def _neighbor(self, r, c, d):
        """Neighbor tile in direction d, wrapping columns through the tunnel."""
        dr, dc = DIRS[d]
        return r + dr, (c + dc) % self.W

    def _is_wall(self, r, c):
        if r < 0 or r >= self.H:
            return True
        c %= self.W
        return self.walls[r, c] or (r, c) in self.gate

    def _bfs(self, sources, passable):
        dist = np.full((self.H, self.W), _LARGE, dtype=int)
        q = deque()
        for s in sources:
            dist[s] = 0
            q.append(s)
        while q:
            r, c = q.popleft()
            for d in DIRS:
                nr, nc = self._neighbor(r, c, d)
                if 0 <= nr < self.H and passable(nr, nc) and dist[nr, nc] == _LARGE:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    def _check_connectivity(self):
        seen = np.zeros((self.H, self.W), dtype=bool)
        q = deque([self._pac_start])
        seen[self._pac_start] = True
        while q:
            r, c = q.popleft()
            for d in DIRS:
                nr, nc = self._neighbor(r, c, d)
                if not self._is_wall(nr, nc) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        bad = [p for p in (self._dot_start | self._power_start) if not seen[p]]
        if bad:
            raise ValueError(f"{len(bad)} pellet(s) unreachable, e.g. {bad[:5]}.")
        if self._home_dist[self._blinky_start] >= _LARGE:
            raise ValueError("Eyes cannot route into the ghost house.")
        self._reachable = seen

    def _pick_fruit_tile(self):
        """A reachable floor tile a few rows below the house (classic fruit spot)."""
        hr = max(r for (r, _) in self.house) + 2
        hc = round(sum(c for (_, c) in self.house) / len(self.house))
        best, bestd = None, None
        for r in range(self.H):
            for c in range(self.W):
                if (self._reachable[r, c] and not self.walls[r, c]
                        and (r, c) not in self.gate and (r, c) not in self.house):
                    d = (r - hr) ** 2 + (c - hc) ** 2
                    if bestd is None or d < bestd:
                        bestd, best = d, (r, c)
        return best

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.lives = self._lives_start
        self.score = 0
        self.level = 1
        self._extra_awarded = False
        self.steps = 0
        self._life_lost = False
        self._level_cleared = False
        self._ate_ghost = False
        self._ate_fruit = False
        self.dots = set(self._dot_start)
        self.power_pellets = set(self._power_start)
        self._fruit_done = set()
        self._reset_positions()
        return self._get_obs(), self._get_info()

    def _reset_positions(self):
        """Reset Pac, ghosts, and the scatter/chase clock -- used by reset(), by
        a respawn after losing a life, and by an endless-mode level refill. The
        board (dots), score, lives, and level are left untouched."""
        self.pac = self._pac_start
        self.pac_dir = LEFT
        self._status = "playing"
        self._fright_chain = 0
        self._ate_this_step = []
        self._global_mode, self._mode_idx = _SCHEDULE[0][0], 0
        self._mode_timer = _SCHEDULE[0][1]
        self.fruit_active = False
        self.fruit_timer = 0
        self.ghosts = []
        for i in range(self.num_ghosts):
            if i == 0:
                self.ghosts.append(_Ghost(self._blinky_start, kind=BLINKY, mode=OUT))
            else:
                slot = self._ghost_slots[(i - 1) % len(self._ghost_slots)]
                g = _Ghost(slot, kind=(i if i < 4 else BLINKY), mode=HOUSE)
                g.house_timer = self.exit_delay * i
                self.ghosts.append(g)

    def step(self, action):
        action = int(action)
        self.steps += 1
        reward = self.reward_step
        terminated = False
        truncated = False
        pac_prev = self.pac
        self._ate_this_step = []
        self._life_lost = False
        self._level_cleared = False
        self._ate_ghost = False
        self._ate_fruit = False
        for g in self.ghosts:
            g.just_eaten = False

        self._update_global_mode()

        # 1. Move Pac-Man (tunnel-wrapping; blocked -> stay).
        nr, nc = self._neighbor(self.pac[0], self.pac[1], action)
        if not self._is_wall(nr, nc):
            self.pac = (nr, nc % self.W)
            self.pac_dir = action

        # 2. Eat dot / power / fruit.
        if self.pac in self.dots:
            self.dots.discard(self.pac)
            reward += self.reward_pellet
            self._add_score(10)
        elif self.pac in self.power_pellets:
            self.power_pellets.discard(self.pac)
            reward += self.reward_power
            self._add_score(50)
            self._fright_chain = 0
            if self.frightened_steps > 0:
                for g in self.ghosts:
                    if g.mode == OUT:
                        g.frightened = self.frightened_steps
                        g.reverse_now = True
        if self.fruit_active and self.pac == self._fruit_pos:
            self.fruit_active = False
            reward += self.reward_fruit
            self._add_score(self.fruit_points)
            self._ate_fruit = True
            self._ate_this_step.append((self._fruit_pos[0], self._fruit_pos[1], self.fruit_points))
        self._maybe_spawn_fruit()

        # 3. Collision after Pac moves.
        reward, terminated = self._collisions(reward, terminated, None, pac_prev)

        # 4. Move ghosts, then collide again.
        if not terminated:
            ghost_prev = [g.pos for g in self.ghosts]
            self._move_ghosts()
            reward, terminated = self._collisions(reward, terminated, ghost_prev, pac_prev)

        # Caught by a ghost: lose a life and respawn, unless it was the last one.
        if terminated and self._status == "died":
            self.lives -= 1
            if self.lives > 0:
                terminated = False
                self._life_lost = True
                self._reset_positions()

        # 5. Timers.
        for g in self.ghosts:
            if g.mode == OUT and g.frightened > 0:
                g.frightened -= 1
        if self.fruit_active:
            self.fruit_timer -= 1
            if self.fruit_timer <= 0:
                self.fruit_active = False

        # 6. Win / level clear / timeout.
        if not terminated and not self.dots and not self.power_pellets:
            reward += self.reward_clear
            if self.endless:
                self.level += 1
                self._level_cleared = True
                self.dots = set(self._dot_start)
                self.power_pellets = set(self._power_start)
                self._fruit_done = set()
                self._reset_positions()
            else:
                terminated = True
                self._status = "won"
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            self._status = "timeout"

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------- mechanics
    def _maybe_spawn_fruit(self):
        eaten = self._total_edible - (len(self.dots) + len(self.power_pellets))
        for t in self.fruit_thresholds:
            if eaten >= t and t not in self._fruit_done:
                self._fruit_done.add(t)
                self.fruit_active = True
                self.fruit_timer = self.fruit_steps

    def _add_score(self, n):
        self.score += n
        # Extra life at 10,000 -- only when a lives system is in use, so the RL
        # default (lives=1) is never altered.
        if not self._extra_awarded and self.score >= 10000:
            self._extra_awarded = True
            if self._lives_start > 1:
                self.lives += 1

    def _update_global_mode(self):
        if any(g.mode == OUT and g.frightened > 0 for g in self.ghosts):
            return  # scatter/chase clock pauses during a fright period
        self._mode_timer -= 1
        if self._mode_timer <= 0 and self._mode_idx < len(_SCHEDULE) - 1:
            self._mode_idx += 1
            self._global_mode, self._mode_timer = _SCHEDULE[self._mode_idx]
            for g in self.ghosts:
                if g.mode == OUT:
                    g.reverse_now = True

    def _collisions(self, reward, terminated, ghost_prev, pac_prev):
        if terminated:
            return reward, terminated
        for i, g in enumerate(self.ghosts):
            if g.just_eaten or g.mode != OUT:
                continue
            hit = g.pos == self.pac
            if not hit and ghost_prev is not None:
                hit = g.pos == pac_prev and ghost_prev[i] == self.pac
            if not hit:
                continue
            if g.frightened > 0:
                self._fright_chain += 1
                points = 200 * (2 ** (self._fright_chain - 1))
                reward += self.reward_ghost
                self._add_score(points)
                self._ate_ghost = True
                g.mode, g.frightened, g.just_eaten = EYES, 0, True
                self._ate_this_step.append((g.pos[0], g.pos[1], points))
            else:
                reward += self.reward_death
                terminated = True
                self._status = "died"
                break
        return reward, terminated

    def _elroy_active(self):
        return (len(self.dots) + len(self.power_pellets)) <= self.elroy1

    def _ghost_speed(self, g):
        if g.mode == EYES:
            return self.eyes_speed
        if g.mode in (HOUSE, EXITING):
            return self.house_speed
        if g.frightened > 0:
            return self.frightened_speed
        if g.pos in self.tunnel_tiles:
            return self.tunnel_speed
        base = self.ghost_speed
        if g.kind == BLINKY:
            rem = len(self.dots) + len(self.power_pellets)
            if rem <= self.elroy2:
                base = 1.0
            elif rem <= self.elroy1:
                base = min(1.0, self.ghost_speed + 0.1)
        return base

    def _move_ghosts(self):
        for g in self.ghosts:
            if g.just_eaten:
                continue
            if g.mode == HOUSE:
                if g.house_timer > 0:
                    g.house_timer -= 1
                    continue
                g.mode = EXITING
            g.accum += self._ghost_speed(g)
            if g.accum >= 1.0:
                g.accum -= 1.0
                self._ghost_substep(g)

    def _ghost_substep(self, g):
        if g.mode == EXITING:
            d, nxt = self._greedy_to(self._exit_dist, g.pos)
            if d is not None:
                g.dir, g.pos = d, nxt
            if g.pos in self._house_exits:
                g.mode, g.frightened, g.dir, g.reverse_now = OUT, 0, LEFT, False
            return
        if g.mode == EYES:
            d, nxt = self._greedy_to(self._home_dist, g.pos)
            if d is not None:
                g.dir, g.pos = d, nxt
            if g.pos in self.house:
                g.mode, g.house_timer = HOUSE, self.respawn_delay
            return
        # OUT
        exits = [d for d in _ORDER if not self._is_wall(*self._neighbor(g.pos[0], g.pos[1], d))]
        if not exits:
            return
        if g.reverse_now:
            g.reverse_now = False
            rev = _OPPOSITE.get(g.dir)
            move = rev if rev in exits else self._target_move(g, exits)
        elif g.frightened > 0:
            choices = [d for d in exits if g.dir is None or d != _OPPOSITE[g.dir]] or exits
            move = choices[int(self.np_random.integers(len(choices)))]
        else:
            choices = [d for d in exits if g.dir is None or d != _OPPOSITE[g.dir]] or exits
            move = self._target_move(g, choices)
        g.pos = self._neighbor(g.pos[0], g.pos[1], move)
        g.dir = move

    def _target_move(self, g, choices):
        target = self._ghost_target(g)
        best, best_d = None, None
        for d in _ORDER:                       # iterate in tie-break order, strict <
            if d not in choices:
                continue
            nr, nc = self._neighbor(g.pos[0], g.pos[1], d)
            dist = (nr - target[0]) ** 2 + (nc - target[1]) ** 2
            if best_d is None or dist < best_d:
                best_d, best = dist, d
        return best

    def _ahead(self, pos, direction, n):
        dr, dc = DIRS[direction]
        r, c = pos[0] + dr * n, pos[1] + dc * n
        if direction == UP:        # arcade overflow quirk: also n tiles left
            c -= n
        return r, c

    def _ghost_target(self, g):
        scatter = self._global_mode == SCATTER
        if g.kind == BLINKY and self._elroy_active():
            scatter = False
        if scatter:
            return self._scatter_corner[g.kind]
        if g.kind == BLINKY:
            return self.pac
        if g.kind == PINKY:
            return self._ahead(self.pac, self.pac_dir, 4)
        if g.kind == INKY:
            pr, pc = self._ahead(self.pac, self.pac_dir, 2)
            br, bc = self.ghosts[0].pos if self.ghosts[0].kind == BLINKY else self.pac
            return 2 * pr - br, 2 * pc - bc
        if g.kind == CLYDE:
            d2 = (g.pos[0] - self.pac[0]) ** 2 + (g.pos[1] - self.pac[1]) ** 2
            return self.pac if d2 > 64 else self._scatter_corner[CLYDE]
        return self.pac

    def _greedy_to(self, dist, pos):
        best, ties = None, []
        for d in _ORDER:
            nr, nc = self._neighbor(pos[0], pos[1], d)
            if 0 <= nr < self.H and dist[nr, nc] < _LARGE:
                v = dist[nr, nc]
                if best is None or v < best:
                    best, ties = v, [(d, (nr, nc))]
                elif v == best:
                    ties.append((d, (nr, nc)))
        if not ties:
            return None, pos
        return ties[0]  # _ORDER already encodes the preference

    # ---------------------------------------------------------- observation
    def _get_obs(self):
        obs = np.zeros((N_CHANNELS, self.H, self.W), dtype=np.float32)
        obs[CH_WALL] = self._wall_mask
        for r, c in self.dots:
            obs[CH_DOT, r, c] = 1.0
        for r, c in self.power_pellets:
            obs[CH_POWER, r, c] = 1.0
        obs[CH_PAC, self.pac[0], self.pac[1]] = 1.0
        for g in self.ghosts:
            if g.mode != OUT:
                continue
            obs[CH_FRIGHT if g.frightened > 0 else CH_GHOST, g.pos[0], g.pos[1]] = 1.0
        if self.fruit_active:
            obs[CH_FRUIT, self._fruit_pos[0], self._fruit_pos[1]] = 1.0
        return obs

    def _get_info(self):
        remaining = len(self.dots) + len(self.power_pellets)
        return {
            "pellets_total": self._total_edible,
            "pellets_remaining": remaining,
            "pellets_eaten": self._total_edible - remaining,
            "steps": self.steps,
            "status": self._status,
            "mode": self._global_mode,
            "fruit_active": self.fruit_active,
            "ate_ghosts": list(self._ate_this_step),
            "score": self.score,
            "lives": self.lives,
            "level": self.level,
            "life_lost": self._life_lost,
            "level_cleared": self._level_cleared,
            "ate_ghost": self._ate_ghost,
            "ate_fruit": self._ate_fruit,
        }

    # --------------------------------------------------------------- render
    def render(self):
        if self.render_mode not in ("ansi", "human"):
            return None
        glyph = {}
        if self.fruit_active:
            glyph[self._fruit_pos] = "F"
        for g in self.ghosts:
            glyph[g.pos] = {OUT: "M" if g.frightened > 0 else "ABID"[g.kind],
                            EYES: '"', HOUSE: "h", EXITING: "h"}[g.mode]
        glyph[self.pac] = "P"
        lines = []
        for r in range(self.H):
            chars = []
            for c in range(self.W):
                if (r, c) in glyph:
                    chars.append(glyph[(r, c)])
                elif self.walls[r, c]:
                    chars.append("#")
                elif (r, c) in self.gate:
                    chars.append("=")
                elif (r, c) in self.power_pellets:
                    chars.append("o")
                elif (r, c) in self.dots:
                    chars.append(".")
                else:
                    chars.append(" ")
            lines.append("".join(chars))
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text + "\n")
            return None
        return text


if __name__ == "__main__":
    env = PacManEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (N_CHANNELS, env.H, env.W), obs.shape
    assert env.observation_space.contains(obs)
    print(f"Maze {env.W}x{env.H} | {info['pellets_total']} edible "
          f"({len(env._dot_start)} dots + {len(env._power_start)} power) "
          f"| {env.num_ghosts} ghosts | tunnel row {env.tunnel_row} "
          f"| fruit @ {env._fruit_pos} | exits {sorted(env._house_exits)}")
    env.render_mode = "ansi"
    print(env.render())
    env.render_mode = None

    # Tunnel wrap: Pac at the left mouth moving LEFT should appear on the right.
    t = PacManEnv()
    t.reset(seed=1)
    tr = t.tunnel_row
    t.pac = (tr, 0)
    t.step(LEFT)
    assert t.pac[1] == t.W - 1, f"tunnel wrap failed: {t.pac}"
    print(f"tunnel wrap ok: ({tr},0) --LEFT--> {t.pac}")

    # Ghost-house cycle: eat a frightened ghost -> eyes -> house -> out, normal.
    t2 = PacManEnv(num_ghosts=1, frightened_steps=200, reward_ghost=10.0)
    t2.reset(seed=2)
    g = t2.ghosts[0]
    g.mode, g.frightened, g.dir = OUT, 120, None
    g.pos = (t2.pac[0], t2.pac[1] - 1)
    _, _, _, _, info = t2.step(LEFT)
    assert g.mode == EYES, f"expected eyes, got {g.mode}"
    saw_house = False
    for _ in range(400):
        _, _, term, trunc, _ = t2.step(UP)
        saw_house |= (g.mode == HOUSE)
        if g.mode == OUT and saw_house:
            break
        if term or trunc:
            break
    assert saw_house and g.mode == OUT and g.frightened == 0, \
        f"house cycle failed: saw_house={saw_house} mode={g.mode} fright={g.frightened}"
    print("ghost-house cycle ok: out -> eyes -> house -> out (re-emerged normal)")

    # Scatter/Chase actually flips.
    t3 = PacManEnv()
    t3.reset(seed=3)
    modes = set()
    for _ in range(260):
        t3.step(LEFT)
        modes.add(t3._global_mode)
    assert modes == {SCATTER, CHASE}, modes
    print(f"scatter/chase ok: observed {modes}")

    # Fruit spawns after 70 dots eaten.
    t4 = PacManEnv()
    t4.reset(seed=4)
    for p in list(t4.dots)[:70]:
        t4.dots.discard(p)
    t4._maybe_spawn_fruit()
    assert t4.fruit_active and t4._fruit_pos is not None, "fruit should spawn at 70 dots"
    print(f"fruit ok: spawns at 70 dots @ {t4._fruit_pos}, lasts {t4.fruit_steps} steps")

    # Collision accuracy (the env is tile-based: death iff Pac and a live ghost
    # share a tile, or swap tiles in one step).
    def _fresh():
        e = PacManEnv(num_ghosts=1)
        e.reset(seed=7)
        return e, e.ghosts[0]
    e, g = _fresh(); g.mode, g.frightened, g.pos = OUT, 0, e.pac
    assert e._collisions(0.0, False, None, e.pac)[1], "same tile must kill"
    e, g = _fresh(); g.mode, g.frightened, e.pac, g.pos = OUT, 0, (5, 6), (5, 5)
    assert e._collisions(0.0, False, [(5, 6)], (5, 5))[1], "tile swap must kill"
    e, g = _fresh(); g.mode, g.frightened, e.pac, g.pos = OUT, 0, (5, 6), (5, 8)
    assert not e._collisions(0.0, False, [(5, 9)], (5, 5))[1], "one tile apart must not kill"
    e, g = _fresh(); g.mode, g.frightened, e.pac, g.pos = OUT, 0, (9, 9), (5, 5)
    assert not e._collisions(0.0, False, [(5, 6)], (5, 6))[1], "ghost passing elsewhere must not kill"
    e, g = _fresh(); g.mode, g.frightened, g.pos = OUT, 5, e.pac
    _, term = e._collisions(0.0, False, None, e.pac)
    assert not term and g.mode == EYES, "frightened same tile = eaten, not death"
    e, g = _fresh(); g.mode, g.pos = EYES, e.pac
    assert not e._collisions(0.0, False, None, e.pac)[1], "eyes are harmless"
    print("collision accuracy ok: same-tile & swap kill; near-miss safe; "
          "frightened eaten; eyes harmless")

    # Lives, score, and the endless level loop (human-style options).
    L = PacManEnv(num_ghosts=1, lives=3)
    L.reset(seed=11); L.ghost_speed = 0.0
    outcomes = []
    for _ in range(3):
        g = L.ghosts[0]
        g.mode, g.frightened, g.pos, g.accum = OUT, 0, (L.pac[0], L.pac[1] + 1), 0.0
        _, _, term, _, info = L.step(RIGHT)            # Pac walks onto the ghost
        outcomes.append((info["lives"], term))
    assert outcomes == [(2, False), (1, False), (0, True)], outcomes
    S = PacManEnv(lives=3); S.reset(seed=12)
    S.score = 9990; S._add_score(50)
    assert S.lives == 4 and S.score == 10040, (S.lives, S.score)   # extra life at 10k
    S1 = PacManEnv(lives=1); S1.reset(seed=12)
    S1.score = 9990; S1._add_score(50)
    assert S1.lives == 1, "lives=1 (RL) must not gain an extra life"
    E = PacManEnv(endless=True); E.reset(seed=0)
    E.dots.clear(); E.power_pellets.clear()
    _, _, term, _, info = E.step(LEFT)
    assert not term and info["level"] == 2 and info["pellets_remaining"] == E._total_edible
    print(f"lives/score/levels ok: deaths {outcomes}; extra life at 10k (lives>1 only); "
          f"endless clear -> level {info['level']}")

    # Stability over random episodes.
    env2 = PacManEnv(frightened_steps=60, reward_ghost=10.0)
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, info = env2.reset(seed=ep)
        total = 0.0
        while True:
            obs, r, term, trunc, info = env2.step(int(rng.integers(4)))
            total += r
            if term or trunc:
                print(f"  episode {ep}: status={info['status']:8s} reward={total:8.2f} "
                      f"steps={info['steps']:4d} eaten={info['pellets_eaten']}/{info['pellets_total']}")
                break
    print("sanity check passed")
