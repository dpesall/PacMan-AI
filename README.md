# Pac-Man RL

A faithful arcade Pac-Man clone and (eventually) a reinforcement-learning agent
that clears the board. This stage is the **playable game** — a Gymnasium
environment plus a pygame viewer you can play yourself — built to match the
original arcade game before any training happens.

## Files

| File | Purpose |
|------|---------|
| `pacman_env.py` | The game, as a Gymnasium env. Tile-based, `(7, H, W)` state observation, clear-the-board reward, authentic ghost AI. |
| `arcade_maze.py` | The exact 28×31 arcade board (240 dots + 4 energizers), generated from the original tile map. |
| `viewer.py` | Pygame viewer: human-playable, agent playback, or random policy. Smooth interpolated movement. |
| `train.py` | *(not built yet)* PPO training. |
| `evaluate.py` | *(not built yet)* clear-rate / metrics. |

## Setup (Windows)

Python 3.12 is installed under your user profile. Create a venv and install the
game dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(The training stack — PyTorch via stable-baselines3 — is a separate, heavier
install in `requirements-train.txt`, for later.)

## Play it

```powershell
.\.venv\Scripts\python.exe viewer.py --human
```

- **Move:** arrow keys or WASD (turns are buffered — press toward a corner early).
- **R:** restart  **Esc / Q:** quit
- Eat all **240 dots + 4 energizers** to clear the board and advance a level.
  You have **3 lives**; a ghost takes one (watch Pac's death animation).
- Grab an **energizer** (big blinking dot in a corner) to turn ghosts blue and
  edible. Eat one and it flees home as **eyes**, regenerates in the ghost house,
  and returns as a normal ghost — not edible again until your next energizer.
- Use the **side tunnel** (middle row) to wrap left↔right and escape.
- Grab the **bonus cherry** that appears below the house after 70 and 170 dots.
- **Score & high score** show in the HUD (dots 10, energizer 50, ghosts
  200→1600, cherry 100); extra life at 10,000. The high score persists between
  sessions. **Sound** is on by default (`--no-sound` to mute).

Watch the built-in random policy instead (no training needed):

```powershell
.\.venv\Scripts\python.exe viewer.py
```

### Useful options

| Flag | Default | Meaning |
|------|---------|---------|
| `--speed` | 7 | logical moves per second (~arcade tiles/sec) |
| `--ghosts` | 4 | number of ghosts (Blinky, Pinky, Inky, Clyde) |
| `--ghost-speed` | 0.75 | ghost speed as a fraction of Pac (0.9+ ≈ arcade level 1) |
| `--frightened` | 45 | frightened steps after an energizer (~6 s; 0 disables) |
| `--cell` | 24 | pixels per tile |
| `--no-sound` | off | mute the synthesized sound |
| `--model PATH` | – | watch a trained model play |

## How it matches the arcade

- **Exact 28×31 board:** 240 dots + 4 energizers, the real maze, and a side
  **tunnel** that wraps left↔right (ghosts slow down inside it).
- **Four ghost personalities** (the big one):
  - **Blinky** (red) chases Pac directly.
  - **Pinky** (pink) aims 4 tiles ahead of Pac.
  - **Inky** (cyan) uses a vector doubled from Blinky through 2 tiles ahead of Pac.
  - **Clyde** (orange) chases when far but peels off to his corner when close.
- **Scatter / Chase** alternation: ghosts periodically retreat to their home
  corners instead of relentlessly chasing (and reverse direction on each switch).
- **Authentic ghost house:** ghosts emerge from the central pen; an eaten ghost
  returns as eyes, regenerates, and comes back out *normal* — never frightened.
- **Authentic speeds:** ghosts are a bit slower than Pac, much slower when
  frightened or in the tunnel, fast as eyes; **Blinky speeds up** as the dots run
  low (Cruise Elroy).
- **Bonus fruit:** a cherry (100 pts) appears below the house after 70 and 170
  dots, for ~9 seconds.

## Game design (locked-in for RL)

- **State observation, not pixels.** A 7-channel grid: walls, dots, energizers,
  Pac-Man, ghosts, frightened ghosts, bonus fruit.
- **Discrete actions:** UP / DOWN / LEFT / RIGHT. A blocked move stays put.
- **Reward:** dot `+1`, clear-all `+100`, death `-50`, per-step `-0.01`.
  (Eating ghosts and fruit are configurable; `0` for the pure board-clearing
  objective.)
- **One decision per tile.** Sub-tile speeds are modeled by ghosts moving a
  fraction of the time; Pac is the 100%-speed reference (1 step = 1 Pac move).
- **Difficulty is parameterized** via constructor knobs (`num_ghosts`,
  `ghost_speed`, `frightened_steps`, …) rather than hard-coded levels.

## Status & roadmap

✅ Playable, arcade-faithful game (env + viewer).
Next up: `train.py` (PPO) and `evaluate.py` (clear-rate metrics), then reward
shaping and difficulty/generalization.
