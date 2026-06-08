"""Evaluate a trained model: clear-rate, mean reward, pellets, episode length.

    python evaluate.py --model pacman_final
    python evaluate.py --model checkpoints/pacman_2000000_steps --episodes 200

Defaults match train.py's easy baseline, so eval reflects the trained setting.
Pass --ghosts/--ghost-speed/--frightened to test on a different difficulty.
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from pacman_env import PacManEnv


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate a Pac-Man PPO model.")
    p.add_argument("--model", required=True, help="path to a saved model (.zip)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--ghosts", type=int, default=2)
    p.add_argument("--ghost-speed", type=float, default=0.5)
    p.add_argument("--frightened", type=int, default=0)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    return p


def main():
    args = build_parser().parse_args()
    env = PacManEnv(num_ghosts=args.ghosts, ghost_speed=args.ghost_speed,
                    frightened_steps=args.frightened)
    model = PPO.load(args.model, device="cpu")

    rewards, pellets, lengths, cleared = [], [], [], 0
    status_counts = {}
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            if term or trunc:
                break
        cleared += (info["status"] == "won")
        status_counts[info["status"]] = status_counts.get(info["status"], 0) + 1
        rewards.append(total)
        pellets.append(info["pellets_eaten"])
        lengths.append(info["steps"])

    n = args.episodes
    print(f"Model: {args.model}   ({n} episodes, "
          f"{'stochastic' if args.stochastic else 'deterministic'}, "
          f"{args.ghosts} ghosts @ {args.ghost_speed})")
    print(f"  clear rate     : {100 * cleared / n:5.1f}%   ({cleared}/{n})")
    print(f"  mean reward    : {np.mean(rewards):8.2f}  (std {np.std(rewards):.2f})")
    print(f"  mean pellets   : {np.mean(pellets):6.1f} / {info['pellets_total']}  "
          f"({100 * np.mean(pellets) / info['pellets_total']:.0f}%)")
    print(f"  mean ep length : {np.mean(lengths):6.1f} steps")
    print(f"  outcomes       : {status_counts}")


if __name__ == "__main__":
    main()
