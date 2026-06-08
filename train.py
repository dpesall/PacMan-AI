"""Train a PPO agent to clear the Pac-Man board.

CPU is the right device: the policy is a small MLP, so the bottleneck is
environment throughput, which we parallelize across cores with SubprocVecEnv.
The first run uses an *easy* difficulty (2 slow ghosts, no frightened mode) per
the project plan -- master the basics first, then raise difficulty later.

    python train.py                     # 2M steps, easy difficulty
    python train.py --steps 5000000     # longer
    python train.py --ghosts 4 --ghost-speed 0.75   # full difficulty
    tensorboard --logdir tb_logs        # watch it learn (separate terminal)

Writes checkpoints/<name>_<N>_steps.zip periodically and <name>_final.zip at the
end.
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from pacman_env import PacManEnv


def build_parser():
    p = argparse.ArgumentParser(description="Train PPO on Pac-Man.")
    p.add_argument("--steps", type=int, default=2_000_000, help="total timesteps")
    p.add_argument("--n-envs", type=int, default=8, help="parallel envs")
    p.add_argument("--ghosts", type=int, default=2, help="number of ghosts")
    p.add_argument("--ghost-speed", type=float, default=0.5, help="ghost speed (fraction of Pac)")
    p.add_argument("--frightened", type=int, default=0, help="frightened steps (0 = pure board clearing)")
    p.add_argument("--save-every", type=int, default=250_000, help="checkpoint interval (env steps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="pacman", help="run / checkpoint name")
    p.add_argument("--subproc", dest="subproc", action="store_true", default=True,
                   help="use SubprocVecEnv (parallel processes)")
    p.add_argument("--no-subproc", dest="subproc", action="store_false",
                   help="use a single-process DummyVecEnv (fallback)")
    return p


def main():
    args = build_parser().parse_args()
    env_kwargs = dict(num_ghosts=args.ghosts, ghost_speed=args.ghost_speed,
                      frightened_steps=args.frightened)
    vec_cls = SubprocVecEnv if args.subproc else None
    vec = make_vec_env(PacManEnv, n_envs=args.n_envs, seed=args.seed,
                       env_kwargs=env_kwargs, vec_env_cls=vec_cls)

    model = PPO(
        "MlpPolicy", vec, device="cpu", seed=args.seed, verbose=1,
        n_steps=1024, batch_size=512, n_epochs=4,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        tensorboard_log="tb_logs",
    )

    ckpt = CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path="checkpoints", name_prefix=args.name)

    print(f"Training {args.steps:,} steps | {args.n_envs} envs | "
          f"{args.ghosts} ghosts @ {args.ghost_speed} | frightened={args.frightened} | device=cpu")
    model.learn(total_timesteps=args.steps, callback=ckpt, tb_log_name=args.name)
    model.save(f"{args.name}_final")
    print(f"Done. Saved {args.name}_final.zip and checkpoints/.")


if __name__ == "__main__":
    main()
