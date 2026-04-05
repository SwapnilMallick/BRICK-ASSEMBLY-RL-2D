"""
train_a2c_rl.py — Train A2C on BrickEnvRL (pure RL, no pre-defined plan)
=========================================================================

The agent receives no slot positions.  It must discover where to place bricks
by exploring the action space (brick, x_bin, y_bin, theta_bin) and learning
from the wall-proximity reward signal alone.

Algorithm details
-----------------
  Update frequency : every 16 steps per env (n_steps=16), 1 epoch (on-policy)
  Policy constraint: gradient clipping (max_grad_norm=0.5)
  Optimizer        : RMSProp (lr=7e-4, default in SB3 A2C)

Usage
-----
python3 train_a2c_rl.py --spec lineSpecifications/spec1.yaml
python3 train_a2c_rl.py --spec lineSpecifications/spec3.yaml --timesteps 1000000 --n_envs 8
python3 train_a2c_rl.py --spec lineSpecifications/spec1.yaml --load_model models/a2c_rl/checkpoints/a2c_rl_brick_10000_steps

Note: A2C updates more frequently than PPO (every 16 steps vs 2048) but with
      higher variance — expect noisier training curves.
"""

import os
import csv
import time
import argparse

import numpy as np
import yaml

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from envWrapper.brick_env_rl import BrickEnvRL
from brick_anim import run_test_episode


# ---------------------------------------------------------------------------
# Rollout CSV callback
# ---------------------------------------------------------------------------

class RolloutCSVCallback(BaseCallback):
    _FIELDS = [
        "total_timesteps", "iterations",
        "ep_len_mean", "ep_rew_mean",
        "fps", "time_elapsed",
    ]

    def __init__(self, csv_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path    = csv_path
        self._iteration  = 0
        self._start_time: float | None = None
        self._file       = None
        self._writer     = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._file   = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def _on_rollout_end(self) -> None:
        self._iteration += 1
        elapsed = time.time() - self._start_time
        fps     = int(self.model.num_timesteps / elapsed) if elapsed > 0 else 0
        buf     = self.model.ep_info_buffer
        if buf:
            ep_rew_mean = float(np.mean([ep["r"] for ep in buf]))
            ep_len_mean = float(np.mean([ep["l"] for ep in buf]))
        else:
            ep_rew_mean = ep_len_mean = float("nan")

        self._writer.writerow({
            "total_timesteps": self.model.num_timesteps,
            "iterations":      self._iteration,
            "ep_len_mean":     round(ep_len_mean, 2),
            "ep_rew_mean":     round(ep_rew_mean, 4),
            "fps":             fps,
            "time_elapsed":    round(elapsed, 1),
        })
        self._file.flush()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train A2C on BrickEnvRL (pure RL — no plan)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec",       type=str, default="lineSpecifications/spec1.yaml")
    p.add_argument("--timesteps",  type=int, default=500_000,
                   help="Total environment steps")
    p.add_argument("--n_envs",     type=int, default=4)
    p.add_argument("--save_dir",   type=str, default="models/a2c_rl")
    p.add_argument("--load_model", type=str, default=None)
    p.add_argument("--eval_freq",  type=int, default=10_000)
    p.add_argument("--n_eval_eps", type=int, default=10)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- directories --------------------------------------------------------
    ckpt_dir    = os.path.join(args.save_dir, "checkpoints")
    best_dir    = os.path.join(args.save_dir, "best")
    log_dir     = os.path.join(args.save_dir, "logs")
    tb_dir      = os.path.join(args.save_dir, "tensorboard")
    rollout_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_logs")
    for d in (ckpt_dir, best_dir, log_dir, tb_dir, rollout_dir):
        os.makedirs(d, exist_ok=True)

    spec = args.spec
    with open(spec, "r") as f:
        _yaml = yaml.safe_load(f)
    spec_label  = _yaml.get("name", os.path.splitext(os.path.basename(spec))[0])
    rollout_csv = os.path.join(rollout_dir, f"a2c_rl_{spec_label}_rollout.csv")

    # ---- environments -------------------------------------------------------
    def make_env():
        return Monitor(BrickEnvRL(spec))

    train_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
    eval_env  = make_vec_env(make_env, n_envs=1,           seed=args.seed + 100)

    # ---- callbacks ----------------------------------------------------------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = best_dir,
        log_path             = log_dir,
        eval_freq            = max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes      = args.n_eval_eps,
        deterministic        = True,
        render               = False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq   = max(20_000 // args.n_envs, 1),
        save_path   = ckpt_dir,
        name_prefix = "a2c_rl_brick",
    )
    rollout_cb = RolloutCSVCallback(csv_path=rollout_csv)
    callbacks  = CallbackList([eval_cb, ckpt_cb, rollout_cb])

    # ---- model --------------------------------------------------------------
    if args.load_model:
        print(f"Resuming from: {args.load_model}")
        model = A2C.load(
            args.load_model,
            env             = train_env,
            tensorboard_log = tb_dir,
            verbose         = 1,
        )
    else:
        model = A2C(
            policy         = "MlpPolicy",
            env            = train_env,
            learning_rate  = 7e-4,       # RMSProp default learning rate
            n_steps        = 16,         # update every 16 steps per env
            gamma          = 0.99,
            gae_lambda     = 1.0,        # pure n-step returns (no GAE bias)
            ent_coef       = 0.05,       # entropy bonus for exploration
            vf_coef        = 0.5,        # value function loss weight
            max_grad_norm  = 0.5,        # gradient clipping
            use_rms_prop   = True,       # RMSProp optimizer
            rms_prop_eps   = 1e-5,
            verbose        = 1,
            seed           = args.seed,
            tensorboard_log = tb_dir,
        )

    print("=" * 60)
    print(f"  Algorithm   : A2C")
    print(f"  Mode        : Pure RL (no plan)")
    print(f"  Spec        : {spec}")
    print(f"  Action space: {train_env.action_space}")
    print(f"  n_steps/env : 16   (update frequency)")
    print(f"  Optimizer   : RMSProp  lr=7e-4")
    print(f"  Grad clip   : norm=0.5")
    print(f"  Timesteps   : {args.timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Save dir    : {args.save_dir}")
    print(f"  Rollout log : {rollout_csv}")
    print("=" * 60)

    model.learn(
        total_timesteps     = args.timesteps,
        callback            = callbacks,
        reset_num_timesteps = not bool(args.load_model),
        progress_bar        = True,
    )

    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Model saved to: {final_path}.zip")

    # ---- test episode -------------------------------------------------------
    run_test_episode(model, spec, algo_label="A2C")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
