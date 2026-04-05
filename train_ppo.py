"""
train_ppo.py — Train a PPO agent on the BrickPlacementEnv
==========================================================

Usage examples
--------------
# Train on the horizontal wall spec for 200 k steps
python3 train_ppo.py --spec lineSpecifications/spec1.yaml

# Train on the L-shaped wall, more timesteps, 8 parallel envs
python3 train_ppo.py --spec lineSpecifications/spec3.yaml --timesteps 500000 --n_envs 8

# Resume from a saved checkpoint
python3 train_ppo.py --spec lineSpecifications/spec1.yaml --load_model models/ppo/checkpoints/ppo_brick_10000_steps

Outputs
-------
models/ppo/best/           — best model found during evaluation
models/ppo/checkpoints/    — periodic checkpoints
models/ppo/final_model.zip — model at end of training
models/ppo/tensorboard/    — TensorBoard logs  (tensorboard --logdir models/ppo/tensorboard)
"""

import os
import csv
import time
import argparse
from datetime import datetime

import numpy as np
import pygame
import imageio
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from envWrapper.brick_env_wrapper import BrickPlacementEnv


# ---------------------------------------------------------------------------
# Rollout CSV callback
# ---------------------------------------------------------------------------

class RolloutCSVCallback(BaseCallback):
    """
    Appends one row to a CSV file at the end of every rollout.

    Columns: total_timesteps, iterations, ep_len_mean, ep_rew_mean,
             fps, time_elapsed
    """

    _FIELDS = [
        "total_timesteps", "iterations",
        "ep_len_mean", "ep_rew_mean",
        "fps", "time_elapsed",
    ]

    def __init__(self, csv_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path   = csv_path
        self._iteration = 0
        self._start_time: float | None = None
        self._file      = None
        self._writer    = None

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

        buf = self.model.ep_info_buffer
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
        description="Train PPO on the 2-D Bricklaying environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec",       type=str, default="lineSpecifications/spec1.yaml",
                   help="Path to the YAML spec file")
    p.add_argument("--timesteps",  type=int, default=100_000,
                   help="Total environment steps to train for")
    p.add_argument("--n_envs",     type=int, default=4,
                   help="Number of parallel training environments")
    p.add_argument("--save_dir",   type=str, default="models/ppo",
                   help="Root directory for saved models and logs")
    p.add_argument("--load_model", type=str, default=None,
                   help="Path to a saved model to resume training from")
    p.add_argument("--eval_freq",  type=int, default=5_000,
                   help="Evaluate every N steps (per environment)")
    p.add_argument("--n_eval_eps", type=int, default=10,
                   help="Number of episodes per evaluation")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- directories --------------------------------------------------------
    ckpt_dir     = os.path.join(args.save_dir, "checkpoints")
    best_dir     = os.path.join(args.save_dir, "best")
    log_dir      = os.path.join(args.save_dir, "logs")
    tb_dir       = os.path.join(args.save_dir, "tensorboard")
    rollout_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_logs")
    for d in (ckpt_dir, best_dir, log_dir, tb_dir, rollout_dir):
        os.makedirs(d, exist_ok=True)

    spec = args.spec

    # derive spec label for filenames (e.g. "horizontal_wall")
    with open(spec, "r") as f:
        _yaml = yaml.safe_load(f)
    spec_label = _yaml.get("name", os.path.splitext(os.path.basename(spec))[0])
    rollout_csv = os.path.join(rollout_dir, f"ppo_{spec_label}_rollout.csv")

    # ---- environments -------------------------------------------------------
    def make_env():
        return Monitor(BrickPlacementEnv(spec))

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
        save_freq  = max(10_000 // args.n_envs, 1),
        save_path  = ckpt_dir,
        name_prefix= "ppo_brick",
    )
    rollout_cb = RolloutCSVCallback(csv_path=rollout_csv)
    print(f"Rollout log   : {rollout_csv}")
    callbacks = CallbackList([eval_cb, ckpt_cb, rollout_cb])

    # ---- model --------------------------------------------------------------
    if args.load_model:
        print(f"Resuming from: {args.load_model}")
        model = PPO.load(
            args.load_model,
            env              = train_env,
            tensorboard_log  = tb_dir,
            verbose          = 1,
        )
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            learning_rate   = 3e-4,
            n_steps         = 2048,      # steps per env before each update
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,      # small entropy bonus encourages exploration
            verbose         = 1,
            seed            = args.seed,
            tensorboard_log = tb_dir,
        )

    # ---- training -----------------------------------------------------------
    print("=" * 60)
    print(f"  Spec        : {spec}")
    print(f"  Timesteps   : {args.timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Save dir    : {args.save_dir}")
    print("=" * 60)

    model.learn(
        total_timesteps  = args.timesteps,
        callback         = callbacks,
        reset_num_timesteps = not bool(args.load_model),
        progress_bar     = True,
    )

    # ---- save final model ---------------------------------------------------
    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}.zip")

    # ---- quick evaluation ---------------------------------------------------
    print("\nRunning final evaluation (20 episodes) …")
    obs = eval_env.reset()
    n_eps, total_reward, successes = 0, 0.0, 0
    ep_reward = 0.0

    while n_eps < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        ep_reward += float(reward[0])
        if done[0]:
            total_reward += ep_reward
            if info[0].get("outcome") != "invalid_pick":
                successes += 1
            ep_reward = 0.0
            n_eps += 1
            obs = eval_env.reset()

    print(f"  Mean episode reward : {total_reward / n_eps:.3f}")
    print(f"  Episodes with ≥1 successful placement: {successes}/{n_eps}")

    train_env.close()
    eval_env.close()

    # ---- traced test episode ------------------------------------------------
    run_test_episode(model, spec)


def _draw_placed_bricks(screen, placed_visuals: list) -> None:
    """Draw all successfully placed bricks as rotated polygons."""
    for pv in placed_visuals:
        angle_rad = np.radians(pv["orientation"])
        u = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
        v = np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)
        a = np.array([pv["x"], pv["y"]], dtype=np.float32)
        b = a + pv["length"] * u
        c = b + pv["width"]  * v
        d = a + pv["width"]  * v
        verts = [(float(p[0]), float(p[1])) for p in (a, b, c, d)]
        pygame.draw.polygon(screen, (200, 50, 0), verts)
        pygame.draw.polygon(screen, (100, 25, 0), verts, 2)


def _capture_frame(screen) -> np.ndarray:
    frame = pygame.surfarray.array3d(screen)
    return np.transpose(frame, (1, 0, 2))


def run_test_episode(model: PPO, spec_path: str) -> None:
    """Run one deterministic episode, print a step-by-step trace, and save an MP4."""

    # ---- derive filenames from spec 'name' field ----------------------------
    with open(spec_path, "r") as f:
        spec_yaml = yaml.safe_load(f)
    spec_label = spec_yaml.get("name", os.path.splitext(os.path.basename(spec_path))[0])
    # e.g. "horizontal_wall" -> "horizontal_wall"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir   = os.path.dirname(os.path.abspath(__file__))

    videos_dir = os.path.join(root_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"ppo_{spec_label}_{timestamp}.mp4")

    logs_dir   = os.path.join(root_dir, "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path   = os.path.join(logs_dir, f"ppo_{spec_label}.txt")

    # Buffer lines so they can be written to the log file at the end
    _lines: list[str] = []

    def _log(text: str = "") -> None:
        print(text)
        _lines.append(text)

    # ---- set up environment with rendering ----------------------------------
    env = BrickPlacementEnv(spec_path, render_mode="human")
    obs, _ = env.reset()
    env.render()   # opens pygame window and creates screen

    # ---- set up video writer ------------------------------------------------
    writer = None
    try:
        writer = imageio.get_writer(video_path, fps=30)
        print(f"\nRecording video to: {video_path}")
    except Exception as e:
        print(f"Warning: could not create video writer — {e}")

    _log("\n" + "=" * 60)
    _log("TEST EPISODE")
    _log("=" * 60)
    _log(f"  Spec  : {spec_path}")
    _log(f"  Bricks: {env.n_bricks}   Slots: {env.n_slots}")
    _log("-" * 60)

    inner         = env._inner
    px            = inner.pixel_scale
    placed_visuals: list[dict] = []
    step          = 0
    total_reward  = 0.0
    done          = False

    # Capture initial frame (empty board)
    if writer and inner.screen:
        writer.append_data(_capture_frame(inner.screen))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        brick_idx = info["brick_idx"]
        slot_idx  = info["slot_idx"]
        outcome   = info["outcome"]
        reward    = float(reward)
        total_reward += reward

        # Slot details
        if slot_idx < env.n_slots:
            slot = env.slots[slot_idx]
            slot_str = (f"slot {slot_idx}  "
                        f"(x={slot['x']:.2f}, y={slot['y']:.2f}, θ={slot['orientation']:.1f}°)")
        else:
            slot_str = f"slot {slot_idx}  (out of range)"

        outcome_symbol = {
            "placed":       "✓",
            "overlap":      "✗ overlap",
            "bad_slot":     "✗ bad slot",
            "invalid_pick": "✗ invalid pick",
        }.get(outcome, outcome)

        _log(
            f"  Step {step + 1:>3} │ brick {brick_idx} → {slot_str}"
            f"\n           │ outcome: {outcome_symbol}   reward: {reward:+.1f}"
        )

        # Register visual for successful placements
        if outcome == "placed":
            placed_visuals.append({
                "x":           slot["x"] * px,
                "y":           slot["y"] * px,
                "length":      inner.brick_length * px,
                "width":       inner.brick_width  * px,
                "orientation": slot["orientation"],
            })

        # Render frame: base env + all placed bricks
        env.render()
        if inner.screen:
            _draw_placed_bricks(inner.screen, placed_visuals)
            pygame.display.flip()
            if writer:
                # Write several frames per step so the video is watchable
                frame = _capture_frame(inner.screen)
                for _ in range(90):
                    writer.append_data(frame)

        step += 1

    # Hold final frame for 5 seconds
    if writer and inner.screen:
        frame = _capture_frame(inner.screen)
        for _ in range(150):
            writer.append_data(frame)

    _log("-" * 60)
    _log(f"  Episode finished in {step} steps")
    _log(f"  Total reward : {total_reward:+.3f}")
    placed_n    = sum(1 for s in env._brick_status if s == 2)
    discarded_n = sum(1 for s in env._brick_status if s == 3)
    _log(f"  Placed: {placed_n}   Discarded: {discarded_n}   In inventory: {env.n_bricks - placed_n - discarded_n}")
    _log("=" * 60)

    if writer:
        try:
            writer.close()
            print(f"Video saved to: {video_path}")
        except Exception as e:
            print(f"Warning: could not save video — {e}")

    # ---- write log file -----------------------------------------------------
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"Log saved to  : {log_path}")

    env.close()


if __name__ == "__main__":
    main()
