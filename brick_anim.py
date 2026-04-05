"""
brick_anim.py — Animated test-episode renderer (shared by PPO and A2C scripts)
===============================================================================

Valid placement:
    Brick slides from the inventory corner to its target position, then
    settles (stays as a red placed brick).

Invalid placement (overlap / off-wall / invalid-pick):
    Brick slides to the target, holds briefly, turns black, blinks twice,
    then disappears.
"""

import os
from datetime import datetime
from typing import Any

import numpy as np
import pygame
import imageio
import yaml

from envWrapper.brick_env_rl import BrickEnvRL, PLACED, DISCARDED

# ---------------------------------------------------------------------------
# Animation timing  (frames at 30 fps)
# ---------------------------------------------------------------------------
_FPS          = 30
_MOVE_FRAMES  = 20   # frames for the slide from inventory → target
_ARRIVE_HOLD  = 8    # frames at target (red) before turning black (invalid)
_BLACK_FRAMES = 8    # frames shown as solid black before blinking
_BLINK_OFF    = 6    # frames invisible  per blink half-cycle
_BLINK_ON     = 6    # frames black      per blink half-cycle
_BLINK_CYCLES = 2    # number of blink cycles
_VANISH_HOLD  = 8    # frames after final disappearance
_SETTLE_HOLD  = 20   # frames holding a freshly placed (valid) brick

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_COL_BRICK   = (200,  50,   0)
_COL_BORDER  = (100,  25,   0)
_COL_INVALID = ( 20,  20,  20)
_COL_INV_B   = (  0,   0,   0)

# Pixel-space starting point for brick animation (inventory top-left corner)
_INV_X = 5.0
_INV_Y = 5.0


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------

def _draw_brick_poly(
    screen: pygame.Surface,
    x_px: float, y_px: float, angle_deg: float,
    length_px: float, width_px: float,
    color: tuple, border: tuple,
) -> None:
    """Draw a single rotated brick rectangle onto *screen*."""
    r = np.radians(angle_deg)
    u = np.array([np.cos(r), np.sin(r)], dtype=np.float32)
    v = np.array([-np.sin(r), np.cos(r)], dtype=np.float32)
    a = np.array([x_px, y_px], dtype=np.float32)
    b = a + length_px * u
    c = b + width_px  * v
    d = a + width_px  * v
    verts = [(float(p[0]), float(p[1])) for p in (a, b, c, d)]
    pygame.draw.polygon(screen, color, verts)
    pygame.draw.polygon(screen, border, verts, 2)


def _draw_placed_bricks(screen: pygame.Surface, placed_visuals: list) -> None:
    for pv in placed_visuals:
        _draw_brick_poly(
            screen,
            pv["x"], pv["y"], pv["orientation"],
            pv["length"], pv["width"],
            _COL_BRICK, _COL_BORDER,
        )


def _capture_frame(screen: pygame.Surface) -> np.ndarray:
    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))


def _redraw_base(inner) -> None:
    """Repaint the static scene layers without calling pygame.display.flip()."""
    pygame.event.pump()          # keep the OS window responsive
    inner._draw_background()
    inner._draw_grid()
    inner._draw_wall_segments()
    inner._draw_inventory()
    inner._draw_hud()


def _flush(screen: pygame.Surface, writer) -> None:
    pygame.display.flip()
    if writer:
        writer.append_data(_capture_frame(screen))


def _lerp_angle(a0: float, a1: float, t: float) -> float:
    """Shortest-path lerp between two angles (degrees)."""
    diff = (a1 - a0 + 180.0) % 360.0 - 180.0
    return a0 + t * diff


# ---------------------------------------------------------------------------
# Composite animation sequences
# ---------------------------------------------------------------------------

def _animate_move(
    screen, writer, inner, placed_visuals,
    sx: float, sy: float, sa: float,
    ex: float, ey: float, ea: float,
    length_px: float, width_px: float,
    color: tuple, border: tuple,
    n_frames: int = _MOVE_FRAMES,
) -> None:
    """Slide a brick from (sx, sy, sa°) to (ex, ey, ea°) over *n_frames* frames.

    *placed_visuals* should NOT contain the brick being animated — it will be
    drawn separately at each intermediate position.
    """
    for i in range(n_frames):
        t  = i / max(n_frames - 1, 1)
        px = sx + t * (ex - sx)
        py = sy + t * (ey - sy)
        pa = _lerp_angle(sa, ea, t)
        _redraw_base(inner)
        _draw_placed_bricks(screen, placed_visuals)
        _draw_brick_poly(screen, px, py, pa, length_px, width_px, color, border)
        _flush(screen, writer)


def _animate_invalid(
    screen, writer, inner, placed_visuals,
    sx: float, sy: float,
    ex: float, ey: float, ea: float,
    length_px: float, width_px: float,
) -> None:
    """
    Full invalid-brick animation:
      1. Slide from inventory → target (red)
      2. Hold briefly at target (red)
      3. Turn black and hold
      4. Blink twice (black / invisible)
      5. Disappear
    """
    # 1 — travel
    _animate_move(
        screen, writer, inner, placed_visuals,
        sx, sy, 0.0, ex, ey, ea,
        length_px, width_px, _COL_BRICK, _COL_BORDER,
    )

    # 2 — arrive hold (red)
    for _ in range(_ARRIVE_HOLD):
        _redraw_base(inner)
        _draw_placed_bricks(screen, placed_visuals)
        _draw_brick_poly(screen, ex, ey, ea, length_px, width_px,
                         _COL_BRICK, _COL_BORDER)
        _flush(screen, writer)

    # 3 — turn black
    for _ in range(_BLACK_FRAMES):
        _redraw_base(inner)
        _draw_placed_bricks(screen, placed_visuals)
        _draw_brick_poly(screen, ex, ey, ea, length_px, width_px,
                         _COL_INVALID, _COL_INV_B)
        _flush(screen, writer)

    # 4 — blink (invisible → black, repeated)
    for _ in range(_BLINK_CYCLES):
        for _ in range(_BLINK_OFF):        # invisible
            _redraw_base(inner)
            _draw_placed_bricks(screen, placed_visuals)
            _flush(screen, writer)
        for _ in range(_BLINK_ON):         # black
            _redraw_base(inner)
            _draw_placed_bricks(screen, placed_visuals)
            _draw_brick_poly(screen, ex, ey, ea, length_px, width_px,
                             _COL_INVALID, _COL_INV_B)
            _flush(screen, writer)

    # 5 — disappear
    for _ in range(_VANISH_HOLD):
        _redraw_base(inner)
        _draw_placed_bricks(screen, placed_visuals)
        _flush(screen, writer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_test_episode(model: Any, spec_path: str, algo_label: str = "RL") -> None:
    """
    Run one deterministic test episode with animated brick placement.

    Parameters
    ----------
    model      : trained SB3 model (PPO, A2C, …)
    spec_path  : path to the YAML wall specification
    algo_label : short name shown in the log header and used as filename prefix
                 (e.g. "PPO", "A2C")
    """
    with open(spec_path, "r") as f:
        spec_yaml = yaml.safe_load(f)
    spec_label = spec_yaml.get("name", os.path.splitext(os.path.basename(spec_path))[0])
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir   = os.path.dirname(os.path.abspath(__file__))
    algo_slug  = algo_label.lower()

    videos_dir = os.path.join(root_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{algo_slug}_rl_{spec_label}_{timestamp}.mp4")

    logs_dir  = os.path.join(root_dir, "test_logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path  = os.path.join(logs_dir, f"{algo_slug}_rl_{spec_label}.txt")

    _lines: list[str] = []

    def _log(text: str = "") -> None:
        print(text)
        _lines.append(text)

    # ---- Environment -------------------------------------------------------
    env = BrickEnvRL(spec_path, render_mode="human")
    obs, _ = env.reset()
    env.render()                          # opens pygame window + initial draw

    inner     = env._inner
    px        = inner.pixel_scale
    length_px = inner.brick_length * px
    width_px  = inner.brick_width  * px

    # ---- Video writer ------------------------------------------------------
    writer = None
    try:
        writer = imageio.get_writer(video_path, fps=_FPS)
        print(f"Recording video to: {video_path}")
    except Exception as e:
        print(f"Warning: could not create video writer — {e}")

    # ---- Initial frame -----------------------------------------------------
    if inner.screen:
        _redraw_base(inner)
        _flush(inner.screen, writer)

    placed_visuals: list[dict] = []
    step         = 0
    total_reward = 0.0
    done         = False

    _log("\n" + "=" * 60)
    _log(f"  TEST EPISODE — pure RL / {algo_label} (deterministic policy)")
    _log("=" * 60)
    _log(f"  Spec  : {spec_path}")
    _log(f"  Bricks: {env.n_bricks}   Action space: {env.action_space}")
    _log("-" * 60)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        brick_idx = info["brick_idx"]
        x, y, θ   = info["x"], info["y"], info["orientation"]
        outcome   = info["outcome"]
        reward    = float(reward)
        total_reward += reward

        outcome_symbol = {
            "placed":       "✓",
            "overlap":      "✗ overlap",
            "off_wall":     "✗ off wall",
            "invalid_pick": "✗ invalid pick",
        }.get(outcome, outcome)

        _log(
            f"  Step {step + 1:>3} │ brick {brick_idx} → "
            f"(x={x:.1f}, y={y:.1f}, θ={θ:.1f}°)"
            f"\n           │ outcome: {outcome_symbol}   reward: {reward:+.1f}"
        )

        # Target in pixel space
        ex, ey = x * px, y * px

        if inner.screen:
            if outcome == "placed":
                # Animate slide (brick NOT yet in placed_visuals → no ghost at target)
                _animate_move(
                    inner.screen, writer, inner, placed_visuals,
                    _INV_X, _INV_Y, 0.0,
                    ex, ey, θ,
                    length_px, width_px,
                    _COL_BRICK, _COL_BORDER,
                )
                # Brick has arrived — add to placed_visuals and hold
                placed_visuals.append({
                    "x": ex, "y": ey, "orientation": θ,
                    "length": length_px, "width": width_px,
                })
                for _ in range(_SETTLE_HOLD):
                    _redraw_base(inner)
                    _draw_placed_bricks(inner.screen, placed_visuals)
                    _flush(inner.screen, writer)
            else:
                _animate_invalid(
                    inner.screen, writer, inner, placed_visuals,
                    _INV_X, _INV_Y,
                    ex, ey, θ,
                    length_px, width_px,
                )
        else:
            # Headless: track placed bricks for the log only
            if outcome == "placed":
                placed_visuals.append({
                    "x": ex, "y": ey, "orientation": θ,
                    "length": length_px, "width": width_px,
                })

        step += 1

    # ---- Final hold --------------------------------------------------------
    if writer and inner.screen:
        _redraw_base(inner)
        _draw_placed_bricks(inner.screen, placed_visuals)
        _flush(inner.screen, writer)
        frame = _capture_frame(inner.screen)
        for _ in range(150):
            writer.append_data(frame)

    placed_n    = sum(1 for s in env._brick_status if s == PLACED)
    discarded_n = sum(1 for s in env._brick_status if s == DISCARDED)

    _log("-" * 60)
    _log(f"  Episode finished in {step} steps")
    _log(f"  Total reward : {total_reward:+.3f}")
    _log(f"  Placed: {placed_n}   Discarded: {discarded_n}   "
         f"In inventory: {env.n_bricks - placed_n - discarded_n}")
    _log("=" * 60)

    if writer:
        try:
            writer.close()
            print(f"Video saved to : {video_path}")
        except Exception as e:
            print(f"Warning: could not save video — {e}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"Log saved to   : {log_path}")

    env.close()
