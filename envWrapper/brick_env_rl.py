"""
BrickEnvRL — Pure RL brick placement environment (no pre-defined plan)
======================================================================

The agent discovers where to place bricks through trial and error.
No slot positions are given — the agent only sees the wall geometry
and must learn to place bricks on/near the target wall by itself.

Action space : MultiDiscrete([N, x_bins, y_bins, theta_bins])
    action[0] — brick index to pick           (0 … N-1)
    action[1] — x position bin               (maps to world x coordinate)
    action[2] — y position bin               (maps to world y coordinate)
    action[3] — orientation bin              (maps to discrete angle in degrees)

    Default discretisation:
        x step    = brick_length  →  x_bins = x_dim // brick_length
        y step    = brick_length  →  y_bins = y_dim // brick_length
        theta bins = 8            →  [0, 45, 90, 135, 180, 225, 270, 315]

Observation : flat float32 vector, all values normalised to [0, 1]
    Per brick  (4 floats): x_norm, y_norm, θ_norm, status_norm
    Per wall segment (4 floats, padded to max_segments=4):
                           start_x, start_y, end_x, end_y  (all normalised)

Reward table
    +1.2  placement on/near wall, no overlap with existing bricks
    -0.3  placement on/near wall, but overlaps an existing brick
    -0.8  placement NOT on/near the target wall
    -2.1  invalid pick (brick already placed, discarded, or out of range)

Termination
    terminated — all bricks are PLACED or DISCARDED
    truncated  — step count reaches max_steps (default 500)
"""

import os
import importlib.util

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Brick status constants
# ---------------------------------------------------------------------------
IN_INVENTORY = 0
HELD         = 1
PLACED       = 2
DISCARDED    = 3

MAX_SEGMENTS = 4     # observation is padded to this many wall segments

# ---------------------------------------------------------------------------
# Helper — load Brick2dEnvObj
# ---------------------------------------------------------------------------
def _load_inner_env():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    path = os.path.join(root, "2dEnv", "2dEnv_obj_v2.py")
    spec = importlib.util.spec_from_file_location("brick2denv_v2", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Brick2dEnvObj, mod.ACTIONS


_EnvClass, _ACTIONS = _load_inner_env()


# ---------------------------------------------------------------------------
# Geometry helper — project point onto segment
# ---------------------------------------------------------------------------
def _project_to_segment(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    t = float(np.clip(t, 0.0, 1.0))
    return ax + t * abx, ay + t * aby


# ---------------------------------------------------------------------------
# Pure RL Gymnasium environment
# ---------------------------------------------------------------------------
class BrickEnvRL(gym.Env):
    """
    Parameters
    ----------
    spec_path     : YAML spec file (only uses environment + line_segments;
                    the plan section is ignored entirely).
    render_mode   : "human" to open a Pygame window, None for headless.
    max_steps     : step budget before truncation (default 500).
    theta_bins    : number of discrete orientations (default 8 → every 45°).
    wall_tol_mult : tolerance multiplier for "on wall" check, relative to
                    brick_width.  1.5 means within 1.5×brick_width of any
                    wall segment counts as a valid placement zone.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        spec_path: str,
        render_mode: str | None = None,
        max_steps: int = 500,
        theta_bins: int = 8,
        wall_tol_mult: float = 1.5,
    ) -> None:
        super().__init__()

        self.spec_path     = spec_path
        self.render_mode   = render_mode
        self.max_steps     = max_steps
        self.theta_bins    = theta_bins
        self.wall_tol_mult = wall_tol_mult

        # ---- load YAML (plan section deliberately ignored) ------------------
        with open(spec_path, "r") as f:
            self._yaml = yaml.safe_load(f)

        env_cfg = self._yaml.get("environment", {})
        self.x_dim        = float(env_cfg.get("x_dim",        100))
        self.y_dim        = float(env_cfg.get("y_dim",        100))
        self.n_bricks     = int(env_cfg.get("total_bricks",   10))
        self.brick_length = float(env_cfg.get("brick_length",  5))
        self.brick_width  = float(env_cfg.get("brick_width",   2))

        # Wall segments (used for proximity reward, not exposed as slots)
        line_thickness = float(self._yaml.get("thickness", 1.0))
        self._line_segs = [
            {
                "start":     np.array(ls["start"], dtype=np.float32),
                "end":       np.array(ls["end"],   dtype=np.float32),
                "thickness": line_thickness,
            }
            for ls in self._yaml.get("line_segments", [])
        ]
        self._wall_tol = self.wall_tol_mult * self.brick_width

        # Which side of the wall bricks must be placed on:
        #   "positive" — cross product (B-A)×(P-A) > 0  (left of directed segment)
        #   "negative" — cross product < 0               (right of directed segment)
        #   "either"   — no side constraint (default)
        self.placement_side: str = str(self._yaml.get("placement_side", "either"))

        # ---- discretise action dimensions -----------------------------------
        self.x_step  = self.brick_length
        self.y_step  = self.brick_length
        self.x_bins  = int(self.x_dim // self.x_step)
        self.y_bins  = int(self.y_dim // self.y_step)
        self._thetas = [
            360.0 * i / self.theta_bins for i in range(self.theta_bins)
        ]   # e.g. [0, 45, 90, …, 315] for theta_bins=8

        # ---- Gymnasium spaces -----------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            [self.n_bricks, self.x_bins, self.y_bins, self.theta_bins]
        )

        # obs = N bricks × 4  +  MAX_SEGMENTS × 4
        obs_dim = self.n_bricks * 4 + MAX_SEGMENTS * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ---- runtime state --------------------------------------------------
        self._inner: object | None         = None
        self._brick_status: list[int]      = []
        self._brick_placed: list[tuple | None] = []
        self._step_count: int              = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._inner = _EnvClass(
            x_dim        = int(self.x_dim),
            y_dim        = int(self.y_dim),
            total_bricks = self.n_bricks,
            brick_length = int(self.brick_length),
            brick_width  = int(self.brick_width),
        )
        self._inner.lineFigure = self._line_segs
        self._inner.reset()

        self._brick_status = [IN_INVENTORY] * self.n_bricks
        self._brick_placed = [None]          * self.n_bricks
        self._step_count   = 0

        return self._build_obs(), {}

    def step(self, action):
        brick_idx  = int(action[0])
        x_bin      = int(action[1])
        y_bin      = int(action[2])
        theta_bin  = int(action[3])
        self._step_count += 1

        x = float(x_bin * self.x_step)
        y = float(y_bin * self.y_step)
        θ = float(self._thetas[theta_bin])

        info = {"brick_idx": brick_idx, "x": x, "y": y, "orientation": θ}

        # ---- 1. Validate pick -----------------------------------------------
        if brick_idx >= self.n_bricks or self._brick_status[brick_idx] != IN_INVENTORY:
            info["outcome"] = "invalid_pick"
            return self._build_obs(), -2.1, self._terminated(), self._truncated(), info

        # Execute PICK
        self._brick_status[brick_idx] = HELD
        self._inner_step("PICK",  brick_idx, 0.0, 0.0, 0.0)
        self._inner_step("ORIENT", brick_idx, θ,  0.0, 0.0)

        # ---- 2. Check wall proximity ----------------------------------------
        if not self._on_wall(x, y, θ):
            self._brick_status[brick_idx] = DISCARDED
            info["outcome"] = "off_wall"
            return self._build_obs(), -0.8, self._terminated(), self._truncated(), info

        # ---- 3. Check overlap -----------------------------------------------
        if not self._inner._check_valid_placement(x, y, θ):
            self._inner_step("PLACE", brick_idx, θ, x, y)
            self._brick_status[brick_idx] = DISCARDED
            info["outcome"] = "overlap"
            return self._build_obs(), -0.3, self._terminated(), self._truncated(), info

        # ---- 4. Successful placement ----------------------------------------
        self._inner_step("PLACE", brick_idx, θ, x, y)
        self._brick_status[brick_idx] = PLACED
        self._brick_placed[brick_idx] = (x, y, θ)
        info["outcome"] = "placed"
        return self._build_obs(), 1.2, self._terminated(), self._truncated(), info

    def render(self):
        if self.render_mode == "human" and self._inner is not None:
            self._inner.render(mode="human")

    def close(self):
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    # ------------------------------------------------------------------
    # Wall proximity check
    # ------------------------------------------------------------------

    def _on_wall(self, x: float, y: float, orientation: float) -> bool:
        """
        Returns True if ALL of the following hold for at least one wall segment:
          1. The brick's centre is within wall_tol of the segment.
          2. The brick's orientation is within 45° of the segment's direction
             (bricks running perpendicular to the wall are rejected).
        """
        angle_rad = np.radians(orientation)
        u = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        v = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

        # Brick centre in world coordinates
        cx = float(x + (self.brick_length / 2) * u[0] + (self.brick_width / 2) * v[0])
        cy = float(y + (self.brick_length / 2) * u[1] + (self.brick_width / 2) * v[1])

        for seg in self._line_segs:
            ax, ay = float(seg["start"][0]), float(seg["start"][1])
            bx, by = float(seg["end"][0]),   float(seg["end"][1])

            # --- proximity check ---
            # Compute unclamped t first: reject if the brick centre projects
            # beyond either endpoint (brick is off the end of the segment).
            abx, aby = bx - ax, by - ay
            ab2 = abx * abx + aby * aby
            if ab2 < 1e-12:
                continue
            t_raw = ((cx - ax) * abx + (cy - ay) * aby) / ab2
            if t_raw < 0.0 or t_raw > 1.0:
                continue  # centre projects outside segment bounds

            sx   = ax + t_raw * abx
            sy   = ay + t_raw * aby
            dist = float(np.hypot(cx - sx, cy - sy))
            if dist > self._wall_tol:
                continue

            # --- orientation alignment check ---
            seg_angle = float(np.degrees(np.arctan2(by - ay, bx - ax)) % 360)
            # allow both forward and reverse directions along the segment
            diff = min(
                abs(orientation % 360 - seg_angle),
                abs((orientation + 180) % 360 - seg_angle),
            )
            diff = min(diff, 360 - diff)   # wrap to [0, 180]
            if diff > 5.0:
                continue

            # --- side check --------------------------------------------------
            # Signed perpendicular distance: (B-A) × (P-A) / |B-A|
            # Positive = left of directed segment, negative = right
            seg_len = float(np.hypot(bx - ax, by - ay))
            if seg_len > 1e-12 and self.placement_side != "either":
                signed = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / seg_len
                if self.placement_side == "positive" and signed < -0.1:
                    continue   # brick is on the wrong side
                if self.placement_side == "negative" and signed > 0.1:
                    continue   # brick is on the wrong side

            return True

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inner_step(self, op, brick_id, orientation, x, y):
        self._inner.step({
            "operation":   _ACTIONS[op],
            "brickID":     brick_id,
            "orientation": np.array([orientation], dtype=np.float32),
            "x":           np.array([x],           dtype=np.float32),
            "y":           np.array([y],            dtype=np.float32),
        })

    def _build_obs(self) -> np.ndarray:
        parts: list[float] = []

        # Per-brick: [x_norm, y_norm, θ_norm, status_norm]
        for i in range(self.n_bricks):
            pos    = self._brick_placed[i]
            status = self._brick_status[i]
            if pos is not None and status == PLACED:
                x_n = pos[0] / self.x_dim
                y_n = pos[1] / self.y_dim
                θ_n = pos[2] / 360.0
            else:
                x_n = y_n = θ_n = 0.0
            parts += [x_n, y_n, θ_n, status / 3.0]

        # Wall segments (padded to MAX_SEGMENTS)
        for i in range(MAX_SEGMENTS):
            if i < len(self._line_segs):
                seg = self._line_segs[i]
                parts += [
                    float(seg["start"][0]) / self.x_dim,
                    float(seg["start"][1]) / self.y_dim,
                    float(seg["end"][0])   / self.x_dim,
                    float(seg["end"][1])   / self.y_dim,
                ]
            else:
                parts += [0.0, 0.0, 0.0, 0.0]   # padding

        return np.array(parts, dtype=np.float32)

    def _terminated(self) -> bool:
        return all(s in (PLACED, DISCARDED) for s in self._brick_status)

    def _truncated(self) -> bool:
        return self._step_count >= self.max_steps
