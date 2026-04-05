"""
BrickPlacementEnv — Gymnasium wrapper around Brick2dEnvObj
==========================================================

Action space : MultiDiscrete([N, S])
    action[0] — brick index  (0 … N-1)
    action[1] — slot index   (0 … S-1)

    The orientation for ORIENT is taken directly from the chosen slot so the
    agent only needs to decide *which brick* goes to *which slot*.

Observation  : flat float32 vector, all values normalised to [0, 1]
    Per brick  (4 floats each): x_norm, y_norm, θ_norm, status_norm
    Per slot   (4 floats each): x_norm, y_norm, θ_norm, occupied

Reward table
    +1.2  successful placement
    -0.3  valid pick, slot valid, but placement overlaps existing brick
    -0.8  valid pick, bad/occupied slot
    -2.1  invalid pick (brick already placed / discarded / out of range)

Termination
    terminated  — all bricks are PLACED or DISCARDED
    truncated   — step count reaches max_steps (default 200)
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

# ---------------------------------------------------------------------------
# Helper — load Brick2dEnvObj from the sibling 2dEnv package
# ---------------------------------------------------------------------------
def _load_inner_env():
    here  = os.path.dirname(os.path.abspath(__file__))
    root  = os.path.dirname(here)
    path  = os.path.join(root, "2dEnv", "2dEnv_obj_v2.py")
    spec  = importlib.util.spec_from_file_location("brick2denv_v2", path)
    mod   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Brick2dEnvObj, mod.ACTIONS


_EnvClass, _ACTIONS = _load_inner_env()


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
class BrickPlacementEnv(gym.Env):
    """
    Parameters
    ----------
    spec_path   : path to a YAML spec file that contains both
                  ``environment`` settings and a ``plan`` list of slots.
    render_mode : ``"human"`` to open a Pygame window, ``None`` for headless.
    max_steps   : step budget before truncation (default 200).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        spec_path: str,
        render_mode: str | None = None,
        max_steps: int = 200,
    ) -> None:
        super().__init__()

        self.spec_path   = spec_path
        self.render_mode = render_mode
        self.max_steps   = max_steps

        # ---- load YAML -------------------------------------------------------
        with open(spec_path, "r") as f:
            self._yaml = yaml.safe_load(f)

        env_cfg = self._yaml.get("environment", {})
        self.x_dim        = float(env_cfg.get("x_dim",        100))
        self.y_dim        = float(env_cfg.get("y_dim",        100))
        self.n_bricks     = int(env_cfg.get("total_bricks",   10))
        self.brick_length = float(env_cfg.get("brick_length",  5))
        self.brick_width  = float(env_cfg.get("brick_width",   2))

        plan = self._yaml.get("plan", [])
        if not plan:
            raise ValueError(f"Spec file has no 'plan' section: {spec_path}")

        # Slots sorted by brickID so indices are stable
        self.slots: list[dict] = [
            {
                "x":           float(e["x"]),
                "y":           float(e["y"]),
                "orientation": float(e["orientation"]),
            }
            for e in sorted(plan, key=lambda e: e["brickID"])
        ]
        self.n_slots = len(self.slots)

        # ---- Gymnasium spaces ------------------------------------------------
        self.action_space = spaces.MultiDiscrete([self.n_bricks, self.n_slots])

        # obs = N bricks × 4 + S slots × 4
        obs_dim = self.n_bricks * 4 + self.n_slots * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ---- runtime state (initialised in reset) ----------------------------
        self._inner: object | None = None
        self._brick_status: list[int]               = []
        self._brick_placed_pos: list[tuple | None]  = []
        self._slot_occupied: list[bool]             = []
        self._step_count: int                       = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Build fresh inner env
        self._inner = _EnvClass(
            x_dim        = int(self.x_dim),
            y_dim        = int(self.y_dim),
            total_bricks = self.n_bricks,
            brick_length = int(self.brick_length),
            brick_width  = int(self.brick_width),
        )

        line_thickness = self._yaml.get("thickness", 1.0)
        self._inner.lineFigure = [
            {
                "start":     np.array(ls["start"], dtype=np.float32),
                "end":       np.array(ls["end"],   dtype=np.float32),
                "thickness": float(line_thickness),
            }
            for ls in self._yaml.get("line_segments", [])
        ]
        self._inner.reset()

        # Reset bookkeeping
        self._brick_status      = [IN_INVENTORY] * self.n_bricks
        self._brick_placed_pos  = [None]          * self.n_bricks
        self._slot_occupied     = [False]         * self.n_slots
        self._step_count        = 0

        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        brick_idx = int(action[0])
        slot_idx  = int(action[1])
        self._step_count += 1

        info = {"brick_idx": brick_idx, "slot_idx": slot_idx}

        # ---- 1. Validate PICK -----------------------------------------------
        if brick_idx >= self.n_bricks or self._brick_status[brick_idx] != IN_INVENTORY:
            info["outcome"] = "invalid_pick"
            return self._build_obs(), -2.1, self._terminated(), self._truncated(), info

        # Execute PICK
        self._brick_status[brick_idx] = HELD
        self._inner_step("PICK", brick_idx, 0.0, 0.0, 0.0)

        # ---- 2. Validate slot -----------------------------------------------
        if slot_idx >= self.n_slots or self._slot_occupied[slot_idx]:
            self._brick_status[brick_idx] = DISCARDED
            info["outcome"] = "bad_slot"
            return self._build_obs(), -0.8, self._terminated(), self._truncated(), info

        slot = self.slots[slot_idx]
        x, y, θ = slot["x"], slot["y"], slot["orientation"]

        # Execute ORIENT
        self._inner_step("ORIENT", brick_idx, θ, 0.0, 0.0)

        # ---- 3. Check overlap -----------------------------------------------
        if not self._inner._check_valid_placement(x, y, θ):
            self._inner_step("PLACE", brick_idx, θ, x, y)   # let inner env discard cleanly
            self._brick_status[brick_idx] = DISCARDED
            info["outcome"] = "overlap"
            return self._build_obs(), -0.3, self._terminated(), self._truncated(), info

        # ---- 4. Successful PLACE --------------------------------------------
        self._inner_step("PLACE", brick_idx, θ, x, y)
        self._brick_status[brick_idx]     = PLACED
        self._brick_placed_pos[brick_idx] = (x, y, θ)
        self._slot_occupied[slot_idx]     = True
        info["outcome"] = "placed"
        return self._build_obs(), 1.2, self._terminated(), self._truncated(), info

    def render(self) -> None:
        if self.render_mode == "human" and self._inner is not None:
            self._inner.render(mode="human")

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inner_step(
        self,
        op: str,
        brick_id: int,
        orientation: float,
        x: float,
        y: float,
    ) -> None:
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
            pos    = self._brick_placed_pos[i]
            status = self._brick_status[i]
            if pos is not None and status == PLACED:
                x_n = pos[0] / self.x_dim
                y_n = pos[1] / self.y_dim
                θ_n = pos[2] / 360.0
            else:
                x_n = y_n = θ_n = 0.0
            parts += [x_n, y_n, θ_n, status / 3.0]

        # Per-slot: [x_norm, y_norm, θ_norm, occupied]
        for j, slot in enumerate(self.slots):
            parts += [
                slot["x"]           / self.x_dim,
                slot["y"]           / self.y_dim,
                slot["orientation"] / 360.0,
                1.0 if self._slot_occupied[j] else 0.0,
            ]

        return np.array(parts, dtype=np.float32)

    def _terminated(self) -> bool:
        return all(s in (PLACED, DISCARDED) for s in self._brick_status)

    def _truncated(self) -> bool:
        return self._step_count >= self.max_steps
