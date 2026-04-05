"""
BrickEnv2D — 2D Bricklaying Simulation Environment (v2)

Gymnasium-style environment for placing bricks along target wall segments.
Compatible with demo_agent_place_v1.py through v3.py.

Action sequence per brick:  PICK → ORIENT → PLACE
"""

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Action identifiers — imported by demo scripts
# ---------------------------------------------------------------------------
ACTIONS = {
    "PICK":  0,
    "ORIENT": 1,
    "PLACE": 2,
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_COL_BG          = (220, 210, 190)   # sandy background
_COL_WALL        = ( 80, 100, 200)   # target wall segments
_COL_INVENTORY   = (255, 200,  50)   # bricks waiting in inventory
_COL_INVENTORY_B = (180, 120,   0)   # inventory brick border
_COL_HELD        = (255, 160,  30)   # currently picked brick (shown in HUD)
_COL_HUD_TEXT    = ( 40,  40,  40)   # HUD label colour
_COL_GRID        = (200, 195, 185)   # faint grid lines


class Brick2dEnvObj:
    """
    2-D bricklaying environment.

    Parameters
    ----------
    x_dim        : world width  (world units)
    y_dim        : world height (world units)
    total_bricks : number of bricks in inventory at reset
    brick_length : long dimension of each brick  (world units)
    brick_width  : short dimension of each brick (world units)
    pixel_scale  : pixels per world unit (controls window size)
    """

    def __init__(
        self,
        x_dim: int = 100,
        y_dim: int = 100,
        total_bricks: int = 10,
        brick_length: int = 5,
        brick_width:  int = 2,
        pixel_scale:  int = 8,
    ) -> None:
        self.x_dim        = x_dim
        self.y_dim        = y_dim
        self.total_bricks = total_bricks
        self.brick_length = brick_length
        self.brick_width  = brick_width
        self.pixel_scale  = pixel_scale

        # Set externally by the demo before reset() is called
        self.lineFigure: list[dict] = []

        # Pygame objects — created lazily in render()
        self.screen: pygame.Surface | None = None
        self.clock:  pygame.time.Clock | None = None
        self._font:  pygame.font.Font | None = None

        # Internal simulation state
        self._state: dict | None = None
        self._current_brick: int | None = None
        self._current_orientation: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium-style interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[dict, dict]:
        """
        Reset the environment to its initial state.

        Returns
        -------
        obs  : observation dict
        info : same keys, useful for logging
        """
        self._state = {
            "bricks_in_inventory": list(range(self.total_bricks)),
            "placed_bricks":       [],
            # Per-segment list of (t_start, t_end) intervals along the segment
            "occupied":            [[] for _ in self.lineFigure],
        }
        self._current_brick       = None
        self._current_orientation = 0.0

        obs  = self._build_obs()
        info = {
            "bricks_in_inventory": list(self._state["bricks_in_inventory"]),
            "placed_bricks":       list(self._state["placed_bricks"]),
        }
        return obs, info

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one action.

        Action dict keys
        ----------------
        operation   : int  — one of ACTIONS values
        brickID     : int
        orientation : np.ndarray shape (1,), degrees
        x           : np.ndarray shape (1,), world units
        y           : np.ndarray shape (1,), world units

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        op          = int(action["operation"])
        brick_id    = int(action["brickID"])
        orientation = float(action["orientation"][0])
        x           = float(action["x"][0])
        y           = float(action["y"][0])

        reward     = 0.0
        terminated = False
        truncated  = False
        info: dict = {"operation": op, "brickID": brick_id}

        # --- PICK -------------------------------------------------------
        if op == ACTIONS["PICK"]:
            if brick_id in self._state["bricks_in_inventory"]:
                self._current_brick = brick_id
                self._state["bricks_in_inventory"].remove(brick_id)
                reward = 0.1
                info["result"] = "picked"
            else:
                reward = -0.1
                info["result"] = "error"
                info["reason"] = f"Brick {brick_id} not in inventory"

        # --- ORIENT -----------------------------------------------------
        elif op == ACTIONS["ORIENT"]:
            if self._current_brick is not None:
                self._current_orientation = orientation % 360.0
                reward = 0.05
                info["result"]      = "oriented"
                info["orientation"] = self._current_orientation
            else:
                reward = -0.1
                info["result"] = "error"
                info["reason"] = "No brick currently held"

        # --- PLACE ------------------------------------------------------
        elif op == ACTIONS["PLACE"]:
            if self._current_brick is not None:
                if self._check_valid_placement(x, y, orientation):
                    self._state["placed_bricks"].append({
                        "brickID":     self._current_brick,
                        "x":           x,
                        "y":           y,
                        "orientation": orientation,
                    })
                    self._update_occupied(x, y, orientation)
                    self._current_brick = None
                    reward = 1.0
                    info["result"] = "placed"
                    info["x"], info["y"] = x, y
                else:
                    self._current_brick = None
                    reward = -1.0
                    info["result"] = "invalid"
                    info["reason"] = "Overlap or out-of-bounds"
            else:
                reward = -0.1
                info["result"] = "error"
                info["reason"] = "No brick currently held"

        obs = self._build_obs()

        # Episode ends when there are no more bricks to place
        if (
            not self._state["bricks_in_inventory"]
            and self._current_brick is None
        ):
            terminated = True

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render the environment with Pygame."""
        if self.screen is None:
            self._init_pygame()

        if self.screen is None:  # window was closed
            return

        # Drain the event queue; set screen=None on QUIT so demos can detect it
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._draw_background()
        self._draw_grid()
        self._draw_wall_segments()
        self._draw_inventory()
        self._draw_hud()

        pygame.display.flip()

    def close(self) -> None:
        """Shut down Pygame and mark the screen as gone."""
        try:
            pygame.quit()
        except Exception:
            pass
        self.screen = None
        self.clock  = None
        self._font  = None

    # ------------------------------------------------------------------
    # Placement validation
    # ------------------------------------------------------------------

    def _check_valid_placement(
        self, x: float, y: float, orientation: float = 0.0
    ) -> bool:
        """
        Return True iff placing a brick at (x, y) with the given orientation
        is within world bounds and does not overlap any existing brick.
        """
        if self._state is None:
            return False

        # --- Bounds check (axis-aligned bounding box of rotated brick) ---
        verts = self._brick_vertices(x, y, orientation)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        if (
            min(xs) < 0.0
            or max(xs) > self.x_dim
            or min(ys) < 0.0
            or max(ys) > self.y_dim
        ):
            return False

        # --- Overlap check against all placed bricks ---------------------
        for placed in self._state["placed_bricks"]:
            if self._bricks_overlap(
                x, y, orientation,
                placed["x"], placed["y"], placed["orientation"],
            ):
                return False

        return True

    # ------------------------------------------------------------------
    # Private helpers — geometry
    # ------------------------------------------------------------------

    def _brick_vertices(
        self, x: float, y: float, orientation: float
    ) -> list[np.ndarray]:
        """Return the four corners of the brick as world-unit vectors."""
        angle_rad = np.radians(orientation)
        u = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float64)
        v = np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float64)

        origin = np.array([x, y], dtype=np.float64)
        a = origin
        b = origin + self.brick_length * u
        c = b      + self.brick_width  * v
        d = origin + self.brick_width  * v
        return [a, b, c, d]

    @staticmethod
    def _sat_axes(vertices: list[np.ndarray]) -> list[np.ndarray]:
        """Separating-axis normals for a convex polygon."""
        axes = []
        n = len(vertices)
        for i in range(n):
            edge = vertices[(i + 1) % n] - vertices[i]
            normal = np.array([-edge[1], edge[0]], dtype=np.float64)
            length = np.linalg.norm(normal)
            if length > 1e-12:
                axes.append(normal / length)
        return axes

    def _bricks_overlap(
        self,
        x1: float, y1: float, o1: float,
        x2: float, y2: float, o2: float,
    ) -> bool:
        """
        SAT-based overlap test between two axis-aligned or rotated bricks.
        Returns True if they overlap (including edge-touching), False otherwise.
        """
        verts1 = self._brick_vertices(x1, y1, o1)
        verts2 = self._brick_vertices(x2, y2, o2)
        axes   = self._sat_axes(verts1) + self._sat_axes(verts2)

        for axis in axes:
            proj1 = [float(np.dot(v, axis)) for v in verts1]
            proj2 = [float(np.dot(v, axis)) for v in verts2]
            # Strict separation check (small tolerance avoids float noise)
            if max(proj1) < min(proj2) + 0.05 or max(proj2) < min(proj1) + 0.05:
                return False  # separating axis found (or edge-touching) → no overlap

        return True  # no separating axis → overlapping

    def _update_occupied(
        self, x: float, y: float, orientation: float
    ) -> None:
        """
        Project the just-placed brick onto each wall segment and record the
        occupied parametric interval [t_start, t_end] ∈ [0, 1].

        This powers the commented-out print_overlap_list() diagnostic in demos.
        """
        cx = x + self.brick_length / 2.0
        cy = y + self.brick_width  / 2.0
        half_len = self.brick_length / 2.0

        for seg_idx, seg in enumerate(self.lineFigure):
            ax, ay = float(seg["start"][0]), float(seg["start"][1])
            bx, by = float(seg["end"][0]),   float(seg["end"][1])
            seg_len = np.hypot(bx - ax, by - ay)
            if seg_len < 1e-12:
                continue

            t = ((cx - ax) * (bx - ax) + (cy - ay) * (by - ay)) / (seg_len ** 2)
            t = float(np.clip(t, 0.0, 1.0))
            half_t = half_len / seg_len

            while len(self._state["occupied"]) <= seg_idx:
                self._state["occupied"].append([])

            self._state["occupied"][seg_idx].append(
                (t - half_t, t + half_t)
            )

    # ------------------------------------------------------------------
    # Private helpers — observation
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        if self._state is None:
            return {"bricks_in_inventory": [], "placed_bricks": []}
        return {
            "bricks_in_inventory": list(self._state["bricks_in_inventory"]),
            "placed_bricks":       list(self._state["placed_bricks"]),
            "current_brick":       self._current_brick,
            "current_orientation": self._current_orientation,
        }

    # ------------------------------------------------------------------
    # Private helpers — rendering
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()

        screen_w = int(self.x_dim * self.pixel_scale)
        screen_h = int(self.y_dim * self.pixel_scale)

        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Bricklaying Simulator 2D")
        self.clock = pygame.time.Clock()

        try:
            self._font = pygame.font.SysFont("monospace", 13)
        except Exception:
            self._font = pygame.font.Font(None, 14)

    def _draw_background(self) -> None:
        self.screen.fill(_COL_BG)

    def _draw_grid(self) -> None:
        """Draw a faint grid every 10 world units."""
        step_px = 10 * self.pixel_scale
        w = self.screen.get_width()
        h = self.screen.get_height()
        for px in range(0, w, step_px):
            pygame.draw.line(self.screen, _COL_GRID, (px, 0), (px, h), 1)
        for py in range(0, h, step_px):
            pygame.draw.line(self.screen, _COL_GRID, (0, py), (w, py), 1)

    def _draw_wall_segments(self) -> None:
        for seg in self.lineFigure:
            ax = int(seg["start"][0] * self.pixel_scale)
            ay = int(seg["start"][1] * self.pixel_scale)
            bx = int(seg["end"][0]   * self.pixel_scale)
            by = int(seg["end"][1]   * self.pixel_scale)
            thickness = max(1, int(seg.get("thickness", 1.0) * self.pixel_scale))
            pygame.draw.line(self.screen, _COL_WALL, (ax, ay), (bx, by), thickness)

    def _draw_inventory(self) -> None:
        """Draw axis-aligned inventory bricks in the top-left strip."""
        if self._state is None:
            return

        inventory  = self._state["bricks_in_inventory"]
        len_px     = int(self.brick_length * self.pixel_scale)
        wid_px     = int(self.brick_width  * self.pixel_scale)
        padding    = 3
        origin_x   = 5
        origin_y   = 5

        for i, _ in enumerate(inventory):
            ix = origin_x + i * (len_px + padding)
            iy = origin_y
            rect = pygame.Rect(ix, iy, len_px, wid_px)
            pygame.draw.rect(self.screen, _COL_INVENTORY, rect)
            pygame.draw.rect(self.screen, _COL_INVENTORY_B, rect, 1)

        # Highlight currently held brick slot in a different colour
        if self._current_brick is not None and self._font is not None:
            held_rect = pygame.Rect(origin_x, origin_y, len_px, wid_px)
            pygame.draw.rect(self.screen, _COL_HELD, held_rect)
            pygame.draw.rect(self.screen, _COL_INVENTORY_B, held_rect, 1)

    def _draw_hud(self) -> None:
        """Overlay small status text in the top-right corner."""
        if self._font is None or self._state is None:
            return

        inv_count    = len(self._state["bricks_in_inventory"])
        placed_count = len(self._state["placed_bricks"])
        held         = self._current_brick

        lines = [
            f"Inventory : {inv_count}",
            f"Placed    : {placed_count}",
            f"Held      : {held if held is not None else '—'}",
        ]
        if held is not None:
            lines.append(f"Angle     : {self._current_orientation:.1f}°")

        x_offset = self.screen.get_width() - 160
        y_offset = 6
        for line in lines:
            surf = self._font.render(line, True, _COL_HUD_TEXT)
            self.screen.blit(surf, (x_offset, y_offset))
            y_offset += 16
