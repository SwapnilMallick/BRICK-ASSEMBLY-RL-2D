"""
plot_bricks_placed.py — Bar chart comparison of bricks placed by PPO, TRPO, A2C
==================================================================================

Reads "Placed: N" from each test_logs/<algo>_rl_<spec>.txt file and produces
one bar chart image per wall spec, saved under plots/.

Usage
-----
python3 plot_bricks_placed.py
"""

import os
import re
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALGOS = ["ppo", "a2c", "trpo"]
ALGO_LABELS = {"ppo": "PPO", "a2c": "A2C", "trpo": "TRPO"}
ALGO_COLORS = {"ppo": "#4C72B0", "a2c": "#DD8452", "trpo": "#55A868"}

SPECS = [
    ("horizontal_wall", "Horizontal Wall"),
    ("vertical_wall",   "Vertical Wall"),
    ("l_shaped_wall",   "L-Shaped Wall"),
]

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logs")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

_PLACED_RE = re.compile(r"Placed:\s*(\d+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_placed(log_path: str) -> int | None:
    """Return the number of placed bricks from a test log file, or None."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()
        m = _PLACED_RE.search(text)
        return int(m.group(1)) if m else None
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    for spec_key, spec_title in SPECS:
        placed: dict[str, int | None] = {}
        for algo in ALGOS:
            log_path = os.path.join(LOG_DIR, f"{algo}_rl_{spec_key}.txt")
            placed[algo] = _parse_placed(log_path)
            if placed[algo] is None:
                print(f"  WARNING: could not read placed count from {log_path}")

        # ---- build chart ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4.5))

        bar_w  = 0.5
        x_pos  = range(len(ALGOS))
        values = [placed[a] if placed[a] is not None else 0 for a in ALGOS]
        colors = [ALGO_COLORS[a] for a in ALGOS]
        labels = [ALGO_LABELS[a] for a in ALGOS]

        bars = ax.bar(x_pos, values, width=bar_w, color=colors, edgecolor="white",
                      linewidth=0.8, zorder=3)

        # Value labels — fixed 5-pt offset above each bar
        for bar, val in zip(bars, values):
            ax.annotate(
                str(val),
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222",
            )

        # Axes styling
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Bricks Placed", fontsize=11)
        ax.set_title(f"Bricks Placed Comparison\n{spec_title}", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(0, max(values, default=1) * 1.25)   # 25 % headroom for labels
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"bricks_placed_{spec_key}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
