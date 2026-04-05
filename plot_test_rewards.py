"""
plot_test_rewards.py — Bar chart comparison of PPO, TRPO, A2C test rewards
===========================================================================

Reads total reward from each test_logs/<algo>_rl_<spec>.txt file and produces
one bar chart image per wall spec, saved under plots/.

Usage
-----
python3 plot_test_rewards.py
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

_REWARD_RE = re.compile(r"Total reward\s*:\s*([+-]?\d+(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_reward(log_path: str) -> float | None:
    """Return the total reward from a test log file, or None if not found."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()
        m = _REWARD_RE.search(text)
        return float(m.group(1)) if m else None
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    for spec_key, spec_title in SPECS:
        rewards: dict[str, float | None] = {}
        for algo in ALGOS:
            log_path = os.path.join(LOG_DIR, f"{algo}_rl_{spec_key}.txt")
            rewards[algo] = _parse_reward(log_path)
            if rewards[algo] is None:
                print(f"  WARNING: could not read reward from {log_path}")

        # ---- build chart ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4.5))

        bar_w  = 0.5
        x_pos  = range(len(ALGOS))
        values = [rewards[a] if rewards[a] is not None else 0.0 for a in ALGOS]
        colors = [ALGO_COLORS[a] for a in ALGOS]
        labels = [ALGO_LABELS[a] for a in ALGOS]

        bars = ax.bar(x_pos, values, width=bar_w, color=colors, edgecolor="white",
                      linewidth=0.8, zorder=3)

        # Value labels — fixed 5-pt offset from bar tip so they never
        # overlap the x-axis tick labels or the axes top edge.
        for bar, val in zip(bars, values):
            if val is not None:
                above = val >= 0
                ax.annotate(
                    f"{val:+.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5 if above else -5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if above else "top",
                    fontsize=10, fontweight="bold", color="#222222",
                )

        # Axes styling
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Total Test Reward", fontsize=11)
        ax.set_title(f"Test Reward Comparison\n{spec_title}", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        # Add 15 % headroom above the tallest bar (and below if negative)
        # so annotated labels are never clipped by the axes boundary.
        y_min, y_max = ax.get_ylim()
        pad = 0.15 * (y_max - y_min)
        ax.set_ylim(y_min - pad if y_min < 0 else y_min, y_max + pad)

        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"reward_comparison_{spec_key}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
