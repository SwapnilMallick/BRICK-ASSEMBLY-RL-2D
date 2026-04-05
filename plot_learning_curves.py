"""
plot_learning_curves.py — Learning curve comparison of PPO, TRPO, A2C
=======================================================================

Reads ep_rew_mean vs total_timesteps from each
rollout_logs/<algo>_rl_<spec>_rollout.csv file and produces one line chart
per wall spec, saved under plots/.

Usage
-----
python3 plot_learning_curves.py
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALGOS = ["ppo", "a2c", "trpo"]
ALGO_LABELS = {"ppo": "PPO", "a2c": "A2C", "trpo": "TRPO"}
ALGO_COLORS = {"ppo": "#4C72B0", "a2c": "#DD8452", "trpo": "#55A868"}
ALGO_LS     = {"ppo": "-",      "a2c": "--",       "trpo": "-."}

SPECS = [
    ("horizontal_wall", "Horizontal Wall"),
    ("vertical_wall",   "Vertical Wall"),
    ("l_shaped_wall",   "L-Shaped Wall"),
]

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_logs")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_rollout(log_path: str) -> tuple[list[int], list[float]]:
    """Return (timesteps, ep_rew_means) lists from a rollout CSV."""
    timesteps, rewards = [], []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = int(row["total_timesteps"])
                    r = float(row["ep_rew_mean"])
                    if not (t != t or r != r):   # skip NaN rows
                        timesteps.append(t)
                        rewards.append(r)
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"  WARNING: file not found — {log_path}")
    return timesteps, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    for spec_key, spec_title in SPECS:
        fig, ax = plt.subplots(figsize=(8, 5))

        any_data = False
        for algo in ALGOS:
            log_path = os.path.join(LOG_DIR, f"{algo}_rl_{spec_key}_rollout.csv")
            timesteps, rewards = _parse_rollout(log_path)
            if not timesteps:
                print(f"  WARNING: no data for {algo} / {spec_key}")
                continue
            any_data = True
            ax.plot(
                timesteps, rewards,
                label     = ALGO_LABELS[algo],
                color     = ALGO_COLORS[algo],
                linestyle = ALGO_LS[algo],
                linewidth = 1.8,
                alpha     = 0.9,
            )

        if not any_data:
            plt.close(fig)
            continue

        # Axes styling
        ax.set_xlabel("Timesteps", fontsize=11)
        ax.set_ylabel("Mean Episode Reward (ep_rew_mean)", fontsize=11)
        ax.set_title(f"Learning Curves — {spec_title}", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=11, framealpha=0.7)

        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"learning_curves_{spec_key}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
