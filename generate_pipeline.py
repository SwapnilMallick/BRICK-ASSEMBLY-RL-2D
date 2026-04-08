"""
generate_pipeline.py — Generate a pipeline diagram for the Bricklaying Simulator 2D project.
Saves to plots/pipeline.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#F7F9FC")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_YAML    = "#4A90D9"   # blue   — input spec
C_ENV     = "#2ECC71"   # green  — environment
C_AGENT   = "#9B59B6"   # purple — RL agent
C_TRAIN   = "#E67E22"   # orange — training loop
C_EVAL    = "#E74C3C"   # red    — evaluation
C_OUT     = "#1ABC9C"   # teal   — outputs
C_ARROW   = "#555555"
C_TEXT    = "white"
C_SUBTEXT = "#EEEEEE"

def box(ax, x, y, w, h, color, title, lines=None, radius=0.25):
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=1.5, edgecolor="white",
        facecolor=color, zorder=3
    )
    ax.add_patch(fancy)
    # title
    ty = y + h - 0.38 if lines else y + h / 2
    ax.text(x + w / 2, ty, title,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT, zorder=4)
    if lines:
        step = (h - 0.55) / (len(lines) + 0.5)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.72 - i * step, line,
                    ha="center", va="center",
                    fontsize=7.8, color=C_SUBTEXT, zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ARROW,
            lw=2.0,
            mutation_scale=18,
        ),
        zorder=5,
    )

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(8, 8.55, "Bricklaying Simulator 2D — RL Pipeline",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="#2C3E50")

# ---------------------------------------------------------------------------
# Row 1:  YAML Spec  →  BrickEnvRL  →  RL Agent
# ---------------------------------------------------------------------------
# [1] YAML Spec
box(ax, 0.4, 5.5, 2.8, 2.4, C_YAML, "Wall Spec (YAML)",
    lines=["• Target wall geometry",
           "• Line segments",
           "• Brick dimensions",
           "• Environment size"])

# [2] BrickEnvRL
box(ax, 4.0, 5.5, 3.8, 2.4, C_ENV, "BrickEnvRL (Gymnasium)",
    lines=["Action: MultiDiscrete [brick, x, y, θ]",
           "Obs: brick states + wall segments",
           "Reward: +1.2 / −0.3 / −0.8 / −2.1",
           "Episode: max 500 steps"])

# [3] RL Agent
box(ax, 8.9, 5.5, 3.0, 2.4, C_AGENT, "RL Agent",
    lines=["PPO  (clip_range=0.2)",
           "A2C  (RMSProp, lr=7e-4)",
           "TRPO (target_kl=0.01)",
           "Policy: MlpPolicy"])

# arrows row 1
arrow(ax, 3.2,  6.70, 4.0,  6.70)   # YAML → Env
arrow(ax, 7.8,  6.70, 8.9,  6.70)   # Env  → Agent

# ---------------------------------------------------------------------------
# Row 2 (centre):  Training Loop
# ---------------------------------------------------------------------------
box(ax, 5.5, 3.2, 6.0, 1.7, C_TRAIN, "Training Loop",
    lines=["1,000,000 timesteps  |  8 parallel environments",
           "EvalCallback every 10k steps  |  CheckpointCallback every 20k steps"])

# arrows into training loop
arrow(ax, 10.4, 5.5,  10.4, 4.9)   # Agent → Train (straight down)
arrow(ax, 5.9,  5.5,  7.5, 4.9)   # Env   → Train (diagonal down)

# feedback arrow: training loop back to env (loop)
ax.annotate("",
    xy=(4.0, 6.20), xytext=(5.5, 3.95),
    arrowprops=dict(
        arrowstyle="-|>", color=C_ARROW, lw=1.8,
        connectionstyle="arc3,rad=-0.35",
        mutation_scale=15,
    ), zorder=5)
ax.text(3.2, 5.05, "agent\ninteracts", ha="center", va="center",
        fontsize=7.5, color="#555555", style="italic")

# ---------------------------------------------------------------------------
# Row 3:  Evaluation  →  Outputs
# ---------------------------------------------------------------------------
box(ax, 4.5, 1.1, 2.6, 1.7, C_EVAL, "Evaluation",
    lines=["Test episode (deterministic)",
           "Best model checkpoint"])

box(ax, 8.2, 1.1, 5.3, 1.7, C_OUT, "Outputs",
    lines=["Learning curves  |  Bricks placed  |  Test reward plots",
           "Animated MP4 video  |  Rollout CSV logs"])

# arrows row 3
arrow(ax, 5.8, 3.2,  5.8, 2.8)    # Train → Eval (straight down)
arrow(ax, 7.1, 1.95, 8.2, 1.95)   # Eval  → Outputs

# ---------------------------------------------------------------------------
# Legend strip at bottom
# ---------------------------------------------------------------------------
legend_items = [
    (C_YAML,  "Input"),
    (C_ENV,   "Environment"),
    (C_AGENT, "Algorithm"),
    (C_TRAIN, "Training"),
    (C_EVAL,  "Evaluation"),
    (C_OUT,   "Outputs"),
]
for i, (color, label) in enumerate(legend_items):
    lx = 1.2 + i * 2.35
    rect = FancyBboxPatch((lx, 0.18), 0.38, 0.38,
                          boxstyle="round,pad=0.04",
                          facecolor=color, edgecolor="white", lw=1, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.52, 0.37, label,
            ha="left", va="center", fontsize=8.5, color="#2C3E50")

plt.tight_layout(pad=0.3)
plt.savefig("plots/pipeline.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved to plots/pipeline.png")
plt.close()
