# Bricklaying Simulator 2D

A 2D brick placement simulator that combines a Gymnasium-style environment with reinforcement learning. The project supports both scripted (plan-driven) brick placement demos and pure RL training using PPO, A2C, and TRPO algorithms. Wall geometry is defined via YAML spec files, placement is rendered with Pygame, and demo videos are exported as MP4 files.

## What the project does

- Defines target wall geometry as one or more line segments via YAML specs.
- Simulates a brick inventory with `PICK`, `ORIENT`, and `PLACE` actions.
- Supports scripted placement demos driven by predefined plans in YAML.
- Trains RL agents (PPO, A2C, TRPO) to place bricks on target walls through trial and error — no plan is given to the agent.
- Renders animated brick placements in a 2D Pygame window.
- Saves placement demos and RL test episodes to `videos/` as MP4 files.
- Generates learning curve and performance comparison plots across algorithms.

## Repository layout

### Environments

- [2dEnv/2dEnv_obj_v2.py](2dEnv/2dEnv_obj_v2.py): Current object-based environment. Supports rotated brick placement and overlap checking. Used by both the scripted demo and the RL wrapper.
- [2dEnv/2dEnv_obj.py](2dEnv/2dEnv_obj.py): Earlier version of the object-based environment.
- [2dEnv/2dEnv.py](2dEnv/2dEnv.py): Array-based environment prototype.
- [envWrapper/brick_env_rl.py](envWrapper/brick_env_rl.py): Gymnasium-compatible RL environment wrapping `2dEnv_obj_v2.py`. The agent observes wall geometry and brick states, and selects actions from a `MultiDiscrete` space (brick index, x bin, y bin, orientation bin). No plan is provided — the agent must discover valid placements from the reward signal alone.

### Scripted demo

- [demo_agent_place_v3.py](demo_agent_place_v3.py): Plan-driven brick placement demo. Reads a wall spec and optional placement plan from a YAML file, animates each brick sliding from the inventory to its target position, validates placements (rejecting overlaps), and records the session to `videos/`.

### RL training scripts

- [train_ppo_rl.py](train_ppo_rl.py): Trains a PPO agent on `BrickEnvRL`. Uses Stable Baselines3 with `n_steps=2048`, `batch_size=64`, `clip_range=0.2`, and an entropy coefficient of `0.05` to encourage exploration.
- [train_a2c_rl.py](train_a2c_rl.py): Trains an A2C agent. Updates every 16 steps per environment using RMSProp (`lr=7e-4`). Higher update frequency than PPO but with more variance in training curves.
- [train_trpo_rl.py](train_trpo_rl.py): Trains a TRPO agent via `sb3-contrib`. Uses conjugate gradient + line search with a hard KL divergence constraint (`target_kl=0.01`). Slower per iteration but more sample-efficient than PPO/A2C.

All three training scripts:
- Support checkpoint saving, best-model saving, and periodic evaluation.
- Log rollout metrics (episode reward, episode length, FPS) to `rollout_logs/` as CSV files.
- Run a test episode after training and save the animated result to `videos/`.

### Animation and plotting

- [brick_anim.py](brick_anim.py): Shared animation renderer used by all three RL training scripts. Animates valid placements (brick slides to target, settles as red) and invalid placements (brick slides to target, turns black, blinks, disappears).
- [plot_learning_curves.py](plot_learning_curves.py): Reads `rollout_logs/<algo>_rl_<spec>_rollout.csv` files and produces per-wall learning curve plots (episode reward vs timesteps) comparing PPO, A2C, and TRPO. Saved to `plots/`.
- [plot_bricks_placed.py](plot_bricks_placed.py): Reads `test_logs/` output and produces bar charts comparing total bricks placed per algorithm per wall type. Saved to `plots/`.
- [plot_test_rewards.py](plot_test_rewards.py): Reads `test_logs/` output and produces bar charts comparing total test episode rewards across algorithms. Saved to `plots/`.

### Wall specifications

- [lineSpecifications/](lineSpecifications/): YAML files defining wall geometry and environment parameters.
  - `spec1.yaml`: Simple horizontal line.
  - `spec2.yaml`: L-shaped line figure.
  - `spec3.yaml`: Rectangle outline.
  - `spec4.yaml`: Triangle outline.

### Output directories

- `models/`: Saved RL model checkpoints and final models, organised by algorithm (`ppo_rl/`, `a2c_rl/`, `trpo_rl/`).
- `rollout_logs/`: CSV files with per-rollout training metrics for each algorithm and wall spec.
- `test_logs/`: Text logs from post-training test episodes.
- `plots/`: Generated comparison plots (learning curves, bricks placed, test rewards).
- `videos/`: MP4 recordings of scripted demos and RL test episodes.

## Setup

The project targets Python 3.10.

```bash
conda create --name bricklaying-env python=3.10
conda activate bricklaying-env
pip install gymnasium stable-baselines3 sb3-contrib pygame imageio imageio-ffmpeg pyyaml numpy matplotlib
```

## Running the scripted demo

`demo_agent_place_v3.py` reads a wall spec and animates brick placement according to the plan defined in the YAML. If no plan is present, it falls back to random valid placements.

```bash
python demo_agent_place_v3.py --spec lineSpecifications/spec1.yaml
python demo_agent_place_v3.py --spec lineSpecifications/spec2.yaml
```

Videos are saved to `videos/placement_YYYYMMDD_HHMMSS.mp4`.

## Training RL agents

Each script accepts the same core arguments: `--spec`, `--timesteps`, `--n_envs`, `--save_dir`, and `--load_model`.

```bash
# PPO
python train_ppo_rl.py --spec lineSpecifications/spec1.yaml --timesteps 500000 --n_envs 4

# A2C
python train_a2c_rl.py --spec lineSpecifications/spec1.yaml --timesteps 500000 --n_envs 4

# TRPO
python train_trpo_rl.py --spec lineSpecifications/spec1.yaml --timesteps 500000 --n_envs 4
```

To resume from a checkpoint:

```bash
python train_ppo_rl.py --spec lineSpecifications/spec1.yaml --load_model models/ppo_rl/checkpoints/ppo_rl_brick_100000_steps
```

## Generating comparison plots

After training, generate plots from the logged data:

```bash
python plot_learning_curves.py   # learning curves per wall spec
python plot_bricks_placed.py     # bricks placed comparison
python plot_test_rewards.py      # test reward comparison
```

Output images are saved to `plots/`.

## YAML spec format

```yaml
environment:
  x_dim: 100
  y_dim: 100
  total_bricks: 10
  brick_length: 5
  brick_width: 2

line_segments:
  - start: [50, 20]
    end: [80, 20]

plan:
  - brickID: 0
    orientation: 0
    x: 50
    y: 20
```

The `plan:` section is optional. When omitted, `demo_agent_place_v3.py` uses random placement and the RL scripts train the agent to discover placements on its own.

## RL environment details

`BrickEnvRL` wraps `2dEnv_obj_v2.py` with the following interface:

**Action space:** `MultiDiscrete([N, x_bins, y_bins, 8])` where bins are derived from environment dimensions and brick size.

**Observation:** Flat float32 vector normalised to `[0, 1]` — 4 floats per brick (x, y, orientation, status) and 4 floats per wall segment (start x, start y, end x, end y), padded to 4 segments.

**Reward table:**

| Outcome | Reward |
|---|---|
| Valid placement on/near wall | +1.2 |
| Overlapping placement on/near wall | -0.3 |
| Placement off the target wall | -0.8 |
| Invalid pick | -2.1 |

**Termination:** Episode ends when all bricks are placed or discarded, or after `max_steps=500` steps.
