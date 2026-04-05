# Bricklaying 2D Environment

This repository contains a 2D prototype for planning, validating, and visualizing brick placement along target wall shapes. The project uses a Gymnasium-style environment for pick/orient/place actions, Pygame for rendering, YAML files for wall and plan definitions, and ImageIO for MP4 export.

## What the project does

- Defines target wall geometry as one or more line segments.
- Simulates a simple brick inventory with `PICK`, `ORIENT`, and `PLACE` actions.
- Supports predefined placement plans in YAML.
- Renders animated placements in a 2D window.
- Saves placement demos to `videos/placement_*.mp4`.
- Includes example target shapes for horizontal, vertical, L-shaped, triangular, and sinusoidal walls.

## Repository layout

- `2dEnv/2dEnv_obj_v2.py`: Current object-based environment with rotated placement and overlap checking.
- `2dEnv/2dEnv_obj.py`: Earlier object-based environment used by some demo variants.
- `2dEnv/2dEnv.py`: Array-based environment prototype.
- `lineSpecifications/`: Example wall specs, plan files, and the YAML format reference.
- `demo_agent_place_v3.py`: Best starting point for plan-driven brick placement demos.
- `demo_visualization.py`: Viewer for rendering wall targets from YAML specs.
- `videos/`: Generated demo videos.

## Recommended setup

The existing project notes target Python 3.10.

```bash
conda create --name lego-env python=3.10
conda activate lego-env
pip install -r requirements.txt
```

If your environment still reports missing imports, install these explicitly:

```bash
pip install pyyaml gymnasium
```

## Running the project

### Visualize a target wall

This opens a Pygame window and draws the wall geometry from a YAML spec.

```bash
python demo_visualization.py --spec spec_plan_triangle.yaml
```

You can also use simpler line-only specs such as:

```bash
python demo_visualization.py --spec spec1.yaml
python demo_visualization.py --spec spec2.yaml
python demo_visualization.py --spec spec4.yaml
```

### Run the planned placement demo

`demo_agent_place_v3.py` is the main scripted demo. It reads the wall definition, loads a placement plan if one exists, animates brick placement, and records a video in `videos/`.

```bash
python demo_agent_place_v3.py --spec spec_plan_horizontal.yaml
python demo_agent_place_v3.py --spec spec_plan_l_shaped.yaml
python demo_agent_place_v3.py --spec spec_plan_sinusoidal.yaml
```

If the selected YAML file does not include a `plan:` section, the demo falls back to random valid placements.

### Test overlap handling

The overlap specs intentionally contain conflicting placements so you can confirm that invalid placements are rejected.

```bash
python demo_agent_place_v3.py --spec spec_plan_horizontal_overlap.yaml
python demo_agent_place_v3.py --spec spec_plan_triangle_overlap.yaml
python demo_agent_place_v3.py --spec spec_plan_sinusoidal_overlap.yaml
```

## YAML spec format

Each spec can define the environment, the wall geometry, and an optional placement plan.

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

See `lineSpecifications/PLAN_FORMAT.md` for the detailed format description.

## Example specs included

- `spec1.yaml`: Simple horizontal line.
- `spec2.yaml`: L-shaped line figure.
- `spec3.yaml`: Rectangle outline.
- `spec4.yaml`: Triangle outline.
- `spec_plan_horizontal.yaml`: Horizontal wall with a full placement plan.
- `spec_plan_vertical.yaml`: Vertical wall with a full placement plan.
- `spec_plan_l_shaped.yaml`: L-shaped wall plan.
- `spec_plan_triangle.yaml`: Triangle wall plan with diagonal placement.
- `spec_plan_sinusoidal.yaml`: Sinusoidal wall plan.
- `*_overlap.yaml`: Variants with intentional collisions for placement validation.

## Which script to start with

- Use `demo_agent_place_v3.py` for the most complete plan-driven demo in this repo.
- `demo_agent_place_v1.py`, `demo_agent_place_v2.py`, and `demo_agent_place_v4.py` are alternative experiment scripts kept in the project.
- `demo_visualization_copy.py` is a duplicate of the visualization script.

## Current limitations

- `demo_visualization.py` and `demo_visualization_copy.py` use a hard-coded absolute import path to `2dEnv/2dEnv_obj.py`. If you move the repository, update that import path or switch the script to the same relative-loading approach used in `demo_agent_place_v3.py`.
- The task notes in `to_do_list.txt` mention Unity, PLAEX-style bricks, and other 3D work, but those assets are not part of this Python prototype repository.

## Output

Successful placement demos write timestamped videos into the `videos/` directory:

```text
videos/placement_YYYYMMDD_HHMMSS.mp4
```
