import sys
import importlib.util
import argparse
import numpy as np
import time
import yaml
import os
import random
import pygame
import imageio
from datetime import datetime

# load the Brick2dEnvObj from the local package path
spec = importlib.util.spec_from_file_location(
    "brick2denv",
    os.path.join(os.path.dirname(__file__), "2dEnv", "2dEnv_obj.py"),
)
brick2denv_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brick2denv_module)
Brick2dEnv = brick2denv_module.Brick2dEnvObj
ACTIONS = brick2denv_module.ACTIONS

#  function to parse CLI arguuments from spec file
def parse_args():
    parser = argparse.ArgumentParser(description="Agent demo placing bricks")
    parser.add_argument("--spec", type=str, default="spec1.yaml")
    parser.add_argument("--x_dim", type=int, default=None)
    parser.add_argument("--y_dim", type=int, default=None)
    parser.add_argument("--total_bricks", type=int, default=None)
    parser.add_argument("--brick_length", type=int, default=None)
    parser.add_argument("--brick_width", type=int, default=None)
    return parser.parse_args()

# helper to find spec file in current dir or lineSpecifications subdir
def find_spec_file(path) -> str:
    if os.path.exists(path):
        return path
    alt = os.path.join("lineSpecifications", path)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Spec file not found: {path}")


def choose_valid_target(env) -> (float, float) or None:
    # Pick a random line segment and choose a position along it (supports horizontal, vertical, diagonal)
    seg = random.choice(env.lineFigure)
    start = seg["start"]
    end = seg["end"]

    # sample a point along the segment (parameter t in [0,1])
    t = random.random()
    target_x = float(start[0] + t * (end[0] - start[0]))
    target_y = float(start[1] + t * (end[1] - start[1]))

    # ensure within brick bounds (top-left placement)
    target_x = float(max(0.0, min(target_x, env.x_dim - env.brick_length)))
    target_y = float(max(0.0, min(target_y, env.y_dim - env.brick_width)))

    # final check
    if env._check_valid_placement(target_x, target_y):
        return target_x, target_y
    return None

def animate_move(env, start_px, end_px, width_px, height_px, placed_visuals, frames=20, color=(255,100,0), writer=None) -> None:
    # animate by interpolating center positions, draw base env and overlay moving rect + placed visuals
    # render once and capture a background snapshot to avoid double-flipping
    env.render(mode="human")
    if writer is not None:
        capture_frame(env, writer)
    if env.screen is None:
        return
    background = env.screen.copy()

    for f in range(frames):
        t = (f + 1) / frames
        cx = start_px[0] * (1 - t) + end_px[0] * t
        cy = start_px[1] * (1 - t) + end_px[1] * t

        # restore static background
        env.screen.blit(background, (0, 0))

        # draw existing placed visuals (use rotated drawing so orientation is visible)
        try:
            _draw_rotated_bricks(env.screen, placed_visuals)
        except Exception:
            # fallback: draw axis-aligned rects if rotated draw fails
            for pv in placed_visuals:
                rect = pygame.Rect(int(pv["x"]), int(pv["y"]), int(pv["width"]), int(pv["height"]))
                pygame.draw.rect(env.screen, pv.get("color", (200, 50, 0)), rect)

        # draw moving brick (centered)
        rect = pygame.Rect(
            int(cx - width_px / 2),
            int(cy - height_px / 2),
            int(width_px),
            int(height_px),
        )
        pygame.draw.rect(env.screen, color, rect)
        pygame.display.flip()
        if writer is not None:
            capture_frame(env, writer)

        # handle events so window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                env.close()
                sys.exit(0)

        # higher FPS for smoother motion
        env.clock.tick(30)

def _draw_rotated_bricks(screen, placed_visuals):
    for pv in placed_visuals:
        # coordinates and sizes are stored in pixels
        x = pv["x"]
        y = pv["y"]
        length = pv["width"]
        width = pv["height"]
        orientation = pv["orientation"]
        color = pv["color"]

        angle_rad = np.radians(orientation)
        u = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
        v = np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)

        a = np.array([x, y], dtype=np.float32)
        b = a + length * u
        c = b + width * v
        d = a + width * v

        rotated_vertices = [(float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
                            (float(c[0]), float(c[1])), (float(d[0]), float(d[1]))]

        pygame.draw.polygon(screen, color, rotated_vertices)
        pygame.draw.polygon(screen, (100, 25, 0), rotated_vertices, 2)

def capture_frame(env, writer):
    if env.screen is None or writer is None:
        return
    frame = pygame.surfarray.array3d(env.screen)
    frame = np.transpose(frame, (1, 0, 2))
    try:
        writer.append_data(frame)
    except Exception as e:
        print("Warning: failed to write video frame:", e)
'''
def print_overlap_list(env):
    occupied = env._state.get("occupied", []) if env._state is not None else []
    print("Occupied intervals by segment:")
    for seg_idx, intervals in enumerate(occupied):
        print(f"  segment {seg_idx}: {intervals}")
'''
def main():

    # parse CLI arguments
    args = parse_args()

    # load spec file
    spec_path = None
    try:
        spec_path = find_spec_file(args.spec)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    with open(spec_path, "r") as f:
        yaml_spec = yaml.safe_load(f)

    # build env config from spec with CLI overrides
    env_config = {
        "x_dim": yaml_spec.get("environment", {}).get("x_dim", 100),
        "y_dim": yaml_spec.get("environment", {}).get("y_dim", 100),
        "total_bricks": yaml_spec.get("environment", {}).get("total_bricks", 10),
        "brick_length": yaml_spec.get("environment", {}).get("brick_length", 5),
        "brick_width": yaml_spec.get("environment", {}).get("brick_width", 2),
    }

    # override CLI args if provided
    if args.x_dim is not None:
        env_config["x_dim"] = args.x_dim
    if args.y_dim is not None:
        env_config["y_dim"] = args.y_dim
    if args.total_bricks is not None:
        env_config["total_bricks"] = args.total_bricks
    if args.brick_length is not None:
        env_config["brick_length"] = args.brick_length
    if args.brick_width is not None:
        env_config["brick_width"] = args.brick_width

    # build line segments from spec
    line_thickness = yaml_spec.get("thickness", 1.0)
    line_segments_spec = yaml_spec.get("line_segments", [])
    if not line_segments_spec:
        print("No line segments in spec.")
        sys.exit(1)

    # initialize env and set line segments
    env = Brick2dEnv(**env_config)
    line_segments = []
    for ls in line_segments_spec:
        line_segments.append({
            "start": np.array(ls.get("start", [0,0]), dtype=np.float32),
            "end": np.array(ls.get("end", [0,0]), dtype=np.float32),
            "thickness": line_thickness
        })
    env.lineFigure = line_segments

    # initial reset and render to create pygame screen
    obs, info = env.reset()
    env.render(mode="human")

    # prepare video writer (save into workspace videos/)
    videos_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, datetime.now().strftime("placement_%Y%m%d_%H%M%S.mp4"))
    writer = None
    try:
        writer = imageio.get_writer(video_path, fps=30)
        print(f"Recording video to: {video_path}")
    except Exception as e:
        writer = None
        print("Warning: could not create MP4 video writer. Video will not be saved.")
        print("Reason:", e)
        print("To enable MP4 output install imageio-ffmpeg and ensure ffmpeg is on PATH.")

    placed_visuals = []  # store visuals to draw (rect, color) since env doesn't draw placed bricks by default

    # read plan from spec if available; otherwise use random placement
    plan = yaml_spec.get("plan", [])
    use_plan = len(plan) > 0
    if use_plan:
        print(f"Using predefined plan with {len(plan)} placements")
    else:
        print("No plan in spec; using random placement")

    # iterate over plans or all bricks
    placement_list = plan if use_plan else [{"brickID": i} for i in range(env.total_bricks)]

    for entry in placement_list:
        brickID = entry["brickID"]
        if brickID >= env.total_bricks:
            print(f"Skipping brick {brickID} (exceeds total_bricks={env.total_bricks})")
            continue

        # pick
        pick_action = {
            "operation": ACTIONS["PICK"],
            "brickID": brickID,
            "orientation": np.array([0.0], dtype=np.float32),
            "x": np.array([0.0], dtype=np.float32),
            "y": np.array([0.0], dtype=np.float32),
        }
        obs, reward, terminated, truncated, info = env.step(pick_action)
        env.render(mode="human")
        time.sleep(0.1)

        # determine orientation: from plan or random
        if use_plan and "orientation" in entry:
            orientation = float(entry["orientation"])
        else:
            orientation = float(random.uniform(0.0, 360.0))
        print(f"Brick {brickID}: orientation = {orientation}")
        orient_action = {
            "operation": ACTIONS["ORIENT"],
            "brickID": brickID,
            "orientation": np.array([orientation], dtype=np.float32),
            "x": np.array([0.0], dtype=np.float32),
            "y": np.array([0.0], dtype=np.float32),
        }
        obs, reward, terminated, truncated, info = env.step(orient_action)
        env.render(mode="human")
        time.sleep(0.05)

        # determine target: from plan or random
        if use_plan and "x" in entry and "y" in entry:
            target_x = float(entry["x"])
            target_y = float(entry["y"])
            target = (target_x, target_y)
            print(f"Placing at planned location: ({target_x}, {target_y})")
        else:
            # choose a valid target; retry a few times if needed
            target = None
            for _ in range(20):
                candidate = choose_valid_target(env)
                if candidate is not None:
                    target = candidate
                    break
            if target is None:
                # fallback: place anywhere within x bounds at line y
                seg = env.lineFigure[0]
                x_min = float(min(seg["start"][0], seg["end"][0]))
                x_max = float(max(seg["start"][0], seg["end"][0]))
                target_x = float(max(0.0, min((x_min + x_max) / 2.0, env.x_dim - env.brick_length)))
                target_y = float(seg["start"][1])
                target = (target_x, target_y)

        target_x, target_y = target

        # compute animation start (inventory top-left area used by env._draw_inventory_bricks)
        inventory_x = 5
        inventory_y = 5
        brick_length_px = env.brick_length * env.pixel_scale
        brick_width_px = env.brick_width * env.pixel_scale

        # start center: use top-left inventory slot center
        start_center_px = (inventory_x + brick_length_px / 2, inventory_y + brick_width_px / 2)
        # end center: world target center in pixels
        end_center_px = (
            target_x * env.pixel_scale + brick_length_px / 2,
            target_y * env.pixel_scale + brick_width_px / 2,
        )

        # animate move (overlay)
        animate_move(
            env,
            start_center_px,
            end_center_px,
            brick_length_px,
            brick_width_px,
            placed_visuals,
            frames=20,
            color=(255, 100, 0),
            writer=writer,
        )

        # finally, issue place action to update env state
        place_action = {
            "operation": ACTIONS["PLACE"],
            "brickID": brickID,
            "orientation": np.array([orientation], dtype=np.float32),
            "x": np.array([target_x], dtype=np.float32),
            "y": np.array([target_y], dtype=np.float32),
        }
        obs, reward, terminated, truncated, info = env.step(place_action)
        #print(f"Brick {brickID}: place reward = {reward}")
        #print_overlap_list(env)

        
        # register a permanent visual so future renders show placed bricks with proper rotation
        # store pixel coordinates so drawing helpers can use them directly
        placed_visuals.append({
            "x": target_x * env.pixel_scale,
            "y": target_y * env.pixel_scale,
            "width": brick_length_px,
            "height": brick_width_px,
            "orientation": orientation,
            "color": (200, 50, 0)
        })

        # draw base + placed visuals
        env.render(mode="human")
        _draw_rotated_bricks(env.screen, placed_visuals)
        pygame.display.flip()
        # write frame to video
        capture_frame(env, writer)

        # small pause between placements
        time.sleep(0.12)

        # check for early termination (user closed window)
        if env.screen is None:
            break

    print("Agent demo finished — saving video and exiting.")
    # render a few final frames so the video has trailing frames and will flush
    try:
        for _ in range(5):
            env.render(mode="human")
            _draw_rotated_bricks(env.screen, placed_visuals)
            pygame.display.flip()
            capture_frame(env, writer)
            time.sleep(0.05)
    except Exception:
        pass

    # close writer if still open
    if writer is not None:
        try:
            writer.close()
            print(f"Video saved to: {video_path}")
        except Exception as e:
            print("Warning: failed to close video writer:", e)

    env.close()
    return

if __name__ == "__main__":
    main()
