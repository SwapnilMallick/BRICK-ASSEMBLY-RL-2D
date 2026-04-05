"""
Sample command-line invocations:
python demo_visualization.py --spec spec1.yaml
python demo_visualization.py --spec spec2.yaml --total_bricks 15
python demo_visualization.py --spec spec3.yaml --x_dim 80 --y_dim 80
"""

import sys
import importlib.util
import argparse
import numpy as np
import time
import yaml
import os

spec = importlib.util.spec_from_file_location("brick2denv", 
                                              "c:\\Users\\iamsw\\Desktop\\LEGO_PLANNING_LEARNING\\2dEnv\\2dEnv_obj.py")
brick2denv_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brick2denv_module)

Brick2dEnv = brick2denv_module.Brick2dEnvObj


def main():
    
    parser = argparse.ArgumentParser(
        description="Visualize the LEGO Brick Placement Environment with YAML specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,  
    )
    
    parser.add_argument("--spec", type=str, default="spec1.yaml",
                        help="YAML specification file for line segments (default: spec1.yaml)")
    parser.add_argument("--x_dim", type=int, default=None, 
                        help="World width dimension (overrides spec file, optional)")
    parser.add_argument("--y_dim", type=int, default=None,
                        help="World height dimension (overrides spec file, optional)")
    parser.add_argument("--total_bricks", type=int, default=None,
                        help="Total number of bricks (overrides spec file, optional)")
    parser.add_argument("--brick_length", type=int, default=None,
                        help="Brick length dimension (overrides spec file, optional)")
    parser.add_argument("--brick_width", type=int, default=None,
                        help="Brick width dimension (overrides spec file, optional)")
    
    args = parser.parse_args()
    
    
    spec_file = args.spec
    if not os.path.exists(spec_file):
        
        spec_file = os.path.join("lineSpecifications", args.spec)
    
    if not os.path.exists(spec_file):
        print(f"Error: Specification file '{args.spec}' not found!")
        print(f"Looked in: {args.spec} and lineSpecifications/{args.spec}")
        sys.exit(1)
    
    try:
        with open(spec_file, 'r') as f:
            yaml_spec = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        sys.exit(1)
    
    
    env_config = {
        "x_dim": yaml_spec.get("environment", {}).get("x_dim", 100),
        "y_dim": yaml_spec.get("environment", {}).get("y_dim", 100),
        "total_bricks": yaml_spec.get("environment", {}).get("total_bricks", 10),
        "brick_length": yaml_spec.get("environment", {}).get("brick_length", 5),
        "brick_width": yaml_spec.get("environment", {}).get("brick_width", 2),
    }
    
    
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
    

    line_thickness = yaml_spec.get("thickness", 1.0)
    line_segments_spec = yaml_spec.get("line_segments", [])
    
    if not line_segments_spec:
        print("Error: No line segments found in specification file!")
        sys.exit(1)
    
    print("=" * 60)
    print("LEGO Brick Placement Environment - Visualization")
    print("=" * 60)
    print(f"Specification File: {spec_file}")
    if "name" in yaml_spec:
        print(f"Name: {yaml_spec['name']}")
    if "description" in yaml_spec:
        print(f"Description: {yaml_spec['description']}")
    print()
    print(f"Configuration:")
    print(f"  World dimensions: {env_config['x_dim']} x {env_config['y_dim']}")
    print(f"  Total bricks: {env_config['total_bricks']}")
    print(f"  Brick dimensions: {env_config['brick_length']} x {env_config['brick_width']}")
    print(f"  Target line segments: {len(line_segments_spec)}")
    for i, line_seg in enumerate(line_segments_spec, 1):
        start = line_seg.get("start", [0, 0])
        end = line_seg.get("end", [0, 0])
        print(f"    Segment {i}: ({start[0]}, {start[1]}) to ({end[0]}, {end[1]})")
    print(f"  Line thickness: {line_thickness}")
    print("=" * 60)
    
    print("\nInitializing LEGO Brick Placement Environment...")
    env = Brick2dEnv(**env_config)
    
    line_segments = []
    for line_seg in line_segments_spec:
        line_segments.append({
            "start": np.array(line_seg.get("start", [0, 0]), dtype=np.float32),
            "end": np.array(line_seg.get("end", [0, 0]), dtype=np.float32),
            "thickness": line_thickness
        })
    
    env.lineFigure = line_segments
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print(f"Environment reset. Info: {info}")
    print(f"Inventory bricks: {info['bricks_in_inventory']}")
    print(f"Placed bricks: {info['placed_bricks']}")
    
    print("\nRendering environment... (window will open)")
    print("Close the window to continue")
    
    for i in range(300):  
        env.render(mode="human")
        if env.screen is None:
            break
    
    env.close()
    print("Demo complete!")

if __name__ == "__main__":
    main()
