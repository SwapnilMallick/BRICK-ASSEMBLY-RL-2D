"""
Compatibility shim — re-exports Brick2dEnvObj and ACTIONS from v2.
Loaded by demo_agent_place_v1.py, v2.py, v4.py, and demo_visualization.py.
"""
from pathlib import Path
import importlib.util, os

_here  = os.path.dirname(__file__)
_spec  = importlib.util.spec_from_file_location(
    "brick2denv_v2",
    os.path.join(_here, "2dEnv_obj_v2.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Brick2dEnvObj = _mod.Brick2dEnvObj
ACTIONS       = _mod.ACTIONS
