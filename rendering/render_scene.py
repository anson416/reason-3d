"""
Re-render a saved scene with current rendering constants.

Usage:
    python rendering/render_scene.py results/<timestamp>/
    python rendering/render_scene.py results/<timestamp>/ --skip-composite
"""

import argparse
import os
import subprocess
import sys
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    BACKGROUND_COLORS,
    FOCAL_LENGTHS,
    HDRIS,
    PITCHS,
    RESOLUTIONS,
    YAWS,
)
from utils.helpers import add_bg_to_rgba


def main():
    parser = argparse.ArgumentParser(description="Re-render a saved scene.")
    parser.add_argument(
        "scene_dir", help="Path to timestamped scene directory"
    )
    parser.add_argument(
        "--skip-composite",
        action="store_true",
        help="Skip background color compositing (keep transparent PNGs only)",
    )
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    blender_json = os.path.join(scene_dir, "raw_blender.json")
    if not os.path.isfile(blender_json):
        print(f"Error: {blender_json} not found")
        sys.exit(1)

    # Phase 1: Blender rendering (transparent RGBA)
    render_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "render_layout.py"
    )
    subprocess.run(
        [
            "blender",
            "--background",
            "--python",
            render_script,
            "--",
            "--scene-dir",
            scene_dir,
        ],
        check=True,
    )

    # Phase 2: Background compositing
    if not args.skip_composite:
        renderings_dir = os.path.join(scene_dir, "renderings")
        for res, focal, pitch, yaw, hdri in product(
            RESOLUTIONS, FOCAL_LENGTHS, PITCHS, YAWS, HDRIS
        ):
            transparent_path = os.path.join(
                renderings_dir,
                f"render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png",
            )
            if not os.path.isfile(transparent_path):
                continue
            for r, g, b in BACKGROUND_COLORS:
                bg_filename = f"render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png"
                bg_path = os.path.join(renderings_dir, bg_filename)
                add_bg_to_rgba(transparent_path, bg_path, color=(r, g, b))
        print(f"Background compositing complete: {renderings_dir}")


if __name__ == "__main__":
    main()
