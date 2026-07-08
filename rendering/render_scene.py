"""Render a saved scene with bpa, then composite backgrounds (phase 2).

Two-phase pipeline (bpa logic, matching vlmunr's anchor rendering):

  Phase 1 (Blender/bpy): ``render_layout.py`` builds the scene and writes
    TRANSPARENT RGBA masters via ``bpa.render_perspective(fit_ratio=1.0,
    background=None)``. Filenames use the common pitch convention
    (0 == top-down): ``render_res-<r>_focal-<f>_pitch-<p>_yaw-<y>_env-<e>.png``.

  Phase 2 (plain Python + PIL): composite each master onto a solid background
    colour (``utils.helpers.add_bg_to_rgba`` -- byte-for-byte bpa's
    ``add_bg_to_rgba``), writing ``..._bg-<r>-<g>-<b>.png``. Per the OFAT
    design, the baseline-camera master gets the full background sweep; every
    other master is composited with the baseline background only.

Usage:
    python rendering/render_scene.py <scene_dir>                     # baseline
    python rendering/render_scene.py <scene_dir> --mode ofat          # full sweep
    python rendering/render_scene.py <scene_dir> --skip-composite     # masters only
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # noqa: E402
from utils.helpers import add_bg_to_rgba  # noqa: E402

# Master filename parser (must match config.render_master_filename).
# env is the trailing token up to ".png" and must NOT itself end in a
# ``_bg-<r>-<g>-<b>`` composite suffix (else a composite would parse as a master).
_MASTER_RE = re.compile(
    r"^render_res-(\d+)_focal-(\d+)_pitch-(\d+)_yaw-(\d+)_env-(.+)\.png$"
)
_COMPOSITE_SUFFIX_RE = re.compile(r"_bg-\d+-\d+-\d+$")


def _parse_master(filename):
    """Return (res, focal, common_pitch, yaw, env) or None if not a master."""
    m = _MASTER_RE.match(filename)
    if not m:
        return None
    res, focal, pitch, yaw, env = m.groups()
    # Reject composites (env token ending in the _bg-<r>-<g>-<b> suffix).
    if _COMPOSITE_SUFFIX_RE.search(env):
        return None
    return int(res), int(focal), int(pitch), int(yaw), env


def _composite_phase(renderings_dir, mode, skip_composite):
    """Composite backgrounds onto the transparent masters written in phase 1."""
    if skip_composite:
        print(f"Skipping compositing (masters only): {renderings_dir}")
        return

    baseline = config.baseline_common_key()  # (res, focal, common_pitch, yaw, env)
    for filename in sorted(os.listdir(renderings_dir)):
        parsed = _parse_master(filename)
        if parsed is None:
            continue  # not a transparent master (e.g. an existing composite)
        res, focal, common_pitch, yaw, env = parsed
        master_path = os.path.join(renderings_dir, filename)

        key = (res, focal, common_pitch, yaw, env)
        is_baseline = key == baseline
        if mode == "ofat" and is_baseline:
            bgs = config.ofat_backgrounds()  # full sweep over baseline master
        else:
            bgs = [config.BASELINE_BG]  # white (baseline) for everyone else

        for bg in bgs:
            comp_name = config.render_composite_filename(
                res, focal, common_pitch, yaw, env, bg
            )
            comp_path = os.path.join(renderings_dir, comp_name)
            add_bg_to_rgba(master_path, comp_path, color=bg)
    print(f"Background compositing complete: {renderings_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render a saved scene (bpa).")
    parser.add_argument("scene_dir", help="Path to timestamped scene directory")
    parser.add_argument(
        "--mode",
        choices=["baseline", "ofat"],
        default="baseline",
        help="baseline = single baseline render; ofat = full OFAT sweep.",
    )
    parser.add_argument(
        "--skip-composite",
        action="store_true",
        help="Skip background compositing (keep transparent masters only).",
    )
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    blender_json = os.path.join(scene_dir, "raw_blender.json")
    if not os.path.isfile(blender_json):
        print(f"Error: {blender_json} not found")
        sys.exit(1)

    # Phase 1: Blender rendering of transparent masters.
    # Prefer a standalone `blender` binary; otherwise fall back to running the
    # script with the current Python (works when bpy is a module).
    render_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "render_layout.py"
    )
    import shutil as _shutil
    if _shutil.which("blender"):
        cmd = ["blender", "--background", "--python", render_script,
               "--", "--scene-dir", scene_dir, "--mode", args.mode]
    else:
        cmd = [sys.executable, render_script, "--scene-dir", scene_dir,
               "--mode", args.mode]
    subprocess.run(cmd, check=True)

    # Phase 2: background compositing (PIL).
    renderings_dir = os.path.join(scene_dir, "renderings")
    _composite_phase(renderings_dir, args.mode, args.skip_composite)


if __name__ == "__main__":
    main()
