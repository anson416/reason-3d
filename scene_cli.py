#!/usr/bin/env python3
"""Text-to-scene CLI for Reason-3D.

Generates a scene from a textual prompt and, optionally, four content-
perturbation variants — all into a timestamped ``outputs/<datetime>/`` folder.
A ``base/`` subfolder holds the generated scene; each variant is a sibling
subfolder. Every subfolder contains placed_objects.json,
placed_objects_data.json, prompt.txt and raw_blender.json.

Variants are pure local edits of the base scene (NO LLM / NO embedding API
calls), so the only billed generation is the single base-scene run.

Usage:
    python scene_cli.py --prompt "a cozy living room with a sofa and coffee table"
    python scene_cli.py --prompt "..." --model gpt-4o --temperature 0.7 \
        --base-url https://api.openai.com/v1 --api-key sk-... --variants
    python scene_cli.py --prompt "..." --variants --render

Output layout:
    outputs/20260708-023434/
        config.json
        base/
            placed_objects.json
            placed_objects_data.json
            prompt.txt
            raw_blender.json
            [renderings/]            # only with --render
        variant_01_half/...
        variant_02_biggest-only/...
        variant_03_scrambled/...
        variant_04_worst-object/...
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname(__file__), "build_scene")))
sys.path.insert(0, abspath(dirname(__file__)))

from config import API_KEY, BASE_URL, MODEL  # legacy defaults

import generate_variants
from llm_config import GenConfig
from PlaceObjects import generate_scene
from rendering.convert_for_blender import convert


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=dirname(abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _render(scene_dir):
    """Render a scene subfolder via render_scene.py (Blender + compositing)."""
    script_path = abspath(
        join(dirname(__file__), "rendering", "render_scene.py")
    )
    subprocess.run(["python", script_path, scene_dir], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Reason-3D scene from a textual prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="Textual scene description.")
    parser.add_argument(
        "--base-url", default=BASE_URL, help="LLM API base URL."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key (defaults to the env/config API key; never printed).",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Chat-LLM model for ALL reasoning roles (extraction, "
        "selection, constraints, order, placement, refinement).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the chat-LLM roles (0..2).",
    )
    parser.add_argument(
        "--num-objects",
        default="",
        help="Override the number of different objects to use.",
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Skip the placement refinement step.",
    )
    parser.add_argument(
        "--variants",
        action="store_true",
        help="Also generate variant_01..04 (half / biggest-only / "
        "scrambled / worst-object) from the base scene.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render PNGs (Blender) for the base (and variants). Off by "
        "default — JSON-only is fast and free.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the seeded variants (half, scrambled).",
    )
    args = parser.parse_args()

    # UTC timestamp, compact: 20260708-023434
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = join("outputs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    gen = GenConfig(
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        api_key=args.api_key if args.api_key else API_KEY,
    )

    config_record = {
        "prompt": args.prompt,
        "model": args.model,
        "temperature": args.temperature,
        "base_url": args.base_url,
        "api_key": "***" if gen.api_key else None,
        "num_objects": args.num_objects,
        "no_refinement": args.no_refinement,
        "variants": args.variants,
        "render": args.render,
        "seed": args.seed,
        "timestamp_utc": timestamp,
        "git_commit": _git_commit(),
    }
    with open(join(run_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=4)

    # 1) Base scene
    base_dir = join(run_dir, "base")
    print(f"[scene_cli] Generating base scene -> {base_dir}")
    generate_scene(
        args.prompt,
        base_dir,
        gen=gen,
        num_objects=args.num_objects,
        skip_refinement=args.no_refinement,
    )

    scene_dirs = [base_dir]

    # 2) Variants (zero-cost local edits of the base)
    if args.variants:
        print(f"[scene_cli] Generating variants -> {run_dir}")
        with open(join(base_dir, "placed_objects.json")) as f:
            placed_objects = json.load(f)
        with open(join(base_dir, "placed_objects_data.json")) as f:
            placed_objects_data = json.load(f)
        generate_variants.generate_named_variants(
            run_dir,
            placed_objects,
            placed_objects_data,
            args.prompt,
            args.seed,
        )
        for v in generate_variants.NAMED_VARIANT_DIRS:
            scene_dirs.append(join(run_dir, v))

    # 3) Optional rendering
    if args.render:
        for d in scene_dirs:
            print(f"[scene_cli] Rendering {d}")
            _render(d)

    print(f"[scene_cli] Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
