#!/usr/bin/env python3
"""Reason-3D text-to-scene CLI.

Generate a 3D scene from a textual prompt (e.g. "a cozy living room with a
sofa, coffee table, and a lamp") and, optionally, four content-perturbation
variants — all into one timestamped ``outputs/<UTC-datetime>/`` folder. Only the
base scene is billed (LLM + embeddings); the variants are pure local edits of
the base scene, so they cost nothing and never regenerate.

------------------------------------------------------------------------
EXTERNAL LOCAL RESOURCES THIS METHOD REQUIRES
------------------------------------------------------------------------
Reason-3D is NOT self-contained. Generating (and rendering) a scene needs four
kinds of external resources on disk. Each is overridable from this CLI; if you
omit a flag the default from ``config.py`` / your environment is used, and the
program will fail loudly with the offending path if a required resource is
missing. Below is exactly what each resource is, when it is needed, how to tell
if you already have it, and how to prepare it if you do not.

1) 3D OBJECT ASSETS  (``--assets <dir>``)  — REQUIRED for everything
   The pool of 3D meshes the scene objects are drawn from. Must contain, for
   every asset, a mesh file named ``<uid>.glb`` (or ``.fbx``/``.obj``/``.blend``)
   at one of:
       <dir>/<uid>.glb                  (flat layout)
       <dir>/<uid>/<uid>.glb            (objathor nested layout)
   Used by: the renderer (``render_layout.py``) to load each object's mesh; and
   by preprocessing (``image_render.py``) to thumbnail + measure assets.
   Default: ``config.ASSETS`` — on this machine that resolves to
   ``~/.objathor-assets/2023_09_23/assets`` (≈50k ObjAtHor GLBs). If you have a
   different asset folder, pass ``--assets /path/to/your/assets``.
   CHECK YOU HAVE IT:  ``ls <dir> | head``  → should list many UIDs/dirs.
   PREPARE IT:  the asset set Reason-3D was built on is the ObjAtHor 2023-09-23
   dataset. Install the downloader and fetch it once (large, multi-GB, slow):

       pip install objathor
       python -m objathor.dataset.download_holodeck_base_data --path ~/.objathor-assets

   This populates ``~/.objathor-assets/2023_09_23/assets/<uid>/<uid>.glb`` plus
   ``annotations.json`` and a ``features/`` dir. That default location is what
   ``config.ASSETS`` already points at, so after the download no flag is needed.
   (If you instead have meshes in a flat ``<dir>/<uid>.glb`` layout, that works
   too — just pass ``--assets <dir>``.)

2) PREPROCESSED DATA TABLES  (``--data-dir <dir>``)  — REQUIRED for generation
   Four JSON files derived from the asset folder by ``python preprocess.py``:
       embeddings.json      — per-asset text embeddings (cosine-similarity
                              retrieval picks which asset each object becomes)
       descriptions.json    — per-asset physical/functional/contextual text
       object_data.json     — per-asset bounding box (center/size) + prefabName
       rotation_data.json   — per-asset alignment rotation (or [] to skip)
   Used by: object retrieval + placement + Blender conversion. Without these the
   method cannot turn a prompt into objects.
   Default: the ``data/`` dir inside the repo (gitignored — not in git).
   CHECK YOU HAVE IT:  ``ls data/``  → should show the four JSON files.
   PREPARE IT:  run the preprocessing pipeline once (needs the assets + Blender
   + a VLM API key). It renders 2 thumbnails per asset and queries a VLM, so for
   the full 50k-asset set it is very slow (hours-to-days):

       pip install -r requirements.txt        # incl. google-genai
       # Make sure a Gemini/OpenAI-compatible API key is set (see #4).
       python preprocess.py                    # writes data/*.json

   To skip the (slow, VLM-billed) rotation alignment when your assets are
   already upright:  ``python preprocess.py --skip-rotation``  (writes
   ``rotation_data.json`` = []).  If you already have a prepared ``data/`` dir
   elsewhere, point at it with ``--data-dir /path/to/data``.

3) RENDERING MATERIALS  (no flag)  — REQUIRED only for --render
   Two Blender material libraries bundled in the repo (not external):
       rendering/materials/wood_floor_worn_1k.blend/...   (floor)
       rendering/materials/beige_wall_001_1k.blend/...    (walls)
   Plus environment maps in ``rendering/hdri/*.exr``. These ship with the repo,
   so you normally do nothing. The renderer also needs either a ``blender``
   binary on PATH (tested with Blender 4.3.2 — https://www.blender.org/download/)
   OR the ``bpy`` Python module installed. Only needed when you pass --render.

4) LLM + EMBEDDING API ACCESS  (``--api-key``, ``--base-url``, ``--model``)
   The pipeline calls a chat LLM (object extraction, best-choice selection,
   constraints, placement order, per-object placement, intersection refinement)
   and an embedding model. Defaults target a ChatAnywhere OpenAI-compatible
   endpoint (``config.BASE_URL``) using the env key
   ``CHATANYWHERE_API_KEY`` / ``CA_API_KEY3``. To use any OpenAI-compatible
   provider instead, pass ``--api-key sk-... --base-url https://... --model ...``.
   Preprocessing (step 2 above) additionally uses a Google Gemini VLM for
   descriptions/rotation; set a ``GEMINI_API_KEY`` / ensure ``google-genai`` can
   authenticate for that one-time step.

------------------------------------------------------------------------
OUTPUT LAYOUT
------------------------------------------------------------------------
    outputs/20260708-023434/
    ├── config.json                  # prompt + every CLI flag + timestamp + git commit
    ├── base/                        # the generated scene
    │   ├── placed_objects.json      # name, center, rotation, size, size_after_rotation
    │   ├── placed_objects_data.json # per-object guid / size / boundsCenter
    │   ├── prompt.txt
    │   ├── raw_blender.json         # converted for the renderer
    │   ├── meshes/                  # each placed object's source mesh (.glb/.fbx/...)
    │   └── renderings/              # PNGs, only with --render / --render-all
    ├── variant_01_half/             # ~50% of objects (seeded subset), positions intact
    ├── variant_02_biggest-only/     # only the largest object (by bbox volume)
    ├── variant_03_scrambled/        # re-positioned + re-rotated within the scene footprint
    └── variant_04_worst-object/     # each asset swapped to the globally least-similar prefab

Each variant folder has the SAME layout as ``base/`` (placed_objects.json,
placed_objects_data.json, prompt.txt, raw_blender.json, meshes/, renderings/).

The four variants are forked from the base scene's saved JSON and rewritten
locally — NO LLM and NO embedding API calls — so the only billed work is the
single base-scene generation. ``variant_04_worst-object`` "hacks" the asset-
selection sort (which normally picks the MOST-similar prefab) to pick the
LEAST-similar one, computed from the stored embeddings of the chosen asset, so
it needs no API at all.

------------------------------------------------------------------------
RENDERING (bpa logic; --render / --render-all / --path)
------------------------------------------------------------------------
Rendering uses the vendored ``rendering/bpa.py`` (from vlmunr):
``bpa.render_perspective(fit_ratio=1.0)`` tight-fits the camera to the scene;
each scene is first rendered as a TRANSPARENT RGBA master, then a solid
background colour is composited onto it (``bpa``-equivalent PIL compositing).
Walls use normal-driven back-face culling (camera-facing faces transparent) so
the camera always sees into the room while far walls stay visible (dollhouse).

Exactly one of ``--prompt`` (generate a new run) or ``--path`` (re-render an
existing ``outputs/<datetime>/`` without generating) is required. At most one
of ``--render`` / ``--render-all`` may be passed:

- ``--render``     : single BASELINE render per scene — 512px, white
                     (255,255,255) bg, "city" env map, 50mm, top-down, yaw 0.
                     Writes a transparent master + a white-bg composite.
- ``--render-all`` : full OFAT sweep per scene — one-factor-at-a-time sweeps
                     over resolution {196,224,256,336,384,448,512,768,1024},
                     focal {16,24,35,50,85,100,200}, pitch {0..90}, yaw
                     {0,45,...,315} (at pitch 45), env map {8 maps}, and
                     background {10 colours}; every non-swept factor is held at
                     the baseline. Background is the swept factor only on the
                     baseline-camera master; every other master is composited
                     with the white baseline background only.
- ``--path DIR``   : skip generation; render every ``base`` + ``variant_*``
                     subfolder of DIR that has ``raw_blender.json``. Requires
                     ``--render`` or ``--render-all``.

CAMERA / FILENAME CONVENTION: this renderer stores pitch NATIVELY as 90 ==
top-down (opposite of bpa, where 0 == top-down). For cross-method filename
consistency, PNG filenames always use the COMMON convention (0 == top-down):
a native pitch p is written as (90 - p). So the top-down baseline appears as
``pitch-0`` in the filename. Yaw is identical in both conventions.

Filename scheme (COMMON pitch convention):

    # transparent master
    render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>.png
    # background-composited
    render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>_bg-<r>-<g>-<b>.png

e.g. ``render_res-512_focal-50_pitch-0_yaw-0_env-city.png`` (transparent) and
``render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png`` (white).

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
    # Minimal: generate the base scene (needs assets + data/ + API key)
    python cli.py --prompt "a cozy living room with a sofa and coffee table"

    # Override the LLM and generate all four variants too
    python cli.py --prompt "..." --variants \
        --model gpt-5.1-2025-11-13 --temperature 0.7 \
        --base-url https://api.openai.com/v1 --api-key sk-...

    # Generate + a single baseline render of the base (and variants)
    python cli.py --prompt "..." --variants --render \
        --assets ~/.objathor-assets/2023_09_23/assets --data-dir ./data

    # Full OFAT sweep of the base (and variants)
    python cli.py --prompt "..." --variants --render-all

    # Re-render an existing run folder without regenerating (baseline or sweep)
    python cli.py --path outputs/20260708-023434/ --render
    python cli.py --path outputs/20260708-023434/ --render-all
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from os.path import abspath, dirname, join

_REPO = abspath(dirname(__file__))
sys.path.insert(0, join(_REPO, "build_scene"))
sys.path.insert(0, _REPO)

import config  # noqa: E402

import generate_variants  # noqa: E402
from llm_config import GenConfig  # noqa: E402
from PlaceObjects import generate_scene  # noqa: E402


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _render(scene_dir, mode):
    """Render a scene subfolder via render_scene.py (Blender + compositing).

    ``mode`` is "baseline" (single 512/white/city/50mm/top-down/yaw0 render) or
    "ofat" (full one-factor-at-a-time sweep).
    """
    script_path = join(_REPO, "rendering", "render_scene.py")
    subprocess.run(
        ["python", script_path, scene_dir, "--mode", mode], check=True
    )


def _scene_dirs_in_path(path):
    """Return ``base`` + ``variant_*`` subdirs of ``path`` that contain a
    ``raw_blender.json`` (i.e. are renderable). Sorted base-first."""
    dirs = []
    base = join(path, "base")
    if os.path.isfile(join(base, "raw_blender.json")):
        dirs.append(base)
    for name in sorted(os.listdir(path)):
        if name.startswith("variant_"):
            d = join(path, name)
            if os.path.isdir(d) and os.path.isfile(join(d, "raw_blender.json")):
                dirs.append(d)
    return dirs


def _resolve_render_mode(args):
    """Map the mutually-exclusive --render / --render-all to a mode string
    (or None if neither was passed)."""
    if args.render:
        return "baseline"
    if args.render_all:
        return "ofat"
    return None


def _set_config_paths(args):
    """Override config.py path constants from CLI flags before any import
    that reads them. Done lazily (after import) but before the pipeline runs."""
    if args.assets is not None:
        config.ASSETS = args.assets
    if args.images is not None:
        config.IMAGES = args.images
    if args.data_dir is not None:
        data_dir = abspath(args.data_dir)
        config.DESCRIPTIONS = join(data_dir, "descriptions.json")
        config.EMBEDDINGS = join(data_dir, "embeddings.json")
        config.OBJ_DATA = join(data_dir, "object_data.json")
        config.ROTATION_DATA = join(data_dir, "rotation_data.json")
        # generate_variants imports EMBEDDINGS/OBJ_DATA by value at load time,
        # so patch its module-level bindings too (it re-reads them per call).
        generate_variants.EMBEDDINGS = config.EMBEDDINGS
        generate_variants.OBJ_DATA = config.OBJ_DATA


def _check_data_files():
    """Fail early with a clear message if the preprocessing data tables are
    missing — otherwise the user gets an opaque FileNotFoundError deep in the
    pipeline after already paying for the object-extraction LLM call."""
    missing = [
        (name, path)
        for name, path in (
            ("EMBEDDINGS", config.EMBEDDINGS),
            ("DESCRIPTIONS", config.DESCRIPTIONS),
            ("OBJ_DATA", config.OBJ_DATA),
            ("ROTATION_DATA", config.ROTATION_DATA),
        )
        if not os.path.isfile(path)
    ]
    if missing:
        lines = "\n".join(f"  - {n}: {p}" for n, p in missing)
        sys.exit(
            "Missing preprocessing data file(s):\n"
            + lines
            + "\n\nThese are produced by `python preprocess.py` (see the module "
            "docstring / README). Either run preprocessing once, or pass "
            "--data-dir <dir> pointing at an existing data/ folder."
        )


def _check_assets():
    if not os.path.isdir(config.ASSETS):
        sys.exit(
            f"Asset folder not found: {config.ASSETS}\n"
            "Pass --assets <dir>, set VLMUNR_REASON_ASSETS, or download the "
            "ObjAtHor assets (see the module docstring)."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Reason-3D scene from a textual prompt, or "
                    "re-render an existing run folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- source: exactly one of --prompt (generate) or --path (re-render) ---
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Textual scene description (generates a new run).")
    src.add_argument(
        "--path",
        help="Path to an existing outputs/<datetime>/ run folder to RE-RENDER "
             "without generating (renders every base + variant_* scene in it).",
    )
    # --- scene ---
    parser.add_argument(
        "--num-objects", default="",
        help="Override the number of different objects to use.",
    )
    parser.add_argument(
        "--no-refinement", action="store_true",
        help="Skip the placement refinement step.",
    )
    parser.add_argument(
        "--variants", action="store_true",
        help="Also generate variant_01..04 (half / biggest-only / scrambled / "
             "worst-object) from the base scene (free, no regeneration).",
    )
    # --- rendering: at most one of --render (baseline) / --render-all (sweep) ---
    rend = parser.add_mutually_exclusive_group()
    rend.add_argument(
        "--render", action="store_true",
        help="Render a single baseline PNG per scene (512px, white bg, city env, "
             "50mm, top-down, yaw 0) via bpa. Saves a transparent master + a "
             "white-bg composite.",
    )
    rend.add_argument(
        "--render-all", action="store_true",
        help="Render the full OFAT sweep per scene (resolution/focal/pitch/yaw/env"
             "/background factor sweeps) via bpa.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the seeded variants (half, scrambled).",
    )
    parser.add_argument(
        "--output-root", default="outputs",
        help="Root directory for timestamped run folders.",
    )
    # --- LLM / embedding API ---
    parser.add_argument(
        "--model", default=config.MODEL,
        help="Chat-LLM model for ALL reasoning roles (extraction, selection, "
             "constraints, order, placement, refinement).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature for the chat-LLM roles (0..2).",
    )
    parser.add_argument("--base-url", default=config.BASE_URL, help="LLM API base URL.")
    parser.add_argument(
        "--api-key", default=None,
        help="LLM API key (defaults to the env/config API key; never printed).",
    )
    # --- external local resources ---
    parser.add_argument(
        "--assets", default=None,
        help="3D asset folder (flat <uid>.glb or nested <uid>/<uid>.glb). "
             "Default: config.ASSETS (env VLMUNR_REASON_ASSETS overrides).",
    )
    parser.add_argument(
        "--images", default=None,
        help="Folder for preprocessing thumbnails. "
             "Default: config.IMAGES (env VLMUNR_REASON_IMAGES overrides).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory with embeddings.json / descriptions.json / "
             "object_data.json / rotation_data.json (produced by preprocess.py). "
             "Default: the repo's data/ dir.",
    )
    args = parser.parse_args()

    render_mode = _resolve_render_mode(args)

    # --- --path mode: no generation, just re-render existing scenes ---
    if args.path is not None:
        if render_mode is None:
            sys.exit("--path requires --render or --render-all (nothing else to do).")
        run_dir = abspath(args.path)
        if not os.path.isdir(run_dir):
            sys.exit(f"--path not found: {run_dir}")
        scene_dirs = _scene_dirs_in_path(run_dir)
        if not scene_dirs:
            sys.exit(
                f"No renderable scene subfolders (base/variant_* with "
                f"raw_blender.json) found in: {run_dir}"
            )
        print(f"[cli] Re-rendering {len(scene_dirs)} scene(s) in {run_dir} "
              f"(mode={render_mode})")
        for d in scene_dirs:
            print(f"[cli] Rendering {d}")
            _render(d, render_mode)
        print(f"[cli] Done. Rendered {len(scene_dirs)} scenes in {run_dir}")
        return

    # --- --prompt mode: generate a new run ---
    _set_config_paths(args)
    _check_assets()
    _check_data_files()

    # UTC timestamp, compact: 20260708-023434
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = join(args.output_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    gen = GenConfig(
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        api_key=args.api_key if args.api_key else config.API_KEY,
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
        "render_mode": render_mode,
        "seed": args.seed,
        "assets": config.ASSETS,
        "images": config.IMAGES,
        "data_dir": abspath(args.data_dir) if args.data_dir else join(_REPO, "data"),
        "embeddings": config.EMBEDDINGS,
        "descriptions": config.DESCRIPTIONS,
        "object_data": config.OBJ_DATA,
        "rotation_data": config.ROTATION_DATA,
        "timestamp_utc": timestamp,
        "git_commit": _git_commit(),
    }
    with open(join(run_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=4)

    # 1) Base scene
    base_dir = join(run_dir, "base")
    print(f"[cli] Generating base scene -> {base_dir}")
    generate_scene(
        args.prompt, base_dir, gen=gen,
        num_objects=args.num_objects,
        skip_refinement=args.no_refinement,
    )
    scene_dirs = [base_dir]

    # 2) Variants (zero-cost local edits of the base)
    if args.variants:
        print(f"[cli] Generating variants -> {run_dir}")
        with open(join(base_dir, "placed_objects.json")) as f:
            placed_objects = json.load(f)
        with open(join(base_dir, "placed_objects_data.json")) as f:
            placed_objects_data = json.load(f)
        generate_variants.generate_named_variants(
            run_dir, placed_objects, placed_objects_data, args.prompt, args.seed,
        )
        for v in generate_variants.NAMED_VARIANT_DIRS:
            scene_dirs.append(join(run_dir, v))

    # 3) Optional rendering
    if render_mode is not None:
        for d in scene_dirs:
            print(f"[cli] Rendering {d} (mode={render_mode})")
            _render(d, render_mode)

    print(f"[cli] Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
