# VLM-Unreliability Evaluation Harness — reason-3d Integration

This fork of *Text-to-Scene with Large Reasoning Models* (Reason-3D) is integrated
as one **target generator** in the VLM-evaluator reliability audit
(`vlm-unreliability`). It generates indoor scenes from text prompts, renders them
under a controlled sweep of camera/lighting factors, and emits content-perturbation
variants. The audit (separate repo) globs the rendered PNGs and parses the swept
factors out of the filenames; there is **no Python import coupling** between the two
repos.

## What was added / changed

| Path | Role |
|---|---|
| `config.py` *(modified)* | Sweep-factor lists + API config. The active factor levels for a given run are set here. |
| `generate_variants.py` | Content-perturbation variant generator (object removal + worst-match asset substitution). |
| `gen.sh` | Batch scene **generation** — 50 `build_scene/PlaceObjects.py --prompt "…" --skip-render` calls. |
| `run.sh` | Batch **render** of base scenes — `rendering/render_scene.py <scene_dir>`. |
| `run2.sh` | Emits the variant-render command list (Cartesian product of scenes × variants). |
| `run3.sh` | Materialized variant-render list (scenes × 6 variants). |

The renderer itself (`rendering/render_layout.py`, `rendering/render_scene.py`)
is built on the vendored `rendering/bpa.py` (copied from vlmunr):
`bpa.render_perspective(fit_ratio=1.0)` tight-fits the camera, each scene is
rendered as a transparent RGBA master then a solid background is composited
onto it, and walls use normal-driven back-face culling (dollhouse).
`rendering/hdri/*.exr` supplies the 8 environment maps.

## Render filename scheme

Two-phase. Blender (phase 1, `render_layout.py`) writes a transparent master;
plain-Python compositing (phase 2, `render_scene.py`) inserts the background
colour. Filenames use the cross-method **common** pitch convention (0 = top-down):

```
# transparent master (5 value tokens, dash-prefixed)
render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>.png
# composited (master + _bg-<r>-<g>-<b> suffix)
render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>_bg-<r>-<g>-<b>.png
```

e.g. `render_res-512_focal-50_pitch-0_yaw-0_env-city.png` (transparent) and
`render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png` (white).
PNGs land in `outputs/<datetime>/[<variant>/]renderings/`.

## Swept factors (`config.py`)

| Factor | Levels |
|---|---|
| `RESOLUTIONS` | 196, 224, 256, 336, 384, 448, 512, 768, 1024 |
| `BACKGROUND_COLORS` | grays 0, 65, 118, 128, 186, 204, 255 + RGB chromatic (10) |
| `FOCAL_LENGTHS` | 16, 24, 35, 50, 85, 100, 200 |
| `PITCHS` | 90, 75, 60, 45, 30, 15, 0 (native; 90 = top-down; **remapped to 0=top-down in filenames**) |
| `YAWS` | 0, 45, 90, 135, 180, 225, 270, 315 (8 azimuths) |
| `HDRIS` | city, courtyard, forest, interior, night, studio, sunrise, sunset |

A single sweep holds all non-swept factors at the baseline (res 512, focal 50,
bg **white (255,255,255)**, hdri city, top-down). Background is the swept factor
only on the baseline-camera master; every other master is composited with the
white baseline background only (OFAT design).

## Content-perturbation variants (`generate_variants.py`)

`python generate_variants.py --scene-dir results/<timestamp> [--seed 42]` writes 6
sibling variant directories, each a full scene (re-converted to `raw_blender.json`):

- **Object removal** — `variant_half`, `variant_quarter`, `variant_eighth`: keep a
  seeded random `round(n/2)`, `round(n/4)`, `round(n/8)` of the objects.
- **Worst-match asset substitution** — `variant_alt_0`, `variant_alt_2`,
  `variant_alt_4`: re-rank each object's candidate assets by ascending CLIP/embedding
  similarity to the same query and swap in the asset at worst-rank 0 / 2 / 4. Positions
  and rotations are preserved.

## Run order

```bash
python build_scene/PlaceObjects.py --prompt "<prompt>" --skip-render   # or: bash gen.sh
python generate_variants.py --scene-dir results/<timestamp>
python rendering/render_scene.py results/<timestamp>                   # base
python rendering/render_scene.py results/<timestamp>/variant_half      # variants (or run3.sh)
```

Requires Blender 4.x on `PATH`, the asset DB under `~/reason_assets/`, and an
OpenAI-compatible LLM endpoint (see `config.py`) — none of which are bundled.

## Option B additions (paper Table 1 alignment)

Factor levels were brought in line with the paper's Table 1 and the render loops
switched from a full Cartesian product to **one-factor-at-a-time (OFAT)**:

- `RESOLUTIONS` = 196, 224, 256, 336, 384, 448, 512, 768, 1024
- `BACKGROUND_COLORS` = 7 grays (0, 65, 118, 128, 186, 204, 255) + 3 chromatic
  (red, green, blue) — 10 colours; baseline bg is white (255,255,255)
- `FOCAL_LENGTHS` = 16, 24, 35, 50, 85, 100, 200
- `PITCHS` = 90, 75, 60, 45, 30, 15, 0 (**pitch 90 = top-down** natively;
  remapped to 0 = top-down in filenames and when passed to bpa)
- `YAWS` = 8 azimuths at 45° steps, swept at the oblique pitch `YAW_SWEEP_PITCH`
  (45)
- `config.ofat_camera_configs()` yields the OFAT (res/focal/pitch/yaw/hdri)
  master set; backgrounds are composited only on the baseline-camera master
  (the rest get the baseline background), matching the OFAT design.

Camera-convention note: this renderer's NATIVE top-down pitch (90) is the
opposite of the bpa-based methods (genxr / HSM / LayoutVLM / IDesign, where
0 = top-down). Rendering now uses `bpa.render_perspective` (bpa convention,
0 = top-down), so a native pitch `p` is remapped with `config.native_to_common_pitch(p) = 90 - p`
both when calling bpa and in the PNG filename. Cross-method filenames therefore
all use 0 = top-down. Yaw is identical in both conventions.

## Rendering entry points (`cli.py`)

- `--prompt "…"` generates a new `outputs/<datetime>/` run (base + `--variants`).
- `--path outputs/<datetime>/` re-renders an existing run folder without
  generating (every `base` + `variant_*` with `raw_blender.json`).
- `--render` = single baseline render per scene (512/white/city/50mm/top-down/0).
- `--render-all` = full OFAT sweep per scene.
- `--prompt`/`--path` are mutually exclusive (one required);
  `--render`/`--render-all` are mutually exclusive (optional).

