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

The renderer itself (`rendering/render_layout.py`, `rendering/render_scene.py`) is
upstream Reason-3D, lightly adapted; `rendering/hdri/*.exr` supplies the 8 environment
maps.

## Render filename scheme

Two-phase. Blender writes a transparent master, then compositing inserts the
background colour:

```
# transparent (5 value tokens)
render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png
# composited (7 value tokens — RGB between focal and pitch)
render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png
```

e.g. `render_512_50_128_128_128_90_0_city.png`. PNGs land in
`results/<timestamp>/[<variant>/]renderings/`.

## Swept factors (`config.py`)

| Factor | Levels |
|---|---|
| `RESOLUTIONS` | 224, 256, 384, 448, 512, 640, 768, 1024 |
| `BACKGROUND_COLORS` | grays 0, 18, 65, 117, 128, 186, 204, 255 |
| `FOCAL_LENGTHS` | 24, 35, 50, 85, 100, 200 |
| `PITCHS` | 60, 90 (90 = top-down in this renderer's camera convention) |
| `YAWS` | 0, 30, 60, …, 330 (12 azimuths) |
| `HDRIS` | city, courtyard, forest, interior, night, studio, sunrise, sunset |

A single sweep holds all non-swept factors at the baseline (res 512, focal 50,
bg 128, hdri city, top-down).

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
- `BACKGROUND_COLORS` = 6 grays (0, 65, 128, 186, 204, 255) + 3 chromatic
  (red, green, blue)
- `FOCAL_LENGTHS` = 16, 24, 35, 50, 85, 100, 200
- `PITCHS` = 90, 75, 60, 45, 30, 15, 0 (**pitch 90 = top-down** in this
  renderer's convention; swept as tilt-from-top-down)
- `YAWS` = 8 azimuths at 45° steps, swept at the oblique pitch `YAW_SWEEP_PITCH`
  (45)
- `config.ofat_camera_configs()` yields the OFAT (res/focal/pitch/yaw/hdri)
  master set; backgrounds are composited only on the baseline-camera master
  (the rest get the baseline background), matching the OFAT design.

Two extra content-perturbation variants were added to `generate_variants.py`
(now 9 variants total):

- `variant_scramble` — relocate every object within the scene's XZ footprint
  (object set, rotation, size preserved; arrangement destroyed).
- `variant_subst_within` / `variant_subst_cross` — substitute each asset for the
  most-similar same-category instance / a dissimilar different-category asset
  (position and rotation fixed).

Camera-convention note: this renderer's top-down pitch (90) is the opposite of
the bpa-based methods (genxr / HSM / LayoutVLM / IDesign, where 0 = top-down).
Cross-method viewpoint analysis maps each method onto a common
"tilt-from-top-down" axis.

