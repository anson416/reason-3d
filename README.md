<div align="center">

# **Reason-3D: Text-to-Scene with Large Reasoning Models**

**Frédéric Berdoz · Luca A. Lanzendörfer · Nick Tuninga · Roger Wattenhofer**

[![arXiv](https://img.shields.io/badge/arXiv-2509.26091-b31b1b.svg)](https://arxiv.org/abs/2509.26091)

Accepted at MAR @ NeurIPS 2025 and AAAI 2026

</div>

---

## 📄 [Sample Page](https://lucala.github.io/reason-3d-demo/)

## Getting Started

### 1. Clone the Repository

Start by cloning the project files from GitHub.

### 2\. Install Dependencies

Install all the necessary Python libraries.

```bash
pip install -r requirements.txt
```

### 3\. Install Blender

This project has been tested with **Blender version 4.3.2**. You can download it from the official Blender website.

[Download Blender](https://www.blender.org/download/)

### 4\. Configure Your Settings

Open **`config.py`** to specify file paths and add your **Gemini API key**. If you're using the free tier, consider adding `time.sleep()` lines in the code to prevent rate limit errors.

### 5\. Add Your 3D Assets

Place your 3D object files into the asset directory defined in `config.py`. All filenames must be unique. The following file types are supported:

- `.fbx`
- `.obj`
- `.glb`
- `.blend`

---

## How to Use

### Step 1: Preprocessing

Run the preprocessing script to prepare your assets.

```bash
python preprocess.py
```

_Optional:_ Use the `--skip-rotation` flag to bypass automatic object alignment if your objects are already correctly oriented.

### Step 2: Build the Scene

Execute the script to have Gemini build and arrange the scene.

```bash
python build_scene/PlaceObjects.py
```

_Required Flags:_

- `--prompt [prompt]`: Input prompt

_Optional Flags:_

- `--num-objects [number]`: Override the default number of objects to be used in the scene.
- `--no-refinement`: Skip the refinement step of the placement process.

### Step 3: Rendering

The final scene will be rendered in Blender, complete with a wooden floor. The rendered images will be saved in the **`results/final_renders`** directory.

---

## Text-to-Scene CLI (`cli.py`)

A single entry point that generates a scene from a textual prompt and, optionally,
four content-perturbation variants — all into one timestamped run folder. Only
the base scene is billed (LLM + embeddings); the variants are pure local edits of
the base scene, so they cost nothing. Run `python cli.py --help` for the full
flag list; the module docstring (`python -c "import cli; help(cli)"`) documents
every external resource the method needs and how to prepare it.

```bash
python cli.py --prompt "a cozy living room with a sofa and coffee table"
```

With variants and rendering:

```bash
python cli.py --prompt "..." --variants --render \
    --model gpt-5.1-2025-11-13 --temperature 0.7 \
    --base-url https://api.openai.com/v1 --api-key sk-...
```

**Flags**

| Flag | Description |
|---|---|
| `--prompt` | (required) Textual scene description. |
| `--model` | Chat-LLM model for **all** reasoning roles (extraction, selection, constraints, order, placement, refinement). |
| `--temperature` | Sampling temperature for the chat-LLM roles (0–2). |
| `--base-url` | LLM API base URL. |
| `--api-key` | LLM API key (defaults to the env/config key; never printed). |
| `--assets` | 3D asset folder (flat `<uid>.glb` or nested `<uid>/<uid>.glb`). Default: `config.ASSETS`. |
| `--images` | Folder for preprocessing thumbnails. Default: `config.IMAGES`. |
| `--data-dir` | Dir with `embeddings.json`/`descriptions.json`/`object_data.json`/`rotation_data.json` (from `preprocess.py`). Default: repo `data/`. |
| `--num-objects` | Override the number of different objects to use. |
| `--no-refinement` | Skip the placement refinement step. |
| `--variants` | Also generate `variant_01..04` from the base scene (free). |
| `--render` | Render PNGs (Blender) for the base (and variants). Off by default — JSON-only is fast and free. |
| `--seed` | Random seed for the seeded variants (`variant_01_half`, `variant_03_scrambled`). |
| `--output-root` | Root dir for timestamped run folders (default `outputs`). |

**External resources.** Reason-3D is not self-contained: it needs (1) a 3D
asset folder (`--assets`), (2) preprocessed data tables (`--data-dir`, made by
`python preprocess.py`), (3) rendering materials shipped in the repo + Blender
(only for `--render`), and (4) an LLM/embedding API (`--api-key`/`--base-url`/
`--model`). The `cli.py` module docstring has the full checklist, including how
to download the ObjAtHor asset set (`pip install objathor` then
`python -m objathor.dataset.download_holodeck_base_data --path ~/.objathor-assets`)
and how to prepare `data/`. The CLI fails fast with a clear message if any
required resource is missing.

**Output layout** (`outputs/<UTC-datetime>/`, e.g. `outputs/20260708-023434/`):

```
outputs/20260708-023434/
├── config.json                  # prompt + all CLI configs + timestamp + git commit
├── base/                        # the generated scene
│   ├── placed_objects.json
│   ├── placed_objects_data.json
│   ├── prompt.txt
│   ├── raw_blender.json
│   └── renderings/              # only with --render
├── variant_01_half/             # ~50% of objects (seeded subset)
├── variant_02_biggest-only/     # only the largest object (by bbox volume)
├── variant_03_scrambled/        # re-positioned + re-rotated within the scene footprint
└── variant_04_worst-object/     # each asset swapped to the globally least-similar prefab
```

> **Variants do not regenerate.** `variant_01..04` fork the base scene's
> `placed_objects` / `placed_objects_data` and rewrite them locally — no LLM
> calls and no embedding API calls. `variant_04_worst-object` "hacks" the
> asset-selection sort (which normally picks the *most*-similar prefab) to pick
> the *least*-similar one, using the stored embeddings of the originally-chosen
> asset, so it needs no API at all.

> **Prerequisite:** generation requires the preprocessed `data/` files
> (`embeddings.json`, `descriptions.json`, `object_data.json`,
> `rotation_data.json`) and 3D assets (`--assets`). Run `python preprocess.py`
> once first (see the `cli.py` docstring for the ObjAtHor asset download).
> JSON-only runs (`--render` omitted) still need the `data/` files and assets
> but not Blender.
