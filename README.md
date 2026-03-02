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
